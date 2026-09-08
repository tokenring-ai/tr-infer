# `tr-infer serve` — OpenAI-compatible API

Source: `crates/tr-infer/src/server/` (`mod.rs` routing and responses, `http.rs` HTTP/1.1,
`chat.rs` request types and chat template, `tools.rs` function calling, `engine.rs` the generation
thread).

## Starting

```bash
./target/release/tr-infer serve --pack /opt/llm/tr-infer/flashnext-v3 \
    --host 0.0.0.0 --port 8089 --api-key SECRET --ctx 8192 \
    [--reasoning xhigh|medium|low|none] [--temp 0.6 --top-p 0.95 --top-k 20] \
    [--think-temp 1.0 --think-top-p 0.95 --think-top-k 20] [--model-name NAME] [--ple-mmap] [--kv f16] \
    [--mtp /opt/llm/tr-infer/flashnext-v3-mtp --spec-k 3] \
    [--vision /opt/llm/tr-infer/flashnext-v3-vision --image-max-tokens 4096] \
    [--moe-mass 0.95 --moe-min 1 --moe-max 10] \
    [--cache-dir /opt/llm/tr-infer/kvcache --cache-max-gib 32]
```

| flag | default | meaning |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8080` | bind address |
| `--api-key` / env `TR_API_KEY` | none | bearer token; the server refuses to start without one unless `--no-auth` |
| `--no-auth` | off | serve without a key (private interfaces only) |
| `--model-name` | pack directory name | id reported by `/v1/models` and echoed in responses |
| `--ctx` | 8192 | context window per request, prompt + completion; the KV cache is reserved for it |
| `--batch` | 256 | prompt tokens per batched prefill step |
| `--temp`, `--top-p`, `--top-k` | 0.6, 0.95, 20 | sampling when the request does not set it (model-card values) |
| `--think-temp`, `--think-top-p`, `--think-top-k` | unset | sampling inside the `<think>` block when the request does not set `reasoning_*` (unset = same as the answer) |
| `--reasoning` | `xhigh` | default reasoning effort; `xhigh` is the template's own default |
| `--mtp DIR` | off | MTP overlay pack (`trpack mtp`): speculative decoding with the model's draft head, see [Speculative decoding](#speculative-decoding) |
| `--spec-k N` | 3 | draft tokens per round with `--mtp` (1–7); the verify batch is N+1 rows |
| `--vision DIR` | off | vision encoder overlay pack (`trpack vision`): image parts in chat messages, see [Images](#images) |
| `--image-min-tokens`, `--image-max-tokens` | 8, 4096 | an image is resized to between these many tokens (32×32 pixels each); the encoder workspace is sized for the maximum (≈250 MiB per tile at 4096) |
| `--moe-mass M` | off | cumulative-mass expert routing: per token and MoE layer, take experts in router order until their mass reaches M, see [Expert routing](#expert-routing) |
| `--moe-min`, `--moe-max` | 1, 10 | fewest / most experts per token with `--moe-mass` (max ≤ 32) |
| `--moe-basis top\|all` | `top` | what M is measured against: the renormalised weights of the `--moe-max` best experts, or the softmax over all 512 |
| `--cache-dir DIR` | off | persistent prefix cache on a real disk, see [Prefix cache](#prefix-cache); `--cache-max-gib` (32), `--cache-chunk-len` (256), `--cache-snapshots message\|request`, `--cache-snapshot-min-tokens` (16), `--cache-staging-mib` (1024) |
| `--cores-per-node`, `--ple-mmap`, `--kv` | | as for `generate` |
| `-c`, `--config FILE` | search path | YAML configuration file, see below |

### Configuration file and environment

Every flag above is also a configuration key, so the command line does not have to carry the whole
setup. Sources, later winning: built-in defaults, YAML file(s), `TR_*` environment, flags. Full
reference in [CONFIG.md](CONFIG.md), annotated example in
[tr-infer.example.yaml](tr-infer.example.yaml).

```yaml
# /etc/tr-infer/config.yaml   (or $XDG_CONFIG_HOME/tr-infer/config.yaml, ./tr-infer.yaml, --config FILE)
pack: /opt/llm/tr-infer/flashnext-v3
runtime: { ctx: 60000, ple_mmap: true }
overlays: { mtp: /opt/llm/tr-infer/flashnext-v3-mtp }
spec: { k: 2 }
moe: { mass: 0.8 }
server: { host: 0.0.0.0, port: 8089 }
```

```bash
TR_API_KEY=SECRET ./target/release/tr-infer serve            # reads the file above
./target/release/tr-infer serve --print-config               # what it would run with, then exit
TR_SERVER__PORT=8090 ./target/release/tr-infer serve         # environment overrides the file
./target/release/tr-infer serve --ctx 8192                   # flags override both
```

The API key can stay out of the file: `TR_API_KEY` (or `TR_SERVER__API_KEY`) still works and
`--print-config` redacts `server.api_key`.

Startup loads the model (about 10 s), then prints one line:
`listening on http://HOST:PORT/v1  model id "..."  ctx N  auth bearer key  default reasoning XHigh`.
Requests are logged to stderr, one line each (see [Logging](#logging)).

Only one engine fits in HBM. Starting a second `serve`, a `bench`, or llama.cpp on the same box while
one is running makes the kernel swap one of them out entirely. Check `pgrep -af 'tr-infer serve'`
first, and stop a server by pid (`kill PID`), not with `pkill -f`, which also matches the shell that
runs it.

## Endpoints

| method and path | auth | response |
|---|---|---|
| `GET /health` (also `/`) | no | `{"status":"ok","model":ID,"uptime_s":N}` |
| `GET /v1/models` | yes | `{"object":"list","data":[{"id":ID,"object":"model","created":T,"owned_by":"tr-infer"}]}` |
| `GET /v1/models/{id}` | yes | the model object, or 404 |
| `POST /v1/chat/completions` | yes | chat completion (JSON) or an SSE stream |
| `OPTIONS *` | no | 204 with CORS headers (all origins, `Authorization` and `Content-Type` allowed) |

Auth is `Authorization: Bearer <key>` on everything under `/v1`; a missing or wrong key returns 401.
Errors use the OpenAI shape:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```

Types: `authentication_error` (401), `not_found_error` (404), `invalid_request_error` (400),
`server_error` (503 when the engine is unavailable or its waiting queue is full).

HTTP: keep-alive, `Content-Length` and chunked request bodies, `Expect: 100-continue`, 64 KiB header and
64 MiB body limits, 60 s socket timeouts, one thread per connection. Requests wait in a queue for the
single engine; `--max-queue` bounds waiting requests (default 128).

## Chat completions request

Unknown fields are ignored. The `model` field is accepted with any value; responses carry the served id.

| field | notes |
|---|---|
| `messages` | roles `system`, `developer`, `user`, `assistant`, `tool`. `content` may be a string, `null`, or an array of `{"type":"text","text":...}` parts. Assistant messages may carry `reasoning_content` and `tool_calls`. |
| `stream` | `true` for server-sent events |
| `stream_options.include_usage` | adds a final usage chunk to a stream |
| `max_tokens`, `max_completion_tokens` | completion cap; default and upper bound is `ctx - prompt_tokens`. A prompt that fills the context is a 400. |
| `temperature`, `top_p`, `top_k`, `seed` | answer sampling; `temperature 0` is greedy; `seed` makes a reply reproducible (one random stream covers thinking and answer) |
| `stop` | string or up to 16 strings; matched on the answer text, never inside the thinking block; partial matches are held back in streams |
| `reasoning_effort` | `xhigh` or `high` (same thing), `medium`, `low`, or `none` / `minimal` / `off` to disable thinking |
| `enable_thinking` | `true` / `false`, also accepted as `chat_template_kwargs.enable_thinking` (vLLM / SGLang convention) |
| `chat_template_kwargs.reasoning_effort` | same values as `reasoning_effort` |
| `reasoning` | object: `{"effort": ..., "enabled": bool, "temperature": f, "top_p": f, "top_k": n}` (OpenRouter-style) |
| `reasoning_temperature`, `reasoning_top_p`, `reasoning_top_k` | sampling inside the `<think>` block |
| `tools`, `tool_choice` | function calling, see below |
| `n` | default 1; up to `--max-sequences` with `--continuous`, otherwise must be 1 |

Rejected with 400: video and audio content parts, image parts without `--vision`, unsupported `n`,
`tool_choice` other than `auto` or `none`, temperature outside 0..2, `top_p` outside (0, 1], tool-call
`arguments` that are not valid JSON, a system message after the first non-system message, a prompt
longer than the context (image tokens included).

### Images

With `--vision DIR` a user message's `content` may be an array mixing text and image parts:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
  {"type": "image_url", "image_url": {"url": "https://example.com/photo.png"}}
]}
```

`image_url.url` is a base64 `data:` URI or an http(s) URL (fetched with a 30 s timeout, 32 MiB
limit; JPEG, PNG, WebP, GIF, BMP, TIFF). `{"type": "image", "image": "<url>"}` is accepted too.
Images are only allowed in user messages (the template refuses them in system messages). Each image
renders as `<|vision_start|><|image_pad|><|vision_end|>` at its place in the text, and the pad expands
into the image's tokens: the image is resized so both sides are multiples of 32 pixels, keeping its
aspect ratio, into between `--image-min-tokens` and `--image-max-tokens` tokens of 32×32 pixels
(a 640×480 photo is 300 tokens, 1024×1024 is 1024, the default cap of 4096 tokens is 4 megapixels).
Those tokens count against `--ctx` and `prompt_tokens`.

The encoder runs on the engine thread before the prefill (≈0.15 s for 640×480, 0.8 s for 1024×1024,
5 s for 3 megapixels; the f32 attention dominates for large images). Its output is cached by the
content hash of the preprocessed image (16 entries), so a multi-turn chat that resends the same image
encodes it once, and prefix reuse works across turns when the images match. `usage.prompt_tokens`
includes the image tokens; the log line shows `N images (E encoded, S s)`.

### Thinking

The generation prompt ends with `<think>\n` when thinking is on and with an empty closed block
`<think>\n\n</think>\n\n` when it is off (exactly what the model's template does). The model then writes
reasoning until it emits the `</think>` token; everything after is the answer.

Resolution order for the mode: `chat_template_kwargs.enable_thinking` and `enable_thinking` and
`reasoning.enabled` (an explicit switch wins), then the effort from `chat_template_kwargs.reasoning_effort`,
`reasoning.effort`, `reasoning_effort`, then the server's `--reasoning`. `enable_thinking: true` with a
server default of `none` gives `xhigh`.

Effort changes the prompt: `xhigh` and `low` add the template's instruction line to the system message
("Reasoning effort is set to xhigh. Please think carefully…" / "…set to low. Keep your thinking brief…");
`medium` adds nothing. At 70 tok/s an `xhigh` reply can think for minutes; `low` or `none` is the
practical setting for interactive chat.

Reasoning comes back as `reasoning_content`: a field on the message in non-streaming replies and a delta
field in streams (the convention llama.cpp, vLLM and DeepSeek use). The `</think>` token itself and the
whitespace the model puts around it are not delivered. `usage.completion_tokens` counts reasoning and
answer tokens; `usage.completion_tokens_details.reasoning_tokens` counts the reasoning alone.

### Per-phase sampling

The thinking block and the answer can use different sampling. The thinking phase uses, in order of
precedence: the request's `reasoning_temperature` / `reasoning_top_p` / `reasoning_top_k` (flat or under
`reasoning`), the server's `--think-temp` / `--think-top-p` / `--think-top-k`, and otherwise the
request's normal `temperature` / `top_p` / `top_k`. Each of the three settings resolves independently.
The switch happens on the `</think>` token; the random stream continues, so `seed` still reproduces a
whole reply. With thinking off the settings are ignored.

Typical use: `--think-temp 1.0` server-wide so the model explores while thinking, while requests keep
their own `temperature` for the visible answer.

### Function calling

Send `tools` in the OpenAI schema (`{"type":"function","function":{"name","description","parameters"}}`).
They are rendered into the template's `<tools>` block as JSON with Python `json.dumps` formatting (the
template's `tojson`), in the order given. The model answers in its own format:

```
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

The server parses each complete block into `tool_calls[i] = {"id":"call_…","type":"function","function":{"name":…,"arguments":"<JSON string>"}}`
and sets `finish_reason: "tool_calls"`. Parameter values are typed from the tool's `parameters.properties`
schema: a `string` parameter stays text even if it looks like a number, any other declared type is
parsed as JSON (falling back to text if that fails), and an undeclared parameter is JSON if it parses as
a non-string value. Text the model writes before the first call is returned as `content` (trimmed);
`content` is `null` when there is only a call. Several calls in one reply are all returned. A block
that does not parse is passed through as text.

In streams, text before the call arrives as `content` deltas (a possible partial `<tool_call>` tag and
the whitespace before it are held back), then one delta per complete call:
`{"tool_calls":[{"index":0,"id":…,"type":"function","function":{"name":…,"arguments":"{…}"}}]}`,
then the finish chunk with `finish_reason: "tool_calls"`.

For the next turn, send the assistant message back with its `tool_calls` (`arguments` as the JSON
string, as OpenAI clients do; the server parses it before rendering) and one `tool` message per result
with `tool_call_id` and the result as `content`. Consecutive tool messages are grouped into one user
turn of `<tool_response>` blocks, as the template does.

`tool_choice`: `auto` (default) or `none` (tools are not rendered). `required` and named functions are a
400 because the template has no way to force a call. `parallel_tool_calls` is ignored (the model may
always emit several).

## Responses

Non-streaming:

```json
{"id":"chatcmpl-…","object":"chat.completion","created":T,"model":ID,
 "choices":[{"index":0,"message":{"role":"assistant","content":"…","reasoning_content":"…","tool_calls":[…]},
             "logprobs":null,"finish_reason":"stop"}],
 "usage":{"prompt_tokens":P,"completion_tokens":C,"total_tokens":P+C,"completion_tokens_details":{"reasoning_tokens":R}}}
```

`reasoning_content` and `tool_calls` are present only when non-empty. `finish_reason` is `stop`
(end-of-turn token or a stop string), `length` (cap reached), or `tool_calls`.

Streaming (`Content-Type: text/event-stream`, chunked): a first chunk with
`delta: {"role":"assistant","content":""}`, then deltas with `reasoning_content`, `content` or
`tool_calls`, a chunk with an empty delta and the `finish_reason`, optionally the usage chunk
(`choices: []`), then `data: [DONE]`.

The end-of-turn tokens are `<|im_end|>` and `<|endoftext|>`; neither is delivered.

## Behaviour worth knowing

- **Scheduling.** Serial execution is the default. `--continuous` batches independent text generations; see [continuous batching](#continuous-batching-and-multiple-choices).
- **Prefix reuse.** The engine keeps the token sequence its state covers and the last logits. A request
  whose prompt starts with exactly that sequence is prefilled from the tail only; anything else resets
  the state and, with `--cache-dir`, restores the deepest prefix the [prefix cache](#prefix-cache)
  holds, else prefills everything (about 900 tok/s). Multi-turn chats hit the in-memory path when the
  client sends `reasoning_content` back with earlier assistant turns; clients that drop it render an
  empty think block and miss the in-memory slot (the cache still resumes at the last matching
  message boundary). No partial reuse inside a chunk: the recurrent GDN state cannot be rolled back
  (speculative decoding checkpoints it per verify row instead, see above).
- **Cancellation.** A detected disconnect cancels its job at the next execution boundary; the
  connection thread peeks the socket every 250 ms while waiting and on every failed write.
- **Text delivery.** Tokens are detokenised incrementally; a token that ends mid-UTF-8-sequence is held
  until the sequence completes, so stream deltas are always valid text.
- **Context.** Every request is independent; a conversation that outgrows `--ctx` gets a 400 from the
  prompt-length check, so clients must truncate history themselves.

## Expert routing

The model routes every token to its 10 best experts per MoE layer (softmax over the 512 router
logits, top-10, weights renormalised). `--moe-mass M` replaces the fixed count by a **cumulative
router mass** policy: experts are taken in descending router order until the selected mass reaches
M, with `--moe-min` as the floor and `--moe-max` as the cap. A token with a peaked router (0.61,
0.19, 0.09, 0.05, 0.025, …) stops after a few experts; a token with a flat router runs up to the cap.
`--moe-max` may exceed 10 (limit 32) to give flat tokens more experts than the model's default. The
weights of the chosen experts are renormalised as in the fixed top-k routing, so `--moe-min 10
--moe-max 10` is bit-identical to the default.

`--moe-basis` says what M is measured against. This model's router is flat: the softmax over all
512 experts puts only ~3 % on the best expert and ~15 % on the best 10 (`bench --moe-profile` and
`generate --moe-profile` print the mean cumulative mass at every k), so with `all` the useful targets
are 0.05–0.15 and 0.95 never stops before the cap. The default `top` measures M against the
renormalised weights of the `--moe-max` best experts, the numbers the model actually multiplies
with, so 0.9 means "the experts that carry 90 % of the top-10 weight". The policy applies to
prefill, decode and the MTP draft layer alike; the log line reports the mean experts per token and
layer of each request (`… 6.31 experts/tok`). This is a speed/quality trade-off outside the model's
training regime: the output changes, so compare against the default on your own prompts before
serving with it (measurements in [NOTEBOOK.md](NOTEBOOK.md)).

## Speculative decoding

With `--mtp DIR` the model's own multi-token-prediction head (one extra full-attention layer packed
by `trpack mtp` from the BF16 safetensors checkpoint, ≈230 MiB per tile) drafts `--spec-k` tokens
after every sampled token; the main model verifies them in one batched step and accepts the longest
prefix by speculative sampling (accept with probability `min(1, p/q)`, resample from the residual on
rejection). The output distribution is the main model's, unchanged; at temperature 0 the output is the
greedy output, up to the numerical difference between the batched and the one-token path (see
[ARCHITECTURE.md](ARCHITECTURE.md#speculative-decoding-mtp)). Drafting stops at a stop id or `</think>`
so a draft never crosses a sampling-phase switch; per-phase sampling, stop strings, tools and prefix
reuse behave as without `--mtp`.

Cost and gain per round: two pool runs — the draft chain (k rows of the draft layer with their LM
head, each token sampled inside the pool from the tiles' top candidates, ≈1.2 ms per draft) and one
(k+1)-row verify step (≈1.5× a one-token step at k=3, which also refreshes the draft head's cache) —
against 1+accepted tokens. Measured on prose at temperature 0 with `--ple-mmap` (250 tokens, warm
page cache): 75 tok/s plain, 81–94 with `--spec-k 3`, 86–93 with `--spec-k 2`; the log line prints
accepted/drafted per request. Memory: the overlay plus `k+1` state checkpoints add ≈300 MiB per
tile. With the PLE table resident that is more than this box's nodes have left (every node ends at
~15.1 GiB used), the kernel swaps hot pages and speculative decoding gets *slower* than plain decoding
— so use `--mtp` together with `--ple-mmap` (or a smaller resident footprint), and watch `VmSwap` in
the startup and log lines.

## Prefix cache

`--cache-dir DIR` (config `cache.dir`) keeps the sequence state of finished requests on disk, so a
later request that shares a prefix — the same system prompt, the same conversation one turn on, an
edited last message, the same prompt after a restart — is *restored* instead of prefilled. A 4 K-token
prefix takes ~40 ms to restore against ~5.8 s to prefill, and the continued generation is
bit-identical to the un-cached run (a restore is a memcpy of the state the engine had).

What is stored, per position range, are **rows chunks**: the K/V rows and QSA block keys of up to
`--cache-chunk-len` tokens (default 256), ~26 KiB per token, each tagged with the role of its
tokens — `system` (the merged system block incl. injected instructions and tool definitions),
`user`, `reasoning` (`<think>` blocks, sent or generated), `tool` (`<tool_response>` blocks) or
`assistant` (answers incl. tool calls). A chunk never crosses a role boundary. Because the model is
recurrent, a run of rows is only resumable where the recurrent state was also saved: **snapshots**
(~115 MiB: the Gated-DeltaNet states of every layer on every tile, the PLE history, the draft head's
carry, and the logits of the position) are taken at the end of every prompt span of at least
`--cache-snapshot-min-tokens` (16) tokens and at the end of every request
(`--cache-snapshots message`, the default), or at request ends only (`request`). A request that
matches a stored prefix resumes at the deepest snapshot whose chunks are all present and prefills the
rest.

Every role has a policy (`cache.roles.<role>`, `--cache-role ROLE=KEY:VALUE,…`): `persist` (store
the role's chunks at all — a chunk that is not stored breaks the chain, so `reasoning: persist:
false` makes prefixes resumable up to the first think block only), `priority` (eviction order 0–9,
lower is evicted first, least recently used within a priority; a snapshot has the role of the span
ending at it), `ttl` (idle seconds after which the role's objects are dropped whatever the budget)
and `snapshot` (whether a prompt span of the role ends in a snapshot). Typical: keep the system
prompt longest (`system=priority:9`), expire tool results (`tool=ttl:3600,priority:1`), keep
reasoning off disk (`reasoning=persist:false`).

Objects are addressed by a chain hash of the pack, the K/V type, the overlays, the token ids and the
image contents before a position, so equal prefixes share objects across conversations, a changed
pack or `--kv` simply misses, and every chunk's stored token ids are compared with the prompt before
it is used. Eviction is least-recently-used by bytes (`--cache-max-gib`, default 32); a hit touches
the whole chain it uses. Writes happen on a separate thread from staging buffers
(`--cache-staging-mib`), through `O_DIRECT` so the page cache stays clear; the store survives a crash
(objects are renamed into place, the index is rebuilt from the object headers when missing). One
directory per pack, on a real disk — never tmpfs — e.g. `/opt/llm/tr-infer/kvcache` on this box.

`tr-infer cache --cache-dir DIR stats` prints objects, bytes and ages per root and the chunk counts
per role (with the role's policy when it is not the default); `tr-infer cache --cache-dir DIR clear`
empties the store. The startup line reports what
was found; each request logs what it restored and what it added.

## Logging

One line per request on stderr:

```
chatcmpl-…: 134 prompt tok (111 reused, 0.07 s), 47 completion tok (42 reasoning, 75.1 tok/s), Low, think temp 1 top_p 0.95 top_k 20 / answer temp 0 top_p 0.95 top_k 20, max 300, finish stop
chatcmpl-…: 54 prompt tok (0 reused, 0.10 s), 147 completion tok (144 reasoning, 96.3 tok/s), drafts 101/141 accepted, Low, …, finish stop
chatcmpl-…: tool calls get_time, get_weather
chatcmpl-…: prefix cache restored 3997 of 4004 prompt tokens (17 rows chunks, 213.1 MiB, 39.5 ms)
chatcmpl-…: 4004 prompt tok (3997 reused of which 3997 restored from the prefix cache in 40 ms, 0.09 s), …
chatcmpl-…: prefix cache saved 5 rows chunks + 1 snapshots (125.1 MiB staged in 14.4 ms)
```

`finish cancelled (client disconnected)` marks a cancelled job.

## Examples

```bash
U=http://127.0.0.1:8089; K="Authorization: Bearer SECRET"; J="Content-Type: application/json"

# stream with brief thinking, answer greedily, think at temperature 1
curl -sN $U/v1/chat/completions -H "$K" -H "$J" -d '{
  "messages":[{"role":"user","content":"Give me three words that rhyme with cat."}],
  "stream":true, "reasoning_effort":"low", "temperature":0, "reasoning_temperature":1.0}'

# no thinking, stop string, fixed seed
curl -s $U/v1/chat/completions -H "$K" -H "$J" -d '{
  "messages":[{"role":"user","content":"Count from 1 to 10, comma separated."}],
  "reasoning_effort":"none", "stop":[", 5"], "seed":7, "max_tokens":64}'

# function calling
curl -s $U/v1/chat/completions -H "$K" -H "$J" -d '{
  "messages":[{"role":"user","content":"Weather in Paris in celsius?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather",
           "parameters":{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}],
  "reasoning_effort":"low"}'
```

Python (`openai` package):

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8089/v1", api_key="SECRET")
r = c.chat.completions.create(model="flashnext-v3", messages=[{"role": "user", "content": "Hi"}],
                              extra_body={"reasoning_effort": "low", "reasoning_temperature": 1.0})
print(r.choices[0].message.reasoning_content, r.choices[0].message.content)
```

Chat front-ends (Open WebUI, Cherry Studio, LibreChat) work with the base URL and key; those that
understand `reasoning_content` show the thinking separately.

## Not implemented

`/v1/completions`, `/v1/embeddings`, logprobs, forced `tool_choice`, video and audio input,
concurrent generation, and a network backend for the prefix cache (the `Backend` trait is there; only
the filesystem implements it).

## Continuous batching and multiple choices

Continuous batching is opt-in and currently supports text without MTP or vision overlays:

```sh
tr-infer serve --pack /opt/llm/tr-infer/flashnext-v3 --no-auth \
  --continuous --ple-mmap --kv-cache-mib 1024 --max-sequences 16 \
  --batch 256 --prefill-chunk 32 --max-queue 128
```

`--kv-cache-mib` is a required **per-NUMA-tile** budget for the shared KV/QSA page pool.
Recurrent state, batch workspaces, weights, host/cache staging, and existing serial state are
additional. Startup reports model and sequence arena sizes per tile and rejects a pool that
cannot fit with 256 MiB of per-tile headroom. The default maximum is 16 resident choices;
`--batch` must be larger than `--max-sequences` so prompt processing can make progress.
The example's 1024 MiB budget is a starting configuration, not a workload-independent capacity
promise. Each admitted request reserves credits for its prompt and `max_tokens` across all choices.
Omitting an output limit reserves the remaining context window.

The scheduler executes one pending token from every decoding sequence, then up to 32 prompt
rows in the same batch. It rotates prompt work between requests. Without active decoding,
prefill may use the full batch. Arrivals and cancellations are handled between iterations.
Requests wait in FIFO order when their sequence or page reservation does not fit; a request
that can never fit is rejected with HTTP 400. The waiting-request limit defaults to 128;
HTTP 503 reports a full queue. The executor never waits for network writes. A full 256-event
output channel cancels that request, preventing a stalled client from blocking other generations.

`POST /v1/chat/completions` accepts `n` from 1 to `--max-sequences` in continuous mode:

```json
{"messages":[{"role":"user","content":"Suggest a title."}],"n":3,"max_tokens":64,"seed":42,"stream":true}
```

The prompt is processed once. Choices share completed prompt pages, copy their recurrent state,
and use copy-on-write for a shared partial page. Choice zero uses the supplied seed; choice i
uses `seed + i` with unsigned wrapping. Greedy choices may be identical. Seeds isolate random
number streams, but outputs can change with batch composition because decode, VNNI batching,
and AMX batching have different floating-point/quantization behavior.

Streaming chunks carry the appropriate `choices[].index` and each choice finishes independently.
The final usage chunk and `[DONE]` follow all choices. Nonstreaming choices are ordered by index.
Prompt tokens count once; completion and reasoning counts are summed over choices. Tool parsing,
stop matching, reasoning settings, and output limits are independent per choice.

Persistent prefix caching remains available with the existing cache options. A bounded background
reader performs lookup and disk reads; the executor imports one ready object per restoring request
between batches. Snapshots and rows retain the existing disk format. Saves are best effort when
staging, index, or writer capacity is unavailable. A failed or incomplete restore falls back to
prefill for that request. Active KV pages are never evicted: admission reserves future growth.
One completed sequence is retained for immediate continuation when the next prompt extends it.
Admission reuses that state or releases it before allocating an unrelated request.

`GET /health` includes `scheduler` counters: queued requests, resident sequences, total/free KV
pages, idle retained pages, iterations, prefill/decode rows, and the largest observed number of sequences in one batch.
These counters describe execution, not reserved capacity. Serial mode retains its vision/MTP
behavior and accepts `n=1` only.

Validation on the target machine:

```sh
cargo test --release -p tr-model --test continuous -- --ignored --nocapture --test-threads=1
python3 tools/continuous_smoke.py --url http://127.0.0.1:8080 --cache
```

The hardware tests default to `/opt/llm/tr-infer/test-l03-v4` (`TR_TEST_PACK` overrides it).
The HTTP smoke test requires at least four sequence slots and a context window of at least 2048.

A separate explicit full-model decode benchmark runs 1, 2, 4, 8, and 16 sequences:

```sh
cargo test --release -p tr-model --test continuous_bench -- --ignored --nocapture
```

It defaults to `/opt/llm/tr-infer/flashnext-v3` (`TR_BENCH_PACK` overrides it), uses file-backed PLE,
and writes `/tmp/tr-continuous-benchmark.json` (`TR_BENCH_OUTPUT` overrides it). Results include
full logits transfer and exclude sampling, network, prefill, and cache I/O.

A warmed full-model run on the dual Xeon Max 9470 system produced the following results
([raw measurements and configuration](continuous-benchmark.json)). Each sequence used a distinct
32-token prompt and token trace; 24 decode iterations were measured after replaying the full trace
once. These are short-context executor measurements, not end-to-end serving latency guarantees.

| Mode | Sequences | Aggregate tokens/s | Iteration p50 / p95 (ms) |
|---|---:|---:|---:|
| Serial | 1 | 72.7 | 13.7 / 14.0 |
| Continuous | 1 | 72.7 | 13.8 / 13.9 |
| Continuous | 2 | 112.7 | 17.6 / 19.1 |
| Continuous | 4 | 168.8 | 23.7 / 24.2 |
| Continuous | 8 | 257.9 | 30.9 / 32.2 |
| Continuous | 16 | 279.1 | 56.8 / 58.4 |
