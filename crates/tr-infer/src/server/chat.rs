//! OpenAI chat-completions request types and the model's chat template (ChatML with the
//! reasoning-effort system line and `<think>` generation prompt), transcribed from the GGUF's
//! `tokenizer.chat_template`. Image parts render as `<|vision_start|><|image_pad|><|vision_end|>`
//! and their sources are returned in order (the server expands each pad into the image's tokens).

use serde::Deserialize;
use serde_json::Value;
pub use tr_cache::{Role, Span};

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: Option<i64>,
    pub stop: Option<Stop>,
    pub n: Option<usize>,
    pub reasoning_effort: Option<String>,
    /// Sampling inside the `<think>` block (also accepted as `reasoning.{temperature,top_p,top_k}`)
    pub reasoning_temperature: Option<f32>,
    pub reasoning_top_p: Option<f32>,
    pub reasoning_top_k: Option<usize>,
    pub enable_thinking: Option<bool>,
    pub chat_template_kwargs: Option<TemplateKwargs>,
    pub reasoning: Option<ReasoningParam>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
}

#[derive(Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize)]
pub struct TemplateKwargs {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
}

#[derive(Deserialize)]
pub struct ReasoningParam {
    pub enabled: Option<bool>,
    pub effort: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Stop::One(s) => vec![s],
            Stop::Many(v) => v,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thinking {
    Off,
    XHigh,
    Medium,
    Low,
}

impl Thinking {
    /// Values accepted for `reasoning_effort` (server flag and request): the template's
    /// xhigh/medium/low, OpenAI's high (= xhigh), and none/minimal/off to disable thinking.
    pub fn parse(s: &str) -> Option<Thinking> {
        Some(match s.to_ascii_lowercase().as_str() {
            "xhigh" | "high" => Thinking::XHigh,
            "medium" => Thinking::Medium,
            "low" => Thinking::Low,
            "none" | "minimal" | "off" | "false" => Thinking::Off,
            _ => return None,
        })
    }
    fn instructions(self) -> &'static str {
        match self {
            Thinking::XHigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            Thinking::Low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            Thinking::Medium | Thinking::Off => "",
        }
    }
}

/// Resolve the thinking mode from the request fields, most specific first:
/// `chat_template_kwargs.{enable_thinking,reasoning_effort}`, `enable_thinking`,
/// `reasoning.{enabled,effort}`, `reasoning_effort`, then the server default.
pub fn resolve_thinking(r: &ChatRequest, default: Thinking) -> Result<Thinking, String> {
    let eff = |s: &Option<String>| -> Result<Option<Thinking>, String> {
        match s {
            Some(v) => Thinking::parse(v).map(Some).ok_or_else(|| format!("unexpected reasoning effort {v:?}: supported are xhigh (default), high, medium, low, none")),
            None => Ok(None),
        }
    };
    let on_default = |d: Thinking| if d == Thinking::Off { Thinking::XHigh } else { d };
    // effort as given anywhere (kwargs > reasoning.effort > reasoning_effort), else the default
    let mut mode = eff(&r.chat_template_kwargs.as_ref().and_then(|k| k.reasoning_effort.clone()))?
        .or(eff(&r.reasoning.as_ref().and_then(|p| p.effort.clone()))?)
        .or(eff(&r.reasoning_effort)?)
        .unwrap_or(default);
    // explicit on/off switches
    let switch = r.chat_template_kwargs.as_ref().and_then(|k| k.enable_thinking).or(r.enable_thinking).or(r.reasoning.as_ref().and_then(|p| p.enabled));
    match switch {
        Some(false) => mode = Thinking::Off,
        Some(true) => mode = on_default(mode),
        None => {}
    }
    Ok(mode)
}

const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Image placeholder the template emits per image part; the server expands the pad token.
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// The source of an image part: `image_url.url` (data URI or http(s)), or `image` (a URL string).
fn image_source(it: &Value) -> Option<String> {
    if let Some(u) = it.get("image_url") {
        return u.get("url").and_then(|x| x.as_str()).or_else(|| u.as_str()).map(|s| s.to_string());
    }
    it.get("image").and_then(|x| x.as_str()).map(|s| s.to_string())
}

fn render_content(v: &Value, images: &mut Vec<String>, allow_images: bool) -> Result<String, String> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(items) => {
            let mut out = String::new();
            for it in items {
                let ty = it.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if let Some(t) = it.get("text").and_then(|t| t.as_str()) {
                    out.push_str(t);
                } else if let Some(src) = image_source(it) {
                    if !allow_images {
                        return Err("System message cannot contain images.".into());
                    }
                    images.push(src);
                    out.push_str(IMAGE_PLACEHOLDER);
                } else if it.get("video").is_some() || ty.starts_with("video") || ty.starts_with("input_audio") {
                    return Err("video and audio content is not supported".into());
                } else if ty.starts_with("image") {
                    return Err("image part needs image_url.url (data: URI or http(s) URL)".into());
                } else {
                    return Err("unexpected content item; only text and image parts are supported".into());
                }
            }
            Ok(out)
        }
        _ => Err("unexpected content type".into()),
    }
}

/// Apply the chat template with `add_generation_prompt=true`. Returns the prompt text and whether
/// the generation starts inside an open `<think>` block.
pub fn render_prompt(msgs: &[Message], mode: Thinking, tools: &[Value]) -> Result<(String, bool), String> {
    let (p, open, images) = render_prompt_images(msgs, mode, tools)?;
    if !images.is_empty() {
        return Err("image content needs the vision encoder (--vision)".into());
    }
    Ok((p, open))
}

/// `render_prompt` that also returns the image sources in prompt order (one placeholder each).
pub fn render_prompt_images(msgs: &[Message], mode: Thinking, tools: &[Value]) -> Result<(String, bool, Vec<String>), String> {
    let r = render_prompt_spans(msgs, mode, tools)?;
    Ok((r.text, r.thinking_open, r.images))
}

/// A rendered prompt with the role of every byte: `segs` are `(byte offset, role)` starts, in
/// order, the first at 0; a segment runs to the next start (or the end of the text).
pub struct Rendered {
    pub text: String,
    pub thinking_open: bool,
    pub images: Vec<String>,
    pub segs: Vec<(usize, Role)>,
}

/// Roles of the rendered prompt, for the prefix cache: the merged system block (with the
/// injected instructions and tool definitions) is `System`; a `<tool_response>` block with its
/// wrapper is `Tool`; an assistant message is `Reasoning` from `<|im_start|>assistant` through
/// `</think>` and `Assistant` after it; the generation tail is `Reasoning` when thinking is on.
pub fn render_prompt_spans(msgs: &[Message], mode: Thinking, tools: &[Value]) -> Result<Rendered, String> {
    use crate::server::tools::py_json;
    if msgs.is_empty() {
        return Err("No messages provided.".into());
    }
    let mut images = Vec::new();
    let mut out = String::new();
    let mut segs: Vec<(usize, Role)> = Vec::new();
    // leading system/developer messages are merged into one system block
    let mut num_sys = 0;
    let mut merged = String::new();
    for m in msgs {
        if m.role == "system" || m.role == "developer" {
            let c = render_content(&m.content, &mut images, false)?;
            let c = c.trim();
            if !c.is_empty() {
                if !merged.is_empty() {
                    merged.push('\n');
                }
                merged.push_str(c);
            }
            num_sys += 1;
        } else {
            break;
        }
    }
    let instr = mode.instructions();
    if !tools.is_empty() || !merged.is_empty() || !instr.is_empty() {
        segs.push((out.len(), Role::System));
    }
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !instr.is_empty() {
            out.push_str(instr);
            out.push_str("\n\n");
        }
        out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            out.push('\n');
            py_json(t, &mut out);
        }
        out.push_str("\n</tools>");
        out.push_str(TOOL_INSTRUCTIONS);
        if !merged.is_empty() {
            out.push_str("\n\n");
            out.push_str(&merged);
        }
        out.push_str("<|im_end|>\n");
    } else if !merged.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !instr.is_empty() {
            out.push_str(instr);
            out.push_str("\n\n");
        }
        out.push_str(&merged);
        out.push_str("<|im_end|>\n");
    } else if !instr.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(instr);
        out.push_str("<|im_end|>\n");
    }
    for (i, m) in msgs.iter().enumerate().skip(num_sys) {
        let c = render_content(&m.content, &mut images, m.role == "user")?;
        let content = c.trim();
        match m.role.as_str() {
            "system" | "developer" => return Err("System message must be at the beginning.".into()),
            "user" => {
                segs.push((out.len(), Role::User));
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                let reasoning = m.reasoning_content.as_deref().unwrap_or("").trim();
                segs.push((out.len(), Role::Reasoning));
                out.push_str("<|im_start|>assistant\n<think>\n");
                out.push_str(reasoning);
                out.push_str("\n</think>");
                segs.push((out.len(), Role::Assistant));
                out.push_str("\n\n");
                out.push_str(content);
                for (j, tc) in m.tool_calls.as_deref().unwrap_or(&[]).iter().enumerate() {
                    let f = tc.get("function").unwrap_or(tc);
                    let name = f.get("name").and_then(|n| n.as_str()).ok_or("Tool call is missing a function name.")?;
                    if j == 0 {
                        if !content.is_empty() {
                            out.push_str("\n\n");
                        }
                    } else {
                        out.push('\n');
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(name);
                    out.push_str(">\n");
                    // OpenAI clients send `arguments` as a JSON string; the template wants a mapping
                    let args = match f.get("arguments") {
                        None | Some(Value::Null) => Value::Object(Default::default()),
                        Some(Value::String(s)) if s.trim().is_empty() => Value::Object(Default::default()),
                        Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| format!("tool call arguments for {name:?} are not valid JSON: {e}"))?,
                        Some(v) => v.clone(),
                    };
                    let Value::Object(args) = args else { return Err(format!("tool call arguments for {name:?} must be an object")) };
                    for (k, v) in &args {
                        out.push_str("<parameter=");
                        out.push_str(k);
                        out.push_str(">\n");
                        match v {
                            Value::String(s) => out.push_str(s),
                            v => py_json(v, &mut out),
                        }
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                if i == 0 || msgs[i - 1].role != "tool" {
                    segs.push((out.len(), Role::Tool));
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if i + 1 == msgs.len() || msgs[i + 1].role != "tool" {
                    out.push_str("<|im_end|>\n");
                }
            }
            other => return Err(format!("Unexpected message role {other:?}.")),
        }
    }
    segs.push((out.len(), if mode == Thinking::Off { Role::Assistant } else { Role::Reasoning }));
    out.push_str("<|im_start|>assistant\n");
    if mode == Thinking::Off {
        out.push_str("<think>\n\n</think>\n\n");
        Ok(Rendered { text: out, thinking_open: false, images, segs })
    } else {
        out.push_str("<think>\n");
        Ok(Rendered { text: out, thinking_open: true, images, segs })
    }
}

/// Token spans of a rendered prompt: token `i` (byte range `offsets[i]`) takes the role of the
/// segment its first byte falls in; runs of one role become one span. Every token gets a role.
pub fn token_spans(segs: &[(usize, Role)], offsets: &[(usize, usize)]) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::new();
    let mut si = 0usize;
    for (i, &(start, _)) in offsets.iter().enumerate() {
        while si + 1 < segs.len() && segs[si + 1].0 <= start {
            si += 1;
        }
        let role = segs.get(si).map(|s| s.1).unwrap_or(Role::User);
        match spans.last_mut() {
            Some(last) if last.role == role && last.end == i => last.end = i + 1,
            _ => spans.push(Span::new(i, i + 1, role)),
        }
    }
    spans
}

/// Spans after the image pads were expanded: a pad token at row `r` of the unexpanded ids
/// became `n` rows, so every span boundary after it moves by `n - 1`.
pub fn expand_spans(spans: &[Span], ids: &[u32], pad: u32, sizes: &[usize]) -> Vec<Span> {
    // cumulative shift at each unexpanded position
    let mut shift = Vec::with_capacity(ids.len() + 1);
    let mut acc = 0usize;
    let mut k = 0usize;
    for &t in ids {
        shift.push(acc);
        if t == pad {
            acc += sizes.get(k).copied().unwrap_or(1) - 1;
            k += 1;
        }
    }
    shift.push(acc);
    spans.iter().map(|s| Span::new(s.start + shift[s.start], s.end + shift[s.end], s.role)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Message {
        Message { role: role.into(), content: Value::String(content.into()), reasoning_content: None, tool_calls: None }
    }
    #[test]
    fn image_parts() {
        let c = serde_json::json!([{"type": "text", "text": "What is this? "}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": " and this "}, {"type": "image", "image": "http://x/y.jpg"}]);
        let m = Message { role: "user".into(), content: c, reasoning_content: None, tool_calls: None };
        let (p, _, imgs) = render_prompt_images(&[m], Thinking::Off, &[]).unwrap();
        assert!(p.contains(&format!("What is this? {IMAGE_PLACEHOLDER} and this {IMAGE_PLACEHOLDER}<|im_end|>")));
        assert_eq!(imgs, vec!["data:image/png;base64,AAAA".to_string(), "http://x/y.jpg".to_string()]);
        let sys = Message { role: "system".into(), content: serde_json::json!([{"type": "image_url", "image_url": {"url": "data:,x"}}]), reasoning_content: None, tool_calls: None };
        assert!(render_prompt_images(&[sys, msg("user", "hi")], Thinking::Off, &[]).is_err());
    }

    #[test]
    fn template_matches_jinja() {
        // Reference strings rendered with the GGUF template (transformers apply_chat_template).
        let (p, open) = render_prompt(&[msg("user", "Hi ")], Thinking::Off, &[]).unwrap();
        assert_eq!(p, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert!(!open);
        let (p, open) = render_prompt(&[msg("system", "Be brief."), msg("user", "Hi")], Thinking::Medium, &[]).unwrap();
        assert_eq!(p, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert!(open);
        let (p, _) = render_prompt(&[msg("user", "Hi")], Thinking::Low, &[]).unwrap();
        assert!(p.starts_with("<|im_start|>system\nReasoning effort is set to low."));
        let mut a = msg("assistant", "Hello!\n");
        a.reasoning_content = Some("thinking\n".into());
        let (p, _) = render_prompt(&[msg("developer", "A"), msg("system", "B"), msg("user", "Hi"), a, msg("user", "Again")], Thinking::XHigh, &[]).unwrap();
        assert_eq!(
            p,
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nA\nB<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nthinking\n</think>\n\nHello!<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert!(render_prompt(&[msg("user", "x"), msg("system", "late")], Thinking::Off, &[]).is_err());
    }

    #[test]
    fn template_tools() {
        // Reference rendered with jinja2 from the GGUF template (see docs/NOTEBOOK.md).
        let tool = serde_json::json!({"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}});
        let mut a = msg("assistant", "");
        a.tool_calls = Some(vec![serde_json::json!({"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}})]);
        let mut t = msg("tool", "sunny");
        t.role = "tool".into();
        let mut t2 = msg("tool", "windy");
        t2.role = "tool".into();
        let (p, _) = render_prompt(&[msg("system", "Sys."), msg("user", "Weather in Paris?"), a, t, t2, msg("user", "thanks")], Thinking::Off, &[tool]).unwrap();
        let want = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tools_prompt.txt")).unwrap();
        assert_eq!(p, want);
    }

    #[test]
    fn spans_follow_the_turns() {
        // one segment per turn, in order, covering the whole text
        let mut a = msg("assistant", "Hello!");
        a.reasoning_content = Some("thinking".into());
        let mut t = msg("tool", "sunny");
        t.role = "tool".into();
        let mut t2 = msg("tool", "windy");
        t2.role = "tool".into();
        let r = render_prompt_spans(&[msg("system", "Sys."), msg("user", "Hi"), a, t, t2, msg("user", "Again")], Thinking::XHigh, &[]).unwrap();
        let roles: Vec<Role> = r.segs.iter().map(|s| s.1).collect();
        assert_eq!(roles, vec![Role::System, Role::User, Role::Reasoning, Role::Assistant, Role::Tool, Role::User, Role::Reasoning]);
        assert_eq!(r.segs[0].0, 0);
        let text_at = |i: usize| &r.text[r.segs[i].0..r.segs.get(i + 1).map(|s| s.0).unwrap_or(r.text.len())];
        assert!(text_at(1).starts_with("<|im_start|>user\nHi<|im_end|>\n"));
        assert_eq!(text_at(2), "<|im_start|>assistant\n<think>\nthinking\n</think>");
        assert_eq!(text_at(3), "\n\nHello!<|im_end|>\n");
        assert_eq!(text_at(4), "<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n<tool_response>\nwindy\n</tool_response><|im_end|>\n");
        assert_eq!(text_at(6), "<|im_start|>assistant\n<think>\n");
        // thinking off: the tail is the answer's; no system message and no instructions: no System
        let r = render_prompt_spans(&[msg("user", "Hi")], Thinking::Off, &[]).unwrap();
        assert_eq!(r.segs.iter().map(|s| s.1).collect::<Vec<_>>(), vec![Role::User, Role::Assistant]);
        assert_eq!(r.segs[0].0, 0);
        // token spans from byte offsets: tokens take the segment of their first byte
        let segs = [(0, Role::System), (10, Role::User), (20, Role::Reasoning)];
        let offsets = [(0, 3), (3, 10), (10, 12), (12, 20), (20, 21)];
        let sp = token_spans(&segs, &offsets);
        assert_eq!(sp, vec![Span::new(0, 2, Role::System), Span::new(2, 4, Role::User), Span::new(4, 5, Role::Reasoning)]);
        // a token straddling a segment start belongs to the segment it starts in
        let sp = token_spans(&[(0, Role::System), (5, Role::User)], &[(0, 4), (4, 8), (8, 9)]);
        assert_eq!(sp, vec![Span::new(0, 2, Role::System), Span::new(2, 3, Role::User)]);
        // image expansion shifts the boundaries after each pad
        let spans = [Span::new(0, 2, Role::System), Span::new(2, 5, Role::User), Span::new(5, 6, Role::Assistant)];
        let ids = [1, 2, 7, 3, 7, 4];
        let e = expand_spans(&spans, &ids, 7, &[4, 3]);
        assert_eq!(e, vec![Span::new(0, 2, Role::System), Span::new(2, 10, Role::User), Span::new(10, 11, Role::Assistant)]);
    }

    #[test]
    fn thinking_resolution() {
        let base = || serde_json::from_str::<ChatRequest>(r#"{"messages":[]}"#).unwrap();
        assert_eq!(resolve_thinking(&base(), Thinking::XHigh).unwrap(), Thinking::XHigh);
        let r: ChatRequest = serde_json::from_str(r#"{"messages":[],"reasoning_effort":"none"}"#).unwrap();
        assert_eq!(resolve_thinking(&r, Thinking::XHigh).unwrap(), Thinking::Off);
        let r: ChatRequest = serde_json::from_str(r#"{"messages":[],"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"low"}"#).unwrap();
        assert_eq!(resolve_thinking(&r, Thinking::XHigh).unwrap(), Thinking::Off);
        let r: ChatRequest = serde_json::from_str(r#"{"messages":[],"enable_thinking":true}"#).unwrap();
        assert_eq!(resolve_thinking(&r, Thinking::Off).unwrap(), Thinking::XHigh);
        let r: ChatRequest = serde_json::from_str(r#"{"messages":[],"reasoning":{"effort":"medium"}}"#).unwrap();
        assert_eq!(resolve_thinking(&r, Thinking::Off).unwrap(), Thinking::Medium);
        let r: ChatRequest = serde_json::from_str(r#"{"messages":[],"reasoning_effort":"ultra"}"#).unwrap();
        assert!(resolve_thinking(&r, Thinking::Off).is_err());
    }
}
