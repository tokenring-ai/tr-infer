#!/usr/bin/env python3
"""Exercise a running continuous-batching server (stdlib only).

Start a text server with >= 4 sequences and --ctx >= 2048, then run:
  python3 tools/continuous_smoke.py --url http://127.0.0.1:18089
Use --cache to also require a persistent exact-prompt cache hit.
"""
import argparse
import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18089')
    parser.add_argument('--api-key')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--queue-capacity', type=int, help='test saturation against a server with this max-queue (use 2)')
    args = parser.parse_args()
    headers = {'Content-Type': 'application/json'}
    if args.api_key:
        headers['Authorization'] = 'Bearer ' + args.api_key

    def health():
        with urllib.request.urlopen(args.url + '/health', timeout=30) as r:
            return json.load(r)

    def request(n=1, limit=32, stream=False, prompt='List the integers from one to thirty.', seed=42):
        payload = {'messages': [{'role': 'user', 'content': prompt}], 'n': n,
                   'max_tokens': limit, 'seed': seed, 'temperature': 0, 'stream': stream,
                   'reasoning_effort': 'none', 'stream_options': {'include_usage': True}}
        req = urllib.request.Request(args.url + '/v1/chat/completions', json.dumps(payload).encode(), headers)
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as response:
            if not stream:
                body = json.load(response)
                assert [c['index'] for c in body['choices']] == list(range(n)), body
                assert all(c['finish_reason'] in ('length', 'stop', 'tool_calls') for c in body['choices']), body
                assert body['usage']['completion_tokens'] <= n * limit, body
                return body
            first = None
            end = set()
            usage = None
            done = False
            text = [''] * n
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                assert 'error' not in event, event
                if 'usage' in event:
                    usage = event['usage']
                for c in event['choices']:
                    i = c['index']
                    assert 0 <= i < n, event
                    if c['delta'].get('content') or c['delta'].get('reasoning_content'):
                        first = first or time.monotonic()
                    text[i] += c['delta'].get('content', '')
                    if c['finish_reason']:
                        assert i not in end, 'duplicate finish'
                        end.add(i)
            assert done and end == set(range(n)) and usage, (done, end, usage)
            return {'first': first, 'ttft_ms': (first - started) * 1000 if first else None, 'end': time.monotonic(), 'text': text, 'usage': usage}

    initial = health()
    assert initial['scheduler']['kv_total_pages'] > 0, 'server is not in continuous mode'
    for n in (0, 100000):
        try:
            request(n=n)
        except urllib.error.HTTPError as e:
            assert e.code == 400, e
        else:
            raise AssertionError('invalid n accepted')
    body = request(n=3, limit=4)
    assert len({c['message'].get('content') for c in body['choices']}) == 1, 'greedy forks differ'
    # Start a long generation, then admit a second request while it is running.
    barrier = threading.Barrier(2)
    def concurrent_request(n, limit, prompt):
        barrier.wait()
        return request(n=n, limit=limit, stream=True, prompt=prompt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(concurrent_request, 2, 96, 'Write a long numbered list of words.')
        b = pool.submit(concurrent_request, 1, 24, 'Write another numbered list of words.')
        a, b = a.result(), b.result()
    assert a['first'] and b['first'], 'test prompts produced no content'
    assert max(a['first'], b['first']) < min(a['end'], b['end']), 'requests did not overlap'
    assert a['text'][0] == a['text'][1], 'identical greedy forks differ within the same batch'
    # Disconnect after response headers, then confirm all slots/pages become reusable.
    payload = {'messages': [{'role': 'user', 'content': 'Keep counting.'}], 'n': 4, 'max_tokens': 1000, 'stream': True}
    req = urllib.request.Request(args.url + '/v1/chat/completions', json.dumps(payload).encode(), headers)
    response = urllib.request.urlopen(req, timeout=120)
    response.close()
    deadline = time.monotonic() + 30
    while True:
        status = health()['scheduler']
        if status['active_sequences'] == 0 and status['queued_requests'] == 0:
            assert status['kv_free_pages'] + status.get('idle_pages', 0) == status['kv_total_pages'], status
            break
        assert time.monotonic() < deadline, status
        time.sleep(.05)
    request(limit=2)
    if args.queue_capacity:
        payload = {'messages': [{'role': 'user', 'content': 'Keep generating a long list.'}], 'n': 4,
                   'max_tokens': 2000, 'temperature': 0, 'reasoning_effort': 'none', 'stream': True}
        req = urllib.request.Request(args.url + '/v1/chat/completions', json.dumps(payload).encode(), headers)
        held = urllib.request.urlopen(req, timeout=120)
        saturated = threading.Event()
        def waiting_request():
            try:
                request(limit=1, prompt='Queued probe')
                return 200
            except urllib.error.HTTPError as e:
                if e.code == 503:
                    saturated.set()
                return e.code
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.queue_capacity + 1) as pool:
            pending = [pool.submit(waiting_request) for _ in range(args.queue_capacity + 1)]
            assert saturated.wait(10), 'queue did not reject an excess request'
            held.close()
            statuses = [future.result() for future in pending]
            assert 503 in statuses and all(s in (200, 503) for s in statuses), statuses
    if args.cache:
        prompt = 'Cache exact prompt probe ' + str(time.time_ns())
        request(n=1, limit=1, prompt=prompt)
        request(limit=1, prompt='Evict the resident prefix ' + str(time.time_ns()))
        time.sleep(1)
        before = health()['scheduler']['prefill_rows']
        request(n=1, limit=1, prompt=prompt)
        after = health()['scheduler']['prefill_rows']
        assert before == after, f'exact prompt was not restored: prefill rows {before} -> {after}'
    deadline = time.monotonic() + 30
    while True:
        final = health()['scheduler']
        if final['active_sequences'] == 0 and final['queued_requests'] == 0:
            break
        assert time.monotonic() < deadline, final
        time.sleep(.05)
    assert final['kv_free_pages'] + final.get('idle_pages', 0) == final['kv_total_pages'], final
    assert final['max_batch_sequences'] >= 3, final
    print(json.dumps({'result': 'passed', 'scheduler': final, 'ttft_ms': [a['ttft_ms'], b['ttft_ms']], 'overlap_seconds': min(a['end'], b['end']) - max(a['first'], b['first'])}, indent=2))


if __name__ == '__main__':
    main()
