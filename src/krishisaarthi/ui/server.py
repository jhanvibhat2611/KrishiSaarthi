"""Loopback-only evidence server with explicit synthetic fallback."""
import argparse
import json
import regex
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .qa import answer_question

STATIC = Path(__file__).with_name('static')
ASSETS = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
DOCUMENTS = [
    {'passage_id': 'synthetic-water-01', 'document_id': 'demo-notebook-water', 'title': 'Water observations', 'section': 'Notebook A · entry 1', 'text': 'SYNTHETIC: An invented wheat research notebook compares two fictional irrigation observation plots. This example contains no irrigation schedule or watering recommendation.', 'terms': 'water irrigation wheat pani paani sinchai gehu gehun पानी सिंचाई गेहूँ गेहूं'},
    {'passage_id': 'synthetic-soil-02', 'document_id': 'demo-notebook-soil', 'title': 'Soil sample records', 'section': 'Notebook B · entry 2', 'text': 'SYNTHETIC: A fictional researcher labels wheat soil samples Cedar and Willow before recording observations. These invented labels are not soil classes or fertilizer advice.', 'terms': 'soil sample fertilizer wheat mitti khaad gehu gehun मिट्टी खाद उर्वरक गेहूँ गेहूं'},
    {'passage_id': 'synthetic-seed-03', 'document_id': 'demo-notebook-seed', 'title': 'Seed observation notes', 'section': 'Notebook C · entry 3', 'text': 'SYNTHETIC: An invented seed notebook records the appearance of two fictional wheat seed batches, Aster and Fern. It specifies no variety, seed rate or sowing recommendation.', 'terms': 'seed sowing wheat beej buwai gehu gehun बीज बुवाई गेहूँ गेहूं'},
]

def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError('Send a question and language.')
    q, language = payload.get('query'), payload.get('language')
    if not isinstance(q, str) or not q.strip() or len(q) > 1000:
        raise ValueError('Enter a question between 1 and 1,000 characters.')
    if language not in ('en', 'hi', 'hinglish'):
        raise ValueError('Choose English, Hindi or Hinglish.')
    return q, language

def search(payload):
    q, language = validate_payload(payload)
    # UI fixture matching only: not BM25, dense retrieval, or a research experiment.
    tokens = set(regex.findall(r'[\p{L}\p{M}\p{N}]+', q.casefold()))
    matches = [(len(tokens & set(regex.findall(r'[\p{L}\p{M}\p{N}]+', d['terms']))), d) for d in DOCUMENTS]
    ranked = sorted((x for x in matches if x[0]), key=lambda x: (-x[0], x[1]['passage_id']))
    return {'mode': 'synthetic', 'live_available': False, 'results': [dict({k: v for k, v in d.items() if k != 'terms'}, rank=i, citation=f"DEMO · {d['document_id']} · {d['section']}") for i, (_, d) in enumerate(ranked, 1)]}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Never log questions, responses or research text.

    def reply(self, status, body, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def allowed_host(self):
        return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

    def do_GET(self):
        if not self.allowed_host():
            self.reply(403, b'{"error":"Host not allowed"}'); return
        if self.path == '/api/status':
            self.reply(200, json.dumps({'live_available':getattr(self.server, 'retrieval', None) is not None}).encode()); return
        asset = ASSETS.get(self.path)
        if not asset:
            self.reply(404, b'{"error":"Not found"}')
            return
        name, kind = asset
        self.reply(200, (STATIC / name).read_bytes(), kind)

    def do_POST(self):
        if self.path not in ('/api/search', '/api/answer', '/api/retrieve'):
            self.reply(404, b'{"error":"Not found"}')
            return
        if not self.allowed_host():
            self.reply(403, b'{"error":"Host not allowed"}'); return
        # Prevent cross-origin browser requests; no CORS is enabled.
        if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}'):
            self.reply(403, b'{"error":"Origin not allowed"}')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192:
                raise ValueError('Request is too large or empty.')
            payload = json.loads(self.rfile.read(length))
            if self.path == '/api/retrieve':
                validate_payload(payload)
                service = getattr(self.server, 'retrieval', None)
                if service is None:
                    self.reply(503, b'{"error":"Local E5 unavailable; choose the synthetic fallback explicitly."}'); return
                try:
                    result = service.search(payload)
                except Exception:
                    self.reply(503, b'{"error":"Local retrieval failed. No fallback was substituted."}'); return
            else:
                result = search(payload)
            if self.path == '/api/answer':
                result['qa'] = answer_question(payload['query'], payload['language'], result['results']).model_dump()
            self.reply(200, json.dumps(result, ensure_ascii=False).encode())
        except (ValueError, UnicodeError):
            self.reply(400, b'{"error":"Enter a valid question (up to 1,000 characters) and language."}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--real', action='store_true', help='Require frozen PAU/E5 artifacts; fail rather than rebuild')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    retrieval = None
    if args.real:
        from .local import LocalRetrieval
        retrieval = LocalRetrieval(args.root)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.retrieval = retrieval
    print(f'KrishiSaarthi local evidence: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
