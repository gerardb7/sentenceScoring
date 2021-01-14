from http.server import HTTPServer, BaseHTTPRequestHandler

from mlm.loaders import Corpus
from mlm.scorers import MLMScorer
from mlm.models import get_pretrained
import mxnet as mx
import argparse
import json
import cgi


class Server(BaseHTTPRequestHandler):
    ctxs = [mx.gpu()]
    model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
    scorer = MLMScorer(model, vocab, tokenizer, ctxs)

    def _set_headers(self, content_type):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.end_headers()

    @staticmethod
    def _html(message):
        """This just generates an HTML document that includes `message`
        in the body. Override, or re-write this do do more interesting stuff.
        """
        content = f"<html><body><h1>{message}</h1></body></html>"
        return content.encode('utf8')  # NOTE: must return a bytes object!

    def do_GET(self):
        self._set_headers('text/html')
        self.wfile.write(self._html('hi'))

    def do_HEAD(self):
        self._set_headers('text/html')

    def do_POST(self):
        print('received request')
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))

        # refuse to receive non-json content
        if ctype != 'application/json' or self.path != '/score':
            self.send_response(400)
            self.end_headers()
            return

        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        request = json.loads(self.rfile.read(length))
        sentences = request['texts']
        corpus = Corpus.from_text(sentences.toList())

        # Sentences are encoded by calling model.encode()
        print(f'scoring {len(sentences)} sentences')
        scores = self.scorer.score(corpus)
        print(f'done')
        response = {'id': request['id'], 'result': scores, 'status': 200}
        self._set_headers('content-type')
        self.wfile.write(json.dumps(response).encode('utf8'))


def run(server_class=HTTPServer, handler_class=Server, addr="localhost", port=8008):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
