import argparse
import cgi
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

# factory method
from FineTuneBERT import BERTTuner, Mode


def make_classify_server(base_model: str, mode: Mode, input_model_path: str):
    bert = BERTTuner(base_model=base_model, mode=mode, checkpoint_path=input_model_path)
    server_with_model = partial(ClassifyServer, bert)
    return server_with_model


class ClassifyServer(BaseHTTPRequestHandler):
    def __init__(self, bert: BERTTuner, *args, **kwargs):
        self.bert = bert
        super().__init__(*args, **kwargs)

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
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return

        if self.path != '/classify':
            self.send_response(400)
            self.end_headers()
            return

        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        request = json.loads(self.rfile.read(length))
        texts = request['texts']

        if not texts or len(texts) % 2 != 0:
            self.send_response(400)
            self.wfile.write(b"<html><head><title>Error: expected an even number of texts</title></head></html>")
            self.end_headers()
            return

        texts1 = texts[:len(texts) // 2]
        texts2 = texts[len(texts) // 2:]
        predictions = self.bert.infer(sentences=texts1, glosses=texts2)

        response = {'id': request['id'], 'result': [predictions], 'status': 200}
        self._set_headers('content-type')
        self.wfile.write(json.dumps(response).encode('utf8'))


def run(*, addr="localhost", port=8008, base_model: str = 'google/bert_uncased_L-12_H-768_A-12', mode: Mode, input_model_path: str):
    server_address = (addr, port)
    httpd = HTTPServer(server_address, make_classify_server(base_model, mode, input_model_path))
    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sentence encoding and scoring HTTP server")
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
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='/home/gerard/data/GlossBERT_datasets/semCor_cased_Mode.Classification-100-6e.pt',
        help="Specify path to model file",
    )
    cml_args = parser.parse_args()
    run(addr=cml_args.listen, port=cml_args.port, base_model='bert-base-cased', mode=Mode.Classification, input_model_path=cml_args.model)
