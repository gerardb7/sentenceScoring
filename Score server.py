from http.server import HTTPServer, BaseHTTPRequestHandler

from sentence_transformers import SentenceTransformer

from mlm.loaders import Corpus
from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx

import math
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

import argparse
import json
import cgi


class Server(BaseHTTPRequestHandler):
    # sentence_transformers
    sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    # gpt with pytorch_pretrained_bert
    # torch.cuda.set_device(0)
    # model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    # model.eval()
    # tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    # mlms scorers
    ctxs = [mx.gpu()]
    # mlms_model, vocab, tokenizer = get_pretrained(ctxs, 'roberta-base-en-cased')
    # scorer = MLMScorer(mlms_model, vocab, tokenizer, ctxs)
    # mlms_model, vocab, tokenizer = get_pretrained(ctxs, 'distilbert-base-cased')
    # scorer = MLMScorerPT(mlms_model, vocab, tokenizer, ctxs)
    mlms_model, vocab, tokenizer = get_pretrained(ctxs, 'gpt2-117m-en-cased')
    scorer = LMScorer(mlms_model, vocab, tokenizer, ctxs)

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

        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        request = json.loads(self.rfile.read(length))
        sentences = request['texts']

        if self.path == '/encode':
            # Encode sentences using Sentence scoring model
            print(f'encoding {len(sentences)} sentences')
            embeddings = self.sbert_model.encode(sentences)
            print(f'done')
            response = {'id': request['id'], 'result': embeddings.tolist(), 'status': 200}
            self._set_headers('content-type')
            self.wfile.write(json.dumps(response).encode('utf8'))
        elif self.path == '/score':
            print(f'scoring {len(sentences)} sentences')
            scores = self.model_score(sentences)
            print(f'done')
            response = {'id': request['id'], 'result': [scores[0]], 'status': 200}
            self._set_headers('content-type')
            self.wfile.write(json.dumps(response).encode('utf8'))

    # uses mlms scorer
    def model_score(self, sentences):
        corpus = Corpus.from_text(sentences)
        return self.scorer.score(corpus, 1.0, 50)

    # uses model directly with pytorch_pretrained_bert
    # def model_score(self, sentences):
    #     scores = []
    #     for sentence in sentences:
    #         tokenize_input = self.tokenizer.tokenize(sentence)
    #         tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
    #         loss = self.model(tensor_input, lm_labels=tensor_input)
    #         scores.append(math.exp(loss))
    #
    #     return scores


def run(server_class=HTTPServer, handler_class=Server, addr="localhost", port=8008):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

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
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
