# sentenceScoring

Set of python scripts used in my expriments with WSD, EL and summarization.

- Score server.py: HTTP server exposing two methods:
  - an `/encode` method that given a sentence returns an embeddings from models of the [transformers](https://github.com/huggingface/transformers) library 
  - a `/score` method that given a sentence returns an acceptability score accoring to a language model such as [GPT-2](https://github.com/openai/gpt-2) 
- MLM score server.py: HTTP server exposing a `/score` method that given a sentence scores them using the [MLM-scorers](https://github.com/awslabs/mlm-scoring)   
- FineTuneBERT.py: trains, evaluates or uses for inference a model based on [BERT](https://github.com/google-research/bert) and fine-tuned using pairs of sentences, such as those used for [GlossBERT](https://github.com/HSLCY/GlossBERT)
- Classify server.py: HTTP server exposing a `/classify` method that, given a pair of sentences, returns a classification score from a model fine-tuned with FineTuneBERT.py

  
