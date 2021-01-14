# sentenceScoring

Set of python scripts used in my expriments with WSD, EL and summarization.

- Score server.py: HTTP server exposing two methods:
  - an '/encode'  method that given a sentence returns an embeddings from models of the transformers library 
  - a '/score' method that given a sentence returns an acceptability score accoring to a language model such as GPT2 
- FineTuneBERT.py: trains, evaluates or uses for inference a model based on BERT and fine-tuned using pairs of sentences, such as those used for GlossBERT
- Classify server.py: HTTP server exposing a '/classify' method that, given a pair of sentences, returns a classification score from a model fine-tuned with FineTuneBERT.py

  
