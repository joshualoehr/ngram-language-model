# N-Gram Language Model
Python implementation of an N-gram language model with Laplace smoothing and sentence generation. 

Some NLTK functions are used (`nltk.ngrams`, `nltk.FreqDist`), but most everything is implemented by hand.

Note: the `LanguageModel` class expects to be given data which is already tokenized by sentences. If using the included `load_data` function, the `train.txt` and `test.txt` files should already be processed such that: 
1. punctuation is removed
2. each sentence is on its own line  

See the `data/` directory for examples.

---

Example output for a trigram model trained on `data/train.txt` and tested against `data/test.txt`:
```
Loading 3-gram model...
Vocabulary size: 23505
Generating sentences...
...
<s> <s> the company said it has agreed to sell its shares in a statement </s> (0.03163)
<s> <s> he said the company also announced measures to boost its domestic economy and could be a long term debt </s> (0.01418)
<s> <s> this is a major trade bill that would be the first quarter of 1987 </s> (0.02182)
...
Model perplexity: 51.555
```
The numbers in parentheses beside the generated sentences are the cumulative probabilities of those sentences occurring.

---

Usage info:
```
usage: N-gram Language Model [-h] --data DATA --n N [--laplace LAPLACE] [--num NUM]

optional arguments:
  -h, --help         show this help message and exit
  --data DATA        Location of the data directory containing train.txt and test.txt
  --n N              Order of N-gram model to create (i.e. 1 for unigram, 2 for bigram, etc.)
  --laplace LAPLACE  Lambda parameter for Laplace smoothing (default is 0.01 -- use 1 for add-1 smoothing)
  --num NUM          Number of sentences to generate (default 10)
```

---

Originally authored by Josh Loehr and Robin Cosbey, with slight modifications. Last edited Feb. 8, 2018. 
