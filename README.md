# headline_generation

This is our final project for W266 in the Berkeley MIDS program (Natural Language Processing with Deep Learning)

## Abstract

In this paper we propose a strategy for sentiment-based preprocessing on the task of headline generation. We seek to show that training on sentiment-matched headlines and bodies would result in similar ROUGE scores while simultaneously improving sentiment metrics. Due to resource constraints, we are unable to train to convergence, but our preliminary results support our hypotheses. Training a Universal Transformer on two equivalent data set sizes, one sentiment matched and one not, we see no noteworthy effect size in ROUGE while seeing improvement of 9% in our sentiment match between the generated headline and the article lede (first paragraph).

## Repo details

We define a new problem using the package `tensor2tensor` and the NYT annotated corpus. We use a series of notebooks with some "problem" files in the background to work through the various steps of this project. The intended folder hierarchy is as follows.

```
# headline_generation (folder)
#      |___ main.ipynb           The main notebook for training and showing test results
#      |___ NYT_parser.py        A class for parsing the raw XML files
#      |___ utilities.py         Some helper functions to keep code from getting cluttered
#      |___ EDA.ipynb            Some initial exploratory data analysis work
#      |___ __init__.py          Required file for t2t
#      |___ Gavrilov.py          Tensor2Tensor subclass that defines our Problem 
#      |___ logs (folder)        Lists of filepaths based on various filters and train/dev/test split
#      |___ data (folder)
#             |___ sentiment (folder)
#                    |___ positive-words.txt (unrar from: http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)
#                    |___ negative-words.txt (unrar from: http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)
#             |___ glove (folder)
#                    |___ glove.42B.300d.txt (unzip from: http://nlp.stanford.edu/data/glove.42B.300d.zip)
#             |___ nyt (folder - unzip/untar from https://catalog.ldc.upenn.edu/download/a22bbeb044db7cb70954c21e130aec48c512cb90a2874a6746e3bc722b3f)
#                    |___ 1987 (folders for all years from 1987 to 2007)
#                           |___ 01 (folders for all months 1 to 12)
#                                 |___ 01 (folders for all days of month)
#                                       |___ 0000000.xml (1.8 million xml files numbered sequentially from 0)
```

### EDA

`EDA.ipynb` - examine various fields of one article and metrics for a sample of articles  
`Sentiment.ipynb` - EDA for sentiment of articles

### Preprocessing

`main.ipynb` - preprocessing notebook that compiles metadata about articles and assembles train/test/dev splits based on filtering options

### Training/Decoding

`Gavrilov.py` - an imitation of Gavrilov et al's parameters. This ultimately proved unsuccessful and we reverted to standard hyperparameters  
`Gavrilov_unfltrd.py` - exactly the same as `Gavrilov.py` except points to `meta_train_unfltrd.log` instead of `meta_train.log`  
`Gavrilov_basic.py` - this is the problem we ended up using for final results.  
`TrainModels.ipynb` - example commands for training and decoding. These can also be run from the command line. For the different filtering methods, these should be customized.

### Scoring

`ROUGE_testing.ipynb` - calculates ROUGE scores  
`Postprocessing_pipeline.ipynb` - calculates sentiment scores  
`Compare_results.ipynb` - print out headlines generated from various methods for a side-by-side comparison
