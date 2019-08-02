import os
import pandas as pd
from NYT_parser import NYTArticle
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import problem
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.models import transformer
# from tensor2tensor.models import UniversalTransformers
# from tensor2tensor.utils import hparam

# Define filepaths
nyt_path = './data/nyt/' # points to folder containing the years folders of the NYT Annotated corpus
log_path = './logs/' # points to folder containing all the logs

GRAF_LIMIT = 10 # this limits the body text to 10 paragraphs

@registry.register_problem
class Gavrilov(text_problems.Text2TextProblem):
  """Headline generation following along Gavrilov et all."""

  @property
  def targeted_vocab_size(self):
    return 40000 # 40000 per Gavrilov but powers of 2 are efficient per
                 # documentation so could also try 32768

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      paths = open(os.path.join(log_path,"meta_train.log"), "r")
    if dataset_split == problem.DatasetSplit.EVAL:
      paths = open(os.path.join(log_path,"meta_dev.log"), "r")

    data_df = pd.read_csv(paths, sep=",", header=0,
                dtype={'filepath': str,'hede_size': int,'wordcount': int,'section': str, 'sent_hede': float, 'sent_lede': float, 'sent_body': float})
    for index, row in data_df.iterrows():
      filepath = row['filepath']
      article = NYTArticle.from_file(os.path.join(nyt_path, filepath))
      lede = " ".join(article.lede)
      body = " ".join(article.paragraphs[:GRAF_LIMIT])
      headline = article.print_hede[0]
      yield {"inputs": body, "targets": headline}

  @property
  def use_subword_tokenizer(self):
    return True

@registry.register_hparams
def universal_transformer_gavrilov():
  """Base parameters for Universal Transformer + Gavrilov hyperparameters."""
  hparams = transformer.transformer_base()
  # To have a similar capacity to the transformer_base with 6 layers,
  # we need to increase the size of the UT's layer
  # since, in fact, UT has a single layer repeating multiple times.

  hparams.hidden_size = 1024 # from t2t universal_tranformer_base
  hparams.filter_size = 4096 # from t2t universal_tranformer_base
  hparams.num_heads = 8
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan" # dropout, add, normalize
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.optimizer="adam_w"
  hparams.optimizer_adam_beta1=0.9
  hparams.optimizer_adam_beta2=0.98
  hparams.num_encoder_layers=4
  hparams.num_decoder_layers=4
  hparams.learning_rate_warmup_steps = 4000
  hparams = universal_transformer.update_hparams_for_universal_transformer(hparams)

  return hparams
