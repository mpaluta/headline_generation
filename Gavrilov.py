# from tensor2tensor.data_generators import text_problems
#
# prob = text_problems.Text2TextProblem()
# prob2 = text_problems.Text2textTmpdir(prob)

import os
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems #, token_generator, EOS
from tensor2tensor.data_generators import problem
import pandas as pd
from .NYT_parser import NYTArticle
# from tensor2tensor.data_generators import text_encoder

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
    return False

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      paths = open(os.path.join(log_path,"meta_train_unfltrd.log"), "r")
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

  # @property
  # def vocab_name(self):
  #   return "vocab.encs.bpe"
  #
  # @property
  # def input_space_id(self):
  #   return problem.SpaceID.EN_BPE_TOK

  #
  # @property
  # def target_space_id(self):
  #   return problem.SpaceID.CS_TOK # TODO CS_BPE_TOK
  #
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
  hparams.layer_postprocess_sequence = "adn" # add, dropout, normalize
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.optimizer="adam_w"
  hparams.optimizer_adam_beta1=0.9
  hparams.optimizer_adam_beta2=0.98
  hparams.num_encoder_layers=4
  hparams.num_decoder_layers=4
  hparams.learning_rate_warmup_steps = 4000
  hparams = update_hparams_for_universal_transformer(hparams)

  return hparams
