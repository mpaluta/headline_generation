# from tensor2tensor.data_generators import text_problems
#
# prob = text_problems.Text2TextProblem()
# prob2 = text_problems.Text2textTmpdir(prob)

import os
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems #, token_generator, EOS
from tensor2tensor.data_generators import problem
# from tensor2tensor.data_generators import text_encoder

@registry.register_problem
class Gavrilov(text_problems.Text2TextProblem):
  """Headline generation following along Gavrilov et all."""

  @property
  def targeted_vocab_size(self):
    return 40000

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    # might need a little fudging after pipeline set up but should be close
    if dataset_split == problem.DatasetSplit.TRAIN:
      paths = open(os.path.join(tpm_dir,"meta_train.log"), "r")
    if dataset_split == problem.DatasetSplit.EVAL:
      paths = open(os.path.join(tpm_dir,"meta_dev.log"), "r")
    # if dataset_split == problem.DatasetSplit.TEST:
    #   train_paths = open(os.path.join(tpm_dir,"meta_test.log"), "r")
    next_line = paths.readlines()
    next_line_as_list = next_line.split(",")
    path = next_line_as_list[0] # assuming first element of log file will be path
    article = NYTArticle.from_file(os.path.join('data', 'nyt', path) # could replace with tmp_dir or data_dir
    lede = " ".join(article.lede)
    # body = " ".join(article.paragraphs) # not using for now
    headline = article.print_hede[0]

    yield {"inputs": lede, "targets": headline}


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
  # @property
  # def use_subword_tokenizer(self):
  #   return False
