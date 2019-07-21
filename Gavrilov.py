# from tensor2tensor.data_generators import text_problems
#
# prob = text_problems.Text2TextProblem()
# prob2 = text_problems.Text2textTmpdir(prob)

import os
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems #, token_generator, EOS
from tensor2tensor.data_generators import problem
# from tensor2tensor.data_generators import text_encoder

# Define filepaths
nyt_path = './data/nyt/' # points to folder containing the years folders of the NYT Annotated corpus
log_path = './logs/' # points to folder containing all the logs

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
    # might need a little fudging after pipeline set up but should be close
    if dataset_split == problem.DatasetSplit.TRAIN:
      paths = open(os.path.join('.',"meta_train.log"), "r")
    if dataset_split == problem.DatasetSplit.EVAL:
      paths = open(os.path.join('.',"meta_dev.log"), "r")
    # if dataset_split == problem.DatasetSplit.TEST:
    #   train_paths = open(os.path.join(tpm_dir,"meta_test.log"), "r")

  data_df = pd.read_csv(paths, sep=",", header=0,
                dtype={'filepath': str,'hede_size': int,'wordcount': int,'section': str, 'sent_hede': float, 'sent_lede': float, 'sent_body': float})
  for index, row in data_df.iterrows():
     filepath = row['filepath']
     article = NYTArticle.from_file(os.path.join(nyt_path, filepath))
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
  @property
  def use_subword_tokenizer(self):
    return True
