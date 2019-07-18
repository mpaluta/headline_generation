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
    input_train = open(os.path.join(tpm_dir,"inputs.train.txt"), "r")
    input_target = open(os.path.join(tmp_dir,"inputs.target.txt"), "r")
    yield {"inputs": input_train, "targets": input_target}


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
