import os
import pandas as pd
from .NYT_parser import NYTArticle
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

GRAF_LIMIT = 3 # this limits the body text to 10 paragraphs

@registry.register_problem
class Gavrilov(text_problems.Text2TextProblem):
  """Headline generation following along Gavrilov et all."""

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

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
