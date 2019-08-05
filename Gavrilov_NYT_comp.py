# from tensor2tensor.data_generators import text_problems
#
# prob = text_problems.Text2TextProblem()
# prob2 = text_problems.Text2textTmpdir(prob)

import csv
import datetime
import xml.etree.ElementTree as ET
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from typing import Sequence, Iterable, Optional, Dict, Any, Union, TextIO
from attr import attrs, attrib, asdict
import os
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.data_generators import text_problems #, token_generator, EOS
from tensor2tensor.data_generators import problem
import pandas as pd
# from tensor2tensor.data_generators import text_encoder

# Define filepaths
nyt_path = './data/nyt/' # points to folder containing the years folders of the NYT Annotated corpus
log_path = './logs/' # points to folder containing all the logs

GRAF_LIMIT = 3 # this limits the body text to 3 paragraphs
NO_INDEX_TERMS = 'NO INDEX TERMS FROM NYTIMES'


# NYTArticle class code adapted and expanded from
# https://github.com/ConstantineLignos/nyt-corpus-reader/blob/master/nytcorpusreader/nyt_parser.py

@attrs
class NYTArticle:
    """
    Note from original author:
    Parse and store the fields of an NYT Annotated Corpus article.
    Note that due to issues with the original data, descriptors,
    general descriptors, and types of material are lowercased. As
    some types of material mistakenly contain article text, long
    entries or entries containing tags in that field are removed.
    Additional adds:
    * added variables for other XML fields of potential interest
    * added a pass_filters function that checks to see if article meets various criteria
    * added simple_csv_output function that saves the doc to a simplified csv format
    """

    docid: str = attrib()
    title: Optional[str] = attrib()
    date: datetime.datetime = attrib()
    descriptors: Sequence[str] = attrib()
    general_descriptors: Sequence[str] = attrib()
    types_of_material: Sequence[str] = attrib()
    paragraphs: Sequence[str] = attrib()
    # added variables
    summary: Sequence[str] = attrib()
    dateline: Sequence[str] = attrib()
    lede: Sequence[str] = attrib()
    print_hede: Sequence[str] = attrib()
    online_hede: Sequence[str] = attrib()
    section: Sequence[str] = attrib()
    wordcount: int = attrib()

    @classmethod
    def from_element_tree(cls, root: Union[ET.Element, ET.ElementTree]) -> 'NYTArticle':
        head = root.find("./head")
        title_element = head.find("./title")
        title = title_element.text if title_element is not None else None
        # pubdata appears to always be there, but publication_{year,month,day_of_month} are missing
        # in some articles
        pubdata = head.find("./pubdata")
        date = datetime.datetime.strptime(pubdata.get('date.publication'), '%Y%m%dT%H%M%S')

        docdata = head.find("./docdata")
        docid = docdata.find("./doc-id").get('id-string')
        assert docid is not None
        descriptors = _clean_descriptors(d.text for d in docdata.findall(
            "./identified-content/*[@type='descriptor']"))
        general_descriptors = _clean_descriptors(d.text for d in docdata.findall(
            "./identified-content/*[@type='general_descriptor']"))
        types_of_material = _clean_types_of_material(d.text for d in docdata.findall(
            "./identified-content/classifier[@type='types_of_material']"))
        paragraphs = [p.text for p in root.findall(
            "./body/body.content/*[@class='full_text']/p")]
        if len(paragraphs) > 1 and "LEAD:" in paragraphs[0]: # remove situation where lede is in paragraph 0
            paragraphs = paragraphs[1:]

        # added fields
        summary = [s.text for s in root.findall("./body/body.head/summary")]
        dateline = [d.text for d in root.findall("./body/body.head/dateline")]
        lede = [l.text for l in root.findall("./body/body.content/block[@class='lead_paragraph']/p")]
        print_hede = [ph.text for ph in root.findall("./body[1]/body.head/hedline/hl1")]
        online_hede = [oh.text for oh in root.findall("./body[1]/body.head/hedline/hl2")]
        try:
            section = head.find("./meta[@name='print_section']").get('content')
        except:
            section = ''
        wordcount = pubdata.get('item-length')

        # Mypy and Pycharm don't understand the attrs __init__ arguments
        # noinspection PyArgumentList
        return cls(docid, title, date, descriptors, general_descriptors,  # type: ignore
                   types_of_material, paragraphs, summary, dateline, lede, print_hede, online_hede,
                   section, wordcount)

    @classmethod
    def from_file(cls, input_file: TextIO) -> 'NYTArticle':
        return cls.from_element_tree(ET.parse(input_file))

    @classmethod
    def from_str(cls, contents: str) -> 'NYTArticle':
        return cls.from_element_tree(ET.fromstring(contents))

    def as_dict(self) -> Dict[Any, Any]:
        return asdict(self)

    def get_meta(self):
        # grabs meta data to write to the meta log file
        tokenizer = RegexpTokenizer(r'\w+')
        hede_words = tokenizer.tokenize(self.print_hede[0])
        return(len(hede_words), self.descriptors, self.wordcount)

    def pass_filters(self):
        # checks to see if doc meets word count, headline size, non-Obituary filters
        keep = False
        if self.wordcount and self.print_hede: # fails filter in cases of empty fields
            meets_wc = int(self.wordcount) >= 20 and int(self.wordcount) <= 2000 # from Gavrilov et al
            tokenizer = RegexpTokenizer(r'\w+')
            words = tokenizer.tokenize(self.print_hede[0])
            meets_hede = len(words) >= 3 and len(words) <= 15 # from Gavrilov et al

            # is it an obituary?
            obit = False
            description = " ".join(self.descriptors) # turn into string for easier lookup
            description = description.lower()
            headline = self.print_hede[0].lower()
            if ("obitua" in description) or ("deaths" in description) or ("paid notice" in headline) or ("obitua" in headline):
                obit = True

            keep = meets_wc and meets_hede and not obit
        return (keep)

    def simple_csv_output(self, dest_folder: str):
        # This function writes the doc to a csv file with just the needed data
        # inputs a pathname
        with open(dest_folder+self.docid+".csv",'w') as resultFile:
            wr = csv.writer(resultFile)
            wr.writerow([self.print_hede[0]])
            wr.writerow([self.paragraphs])
        return

def _clean_descriptors(descriptors: Iterable[str]) -> Sequence[str]:
    """Deduplicate and clean descriptors
    The returned sequence is sorted to ensure the ordering is deterministic.
    """
    return sorted(set(descriptor.lower() for descriptor in descriptors
                      if descriptor is not None and descriptor != NO_INDEX_TERMS))


def _clean_types_of_material(types_of_material: Iterable[str]) -> Sequence[str]:
    """Remove any bad data in types_of_material"""
    # We use the presence of newline or <> (for tags) as evidence that something is wrong
    return [item.lower() for item in types_of_material
            if item is not None and len(item) < 50 and '\n' not in item
            and '<' not in item and '>' not in item]


# Code to write our own problem in t2t

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
      paths = open(os.path.join(log_path,"meta_comp.log"), "r")
    if dataset_split == problem.DatasetSplit.EVAL:
      paths = open(os.path.join(log_path,"meta_dev.log"), "r")

    data_df = pd.read_csv(paths, sep=",", header=0,
                dtype={'filepath': str,'hede_size': int,'wordcount': int,'section': str, 'sent_hede': float, 'sent_lede': float, 'sent_body': float})
    for index, row in data_df.iterrows():
      filepath = row['filepath']

      # debug portion
      if dataset_split == problem.DatasetSplit.EVAL:
          print("EVAL!", filepath)
    
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
