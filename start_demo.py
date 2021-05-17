import pathlib
import gdown
import argparse
import torch
from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.vocabulary import Vocabulary
from smbop.modules.relation_transformer import *
from allennlp.common import Params
from smbop.models.smbop import SmbopParser
from smbop.modules.lxmert import LxmertCrossAttentionLayer
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
import itertools
import smbop.utils.node_util as node_util
import numpy as np
from allennlp.models import Model
from allennlp.common.params import *
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.predictors import Predictor
import json

overrides = {
    "dataset_reader": {
        "tables_file": "dataset/tables.json",
        "dataset_path": "dataset/database",
    },
    "validation_dataset_reader": {
    "tables_file": "dataset/tables.json",
    "dataset_path": "dataset/database",
    }
}
predictor = Predictor.from_path(
    "model.tar.gz", cuda_device=0, overrides=overrides
)
instance_0 = predictor._dataset_reader.text_to_instance(
    utterance="asds", db_id="aircraft"
)
predictor._dataset_reader.apply_token_indexers(instance_0)  
def inference(question,db_id):
  instance = predictor._dataset_reader.text_to_instance(
      utterance=question, db_id=db_id,
  )
  predictor._dataset_reader.apply_token_indexers(instance)  
  with torch.cuda.amp.autocast(enabled=True):
        
      out = predictor._model.forward_on_instances(
          [instance, instance_0]
      )
      return out[0]["sql_list"]
print(inference("How many films cost below 10 dollars?","cinema"))
