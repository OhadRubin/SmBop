import comet_ml


from overrides.overrides import overrides

# import sklearn
# import torch
from sh import sed
import sh

# import allennlp


from allennlp.commands.train import train_model
from allennlp.common import Params

#Old imports causing app to break
#from dataset_readers.forest_spider import ForestSpiderDatasetReader
#from models.semantic_parsing.smbop import SmbopParser
#from modules.relation_transformer import RelationTransformer
#from modules.lxmert import LxmertCrossAttentionLayer
#from training.callbacks import PredictionLogger, PerformanceLogger

from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
from smbop.models.smbop import SmbopParser
from smbop.modules.relation_transformer import RelationTransformer
from smbop.modules.lxmert import LxmertCrossAttentionLayer



# from allennlp.common.params import with_fallback
import namegenerator
from allennlp.commands.train import train_model_from_file

serialization_dir = "/home/ohadr/experiments/crabby-auburn-catfish_gpu7"
parameter_filename = f"{serialization_dir}/config.json"
# params = Params.from_file(parameter_filename)


# params['trainer']['cuda_device']=4
# print("hi")
train_model_from_file(
    parameter_filename=parameter_filename,
    serialization_dir=serialization_dir,
    recover=True,
)
# train_model(params,)
# train_mo
