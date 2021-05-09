from overrides.overrides import overrides
import argparse
from allennlp.commands.train import train_model
from allennlp.common import Params
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
from smbop.models.smbop import SmbopParser
from smbop.modules.relation_transformer import RelationTransformer
from smbop.modules.lxmert import LxmertCrossAttentionLayer
import namegenerator
from allennlp.commands.train import train_model_from_file,train_model
from allennlp.common import Params
import json
from allennlp.common.params import with_fallback

def modify_config(config_dir, override):
    with open(f"{config_dir}/config.json","r") as f:
        new_config  = with_fallback(override,json.load(f))
    with open(f"{config_dir}/config.json","w") as g:
        json.dump(new_config,g)


#usage
#python resume.py --dir /home/ohadr/experiments/wimpy-cobalt-ragdoll --gpu 1
def run():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--dir", type=str)
    parser.add_argument("--gpu", type=int,default=0)
    
    args = parser.parse_args()
    overrides_dict = {"trainer":{
                            "cuda_device": args.gpu,
                            }
    }
    modify_config(args.dir, overrides_dict)
    train_model_from_file(
        serialization_dir=args.dir,
        parameter_filename=f"{args.dir}/config.json",
        recover=True,
        )
    
if __name__ == "__main__":
    run()



