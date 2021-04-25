local stringToBool(s) =
  if std.extVar(s) == "true" then true
  else if std.extVar(s) == "false" then false
  else error "invalid boolean: " + std.manifestJson(std.extVar(s));
local stringToFloat(s) = std.parseJson(std.extVar(s));
local stringToInt(s) = std.parseInt(std.extVar(s));

local misc_params = {
          "is_oracle": stringToBool("is_oracle"),
          "debug": stringToBool("debug"),
          "cntx_rep": stringToBool("cntx_rep"), 
          "cntx_reranker": stringToBool("cntx_reranker"),
          "utt_aug": stringToBool("utt_aug"),
          "should_rerank": stringToBool("should_rerank"),
          "lin_after_cntx": stringToBool("lin_after_cntx"),
          "cntx_beam":stringToBool("cntx_beam"),
          "disentangle_cntx":stringToBool("disentangle_cntx"),
          "value_pred":stringToBool("value_pred"),
          "uniquify":stringToBool("uniquify"),
          "use_bce": stringToBool("use_bce"),
          "tiny_dataset":stringToBool("tiny_dataset"),
          "load_less":stringToBool("load_less"),
          "rat_layers": stringToInt("rat_layers"),
          "base_dim":stringToInt("base_dim"),
          "num_heads":stringToInt("num_heads"),
          "embedding_dim":stringToInt("num_heads")*stringToInt("base_dim"),
          "lr": stringToFloat("lr"),
          "lm_lr": stringToFloat("lm_lr"),
          "rat_dropout":stringToFloat("rat_dropout"),
          "tree_rep_transformer_num_layers":stringToInt("tree_rep_transformer_num_layers"),
          "beam_encoder_num_layers":stringToInt("beam_encoder_num_layers"),
          "use_treelstm":stringToBool("use_treelstm"),
          "batch_size":stringToInt("batch_size"),
          "grad_acum":stringToInt("grad_acum"),
          "power":stringToFloat("power"),
          "max_steps":stringToInt("max_steps"),
          "tfixup": stringToBool("tfixup"),
          "amp": stringToBool("amp"),
          "train_as_dev": stringToBool("train_as_dev"),
          "temperature": stringToFloat("temperature"),
          "grad_clip" : if stringToFloat("grad_clip")>0 then stringToFloat("grad_clip")  else null,
          "grad_norm" : if stringToFloat("grad_norm")>0 then stringToFloat("grad_norm")  else null,
        };

local scheduler = {
                      "type": "polynomial_decay",
                      "warmup_steps": 1,
                      "power": misc_params.power,
};

local large_setting = {
  batch_size :: misc_params.batch_size,
  rat_layers :: misc_params.rat_layers,
  grad_acum ::  misc_params.grad_acum,
  model_name :: "Salesforce/grappa_large_jnt",
  // model_name :: "google/bigbird-roberta-large",
  
  pretrained_embedding_dim :: 1024,
  // cache_path ::  if misc_params.value_pred then "cache/exp700" else "cache/exp304_no_values", 
  cache_path ::  if misc_params.value_pred then "cache/exp1000" else "cache/exp304_no_values", 
  

};
local max_instances = if misc_params.tiny_dataset then 3500 else 1000000;



local devset_config = {
  cache_suffix :: "val", 
  data_suffix :: "dev.json",
  limit_instances_val :: if misc_params.load_less then 200 else -1,
  limit_instances :: if misc_params.load_less then 200 else -1,
  
};
local trainset_config = {
  cache_suffix :: "train",
  data_suffix :: "train_spider.json",
  limit_instances :: -1,
  limit_instances_val :: -1,
};



local dataset_path = "dataset/";





local max_steps = misc_params.max_steps;
local examples = 7000;

local setting = large_setting + if misc_params.train_as_dev then trainset_config else devset_config;



local should_rerank = misc_params.should_rerank;


local dataset_reader_name = "smbop";

{
  "dataset_reader": {
    "type": dataset_reader_name,
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "question_token_indexers":{
            "tokens":{
              "type":"pretrained_transformer",
                    "model_name":setting.model_name,
            },

              },
    "cache_directory": setting.cache_path + "train",
    "keep_if_unparsable": false,
    "max_instances": max_instances,
    "limit_instances" : setting.limit_instances, 
    "value_pred":misc_params.value_pred,
  },
  "validation_dataset_reader": {
    "type": dataset_reader_name,
    "question_token_indexers":{
            "tokens":{
              "type":"pretrained_transformer",
              "model_name":setting.model_name,
              
            },
      },
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "cache_directory": setting.cache_path + setting.cache_suffix,
    "lazy": false,
    "keep_if_unparsable": true,
    "max_instances": max_instances,
    "limit_instances" : setting.limit_instances_val,
    "value_pred":misc_params.value_pred,
  },
  "train_data_path": dataset_path + "train_spider.json",
  "validation_data_path": dataset_path + setting.data_suffix,

  "model": {
    "experiment_name": std.extVar('experiment_name'),
    "type": "smbop_parser",
    "beam_size" : stringToInt("beam_size"),
     
     "decoder_timesteps" : 9,
    "misc_params": misc_params,
    "question_embedder": {
        "token_embedders":{
          "tokens":{
              "type":"pretrained_transformer",
              "model_name":setting.model_name,
              "gradient_checkpointing":true,
            },
          },
      },
    "beam_encoder": {
      "type": "pytorch_transformer",
      "input_dim": misc_params.embedding_dim,
      "num_layers": misc_params.beam_encoder_num_layers,
      "feedforward_hidden_dim": 4*misc_params.embedding_dim,
      
      "num_attention_heads": misc_params.num_heads,
      "dropout_prob": stringToFloat("dropout"),
    },
    "tree_rep_transformer": {
      "type": "pytorch_transformer",
      "input_dim": misc_params.embedding_dim,
       "positional_encoding":"embedding",
      "num_layers": misc_params.tree_rep_transformer_num_layers,
      "feedforward_hidden_dim": 4*misc_params.embedding_dim,
      "num_attention_heads": misc_params.num_heads,

    },
    "beam_summarizer": {
      "type": "pytorch_transformer",
      "input_dim": misc_params.embedding_dim,
      "num_layers": 1,
      "feedforward_hidden_dim": 4*misc_params.embedding_dim,
      "num_attention_heads": misc_params.num_heads,

    },
    "schema_encoder": {
      "type": "relation_transformer",
      "hidden_size": misc_params.embedding_dim,
      "ff_size": 4*misc_params.embedding_dim,
      "num_layers": setting.rat_layers,
      "tfixup" : misc_params.tfixup,
      "dropout":misc_params.rat_dropout,
    },
    "utterance_augmenter": {
      "type": "cross_attention",
      "hidden_size": misc_params.embedding_dim,
      
      "num_attention_heads":misc_params.num_heads,
      "attention_probs_dropout_prob":stringToFloat("dropout"),
      "ctx_dim": misc_params.embedding_dim,
      "hidden_dropout_prob":stringToFloat("dropout"),
    },
    "dropout": stringToFloat("dropout"),
  },
  "data_loader": {
     "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["enc","depth"],
        "batch_size" : setting.batch_size,
    },
  },
  "validation_data_loader": {
    "batch_size" : 20,
    "shuffle": true,
  },
  "trainer": {
    "grad_norm": misc_params.grad_norm,
    "grad_clipping": misc_params.grad_clip,
    
    "use_amp":misc_params.amp,
    "num_epochs": std.floor((max_steps*setting.batch_size*setting.grad_acum)/examples),
    "cuda_device": std.parseInt(std.extVar('gpu')),
    "patience": 100,
    "validation_metric":  "+spider",

    
  
  "num_gradient_accumulation_steps" : setting.grad_acum,
  "checkpointer": {"num_serialized_models_to_keep": 1},
    "optimizer": {
              "type": std.extVar("optimizer") ,
              
              "lr": misc_params.lr,
              "parameter_groups": [
                [["question_embedder"], {"lr": misc_params.lm_lr}] 
                  ],
            },
    "learning_rate_scheduler": scheduler,
  },

}
