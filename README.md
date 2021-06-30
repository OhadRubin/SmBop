# SmBoP: Semi-autoregressive Bottom-up Semantic Parsing 


Author implementation of this [NAACL 2021 paper](https://arxiv.org/abs/2010.12412).

## Install & Configure

1. Install pytorch 1.8.1 that fits your CUDA version 

    
2. Install the rest of required packages
    ```
    pip install -r requirements.txt
    ```
    
3. Run this command to install NLTK punkt.
    ```
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    ```

4. Download the Spider dataset with the following command:
    ```
    bash scripts/download_spider.sh 
    ```

## Training the parser
Use the following command to train:
```
python exec.py 
``` 

First time loading of the dataset might take a while (a few hours) since the model first loads values from tables and calculates similarity features with the relevant question. It will then be cached for subsequent runs. Use the `disable_db_content` argument to reduce the pre-processing time in exchange of not performing IR on some (incredibly large) tables.


## Evaluation
To create predictions run the following command:
```
python eval.py --archive_path {model_path} --output preds.sql
``` 
To run the evalutation with the official spider script:

```
python smbop/eval_final/evaluation.py --gold dataset/dev_gold.sql --pred preds.sql --etype all --db  dataset/database  --table dataset/tables.json
``` 

## Pretrained model
You can download a pretrained model from [here](https://drive.google.com/file/d/1pQvg2sT7h9t_srgmN1nGGMfIPa62U9ag/view?usp=sharing).
It achieves the following results on the offical script:

```
                     easy                 medium               hard                 extra                all                 
count                248                  446                  174                  166                  1034                
=====================   EXECUTION ACCURACY     =====================
execution            0.883                0.791                0.684                0.530                0.753             

====================== EXACT MATCHING ACCURACY =====================
exact match          0.883                0.791                0.655                0.512                0.746
``` 

## Demo
You can run SmBoP on a Google Colab notebook [here](https://colab.research.google.com/drive/1KGlETGn9wngUPQrkFfa7ySecU-t_I3Y2#scrollTo=X1v6F3TlOMKH).


### Docker
You could also use the demo with docker:
```
docker build -t smbop .
docker run -it --gpus=all smbop:latest
```

This will create a infrence terminal similar to the Google Colab demo, you could  run for example:
```
>>>inference("Which films cost more than 50 dollars or less than 10?","cinema")
SELECT film.title FROM schedule JOIN film ON schedule.film_id = film.film_id WHERE schedule.price > 50 OR schedule.price<10
```
<!-- 


1

You should get results similar to the following (the `sql_match` is the one measured in the official evaluation test):
```
  "best_validation__match/exact_match": 0.3911764705882353,
  "best_validation_sql_match": 0.4931372549019608,
  "best_validation__others/action_similarity": 0.5847554769212673,
  "best_validation__match/match_single": 0.6383763837638377,
  "best_validation__match/match_hard": 0.3284518828451883,
  "best_validation_beam_hit": 0.6127450980392157,
  "best_validation_loss": 8.254135131835938
  "best_epoch": 71
```

## Training the re-ranker

1. First, you will need to run the trained parser to output a set of candidates for each one of the spider examples.
This will be the dataset that the re-ranker is trained on.

Use the following AllenNLP command to create the training dataset (this currently requires a few hours to produce,
and will require a few optimizations or reducing beam size to improve this running-time):

```
allennlp predict experiments/experiment dataset/train_spider.json \
--use-dataset-reader --predictor spider_candidates --cuda-device=0 --silent \
--output-file experiments/experiment/candidates_train.json \
--include-package models.semantic_parsing.spider_parser \
--include-package dataset_readers.spider \
--include-package predictors.spider_predictor_candidates \ 
--weights-file experiments/experiment/best.th \
-o "{\"dataset_reader\":{\"keep_if_unparsable\":true}}"
```

3. Use the following AllenNLP command to create the validation dataset:

```
allennlp predict experiments/experiment dataset/dev.json \
--use-dataset-reader --predictor spider_candidates --cuda-device=0 --silent \
--output-file experiments/experiment/candidates_dev.json \
--include-package models.semantic_parsing.spider_parser \
--include-package dataset_readers.spider \
--include-package predictors.spider_predictor_candidates \ 
--weights-file experiments/experiment/best.th \
-o "{\"dataset_reader\":{\"keep_if_unparsable\":true}}"
```

4. Use the following AllenNLP command to train the re-ranker:
```
allennlp train train_configs/defaults_rerank.jsonnet -s experiments/experiment_rerank \
--include-package models.semantic_parsing.spider_reranker \
--include-package dataset_readers.spider_rerank
```

You should get results similar to the following:
```
  "best_query_accuracy": 0.528046421663443,
  "best_query_accuracy_single": 0.6660869565217391,
  "best_query_accuracy_multi": 0.355119825708061,
  "best_validation_loss": 8.254135131835938
  "best_epoch": 82,
```

## Trained models

You can skip the above steps and download our trained models:
https://drive.google.com/open?id=1NdSubOVx6IsCpNvkzjTPovsIHEuuebyi

This includes (1) the parser model, (2) the output train/dev candidates and (3) the re-ranker model. 

## Inference

Use the following AllenNLP command to output a file with the predicted queries.

This will require both models (parser and re-ranker) to exist, but will work without the candidates files (it creates
the queries candidates in the process).

```
allennlp predict experiments/experiment dataset/dev.json \
--predictor spider_predict_complete \
--use-dataset-reader \
--cuda-device=0 \
--output-file output.sql \
--silent \
--include-package models.semantic_parsing.spider_parser \
--include-package dataset_readers.spider \
--include-package predictors.spider_predictor_complete \
--weights-file experiments/experiment/best.th \
-o "{\"dataset_reader\":{\"keep_if_unparsable\":true}}"
``` -->
