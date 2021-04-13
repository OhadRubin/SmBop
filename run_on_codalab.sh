pip install -r  req_codalab.txt
python -m nltk.downloader 'punkt'
python -m nltk.downloader 'stopwords'
echo "Start running"
# export CUDA_VISIBLE_DEVICES=$2 
python smbop/eval.py --output $1 --archive_path $5 --dataset_path  "database" --table_path "data/tables.json" --dev_path  $4 --gpu $2
python smbop/eval_final/evaluation.py --gold $3 --pred $1 --etype all --db  database  --table data/tables.json

# python -m codalab.bin.cl run  "req_codalab.txt:0xafe782"  "data:0x03a342" "dev_gold.sql:0x782a1b" "run_on_codalab.sh:0xc66119" "smbop:0x78a186" "model_tar:0x2591cb" "database:0x853385" 'sh run_on_codalab.sh predictions.txt 0 dev_gold.sql data/dev.json'  --request-network --request-docker-image floydhub/pytorch:1.6.0-gpu.cuda10cudnn7-py3.56  --request-time 20m --request-gpus 1 --request-memory 16g --request-disk 3g