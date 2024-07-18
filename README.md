# Fine_Tune_QA
Fine tune LLM on HuggingFace dataset for Question-Answering

#Dataset Preparation
```bash
python Pre-Processing_Dataset.py --dataset Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning --split train --drop_columns prompt_word_count response_word_count --output_file data.csv
```
#Fine-tune LLM
```bash
python fine_tune.py --dataset_path path/to/data.csv --model_name unsloth/mistral-7b-bnb-4bit --max_seq_length 2048 --output_dir unsloth-test
```
#Inference
```bash
python inference.py
```



