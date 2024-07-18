import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

class LanguageModelTrainer:
    def __init__(self, dataset_path, model_name, max_seq_length, output_dir):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.training_dataset = None
        self.model = None
        self.tokenizer = None

    def load_dataset(self):
        self.training_dataset = load_dataset("csv", data_files=self.dataset_path, split='train')

    def load_model_and_tokenizer(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True
        )

    def configure_model(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
            max_seq_length=self.max_seq_length
        )

    def train(self):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.training_dataset,
            dataset_text_field="Text",
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=TrainingArguments(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                num_train_epochs=10,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                output_dir=self.output_dir,
                optim="adamw_8bit",
                seed=3407,
            )
        )
        trainer.train()

    def run(self):
        self.load_dataset()
        self.load_model_and_tokenizer()
        self.configure_model()
        self.train()

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a language model.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training dataset (CSV file).')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pretrained model.')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the trained model.')

    args = parser.parse_args()

    trainer = LanguageModelTrainer(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir
    )
    trainer.run()

if __name__ == "__main__":
    main()
