import argparse
import pandas as pd
from datasets import load_dataset, Dataset

class DataPreprocessor:
    def __init__(self, dataset_name, split, drop_columns, output_file):
        self.dataset_name = dataset_name
        self.split = split
        self.drop_columns = drop_columns
        self.output_file = output_file

    def load_data(self):
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        self.df = pd.DataFrame(self.dataset)

    def process_data(self):
        self.df.drop(columns=self.drop_columns, inplace=True)
        self.df['Formatted'] = self.df.apply(self.format_row, axis=1)
        self.new_df = self.df.rename(columns={"Formatted": "Text"})
        self.new_df = self.new_df[["Text"]]
        
    def save_data(self):
        self.new_df.to_csv(self.output_file, index=False)

    @staticmethod
    def format_row(row):
        question = row['prompt']
        answer = row['response']
        formatted_string = f"[INST] {question} [/INST] {answer}"
        return formatted_string

    def run(self):
        self.load_data()
        self.process_data()
        self.save_data()

def main():
    parser = argparse.ArgumentParser(description='Preprocess the dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to load')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to load')
    parser.add_argument('--drop_columns', type=str, nargs='+', required=True, help='Columns to drop')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path to save processed data')

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        dataset_name=args.dataset,
        split=args.split,
        drop_columns=args.drop_columns,
        output_file=args.output_file
    )

    preprocessor.run()

if __name__ == "__main__":
    main()
