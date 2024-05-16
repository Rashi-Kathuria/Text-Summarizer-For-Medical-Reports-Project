import os
from textSummarizer.logging import logger
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict, Features, Value
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, examples):
        # Ensure 'article' and 'abstract' are strings
        articles = [str(article) for article in examples['article']]
        abstracts = [str(abstract) for abstract in examples['abstract']]

        # Tokenize with truncation and padding
        inputs = self.tokenizer(
            articles,
            abstracts,
            max_length=1024,
            truncation=True,
            padding='max_length'
        )
        return inputs

    def convert(self):
        # Load dataset folder
        dataset_path = self.config.data_path
        train_df = pd.read_csv(os.path.join(dataset_path, 'train', 'train.csv'))
        test_df = pd.read_csv(os.path.join(dataset_path, 'test', 'test.csv'))
        val_df = pd.read_csv(os.path.join(dataset_path, 'validation', 'validation.csv'))

        # Define features
        features = Features({
            "article": Value("string"),
            "abstract": Value("string")
        })

        # Create datasets from DataFrames
        train_dataset = Dataset.from_pandas(train_df, features=features)
        test_dataset = Dataset.from_pandas(test_df, features=features)
        val_dataset = Dataset.from_pandas(val_df, features=features)

        # Create DatasetDict
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
            "validation": val_dataset
        })

        # Map each dataset to features
        dataset = dataset.map(self.convert_examples_to_features, batched=True)

        # Save processed datasets
        output_path = os.path.join(self.config.root_dir, "dataset")
        os.makedirs(output_path, exist_ok=True)
        dataset.save_to_disk(output_path)
        print("Dataset saved successfully!")


