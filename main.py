import time, torch
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Interface for the data arguments
@dataclass
class DataTrainingArguments:
    """
    Arguments pretraining to what data we are going to input model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The configuration name of dataset to use (via the datasets library)"})
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input traing data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the preplexity on (a text file)."})
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."})
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there;s no validation split"})
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": (
            "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
            "will be truncated. Default to the max input length of the model"
        )})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."})
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"})
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"})
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation)flie` should be a csv, a json or a txt file."

# Define the Abstract class
class preprocess(DataTrainingArguments):
    def preprocessing():
        pass

# class Vietnameese
class preprocess_vi(preprocess):
    def preprocessing():
        pass


def main():
    # data_args = DataTrainingArguments(
    #     dataset_name="my_dataset",
    #     train_file="train_data.txt",
    #     validation_file="val_data.txt",
    #     mlm_probability=0.2)
    # print(data_args)
    print(DataTrainingArguments.__dataclass_fields__['dataset_config_name'].metadata['help'])
    vietnamese_preprocessor = preprocess_vi()
    vietnamese_text = "Xin chào! Đây là một ví dụ về xử lý tiền xử lý cho tiếng Việt."
    preprocessed_vietnamese = vietnamese_preprocessor.preprocess(vietnamese_text)
    print("Preprocessed Vietnamese:", preprocessed_vietnamese)
    return
# oscar
    
if __name__ == "__main__":
    main()
