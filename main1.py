from dataclasses import field, dataclass
from typing import Optional
from datasets import load_dataset
from abc import ABC, abstractclassmethod

@dataclass
class DataArguments(ABC):
    """
    Arguments for setup data
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: str = field(
        default=None,
        metadata={"help":"The configuration name of dataset to use (via the datasets library)"})
    dataset_language: Optional[str] = field(
        default=None,
        metadata={"help": "The language use in the dataset"})
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "An optional using streaming option or not"})
    random_seed: Optional[int] = field(
        default=42,
        metadata={"help": "An optinal give the random seed to shuffer data"})
    get_valid: Optional[bool] = field(
        default=False,
        metadata={"help": "An optional to get validation set"})
    validation_split_percentage = field(
        default=0.2,
        metadata={"help": "An option to slipt dataset, default = 0.2"})
    shuffle: Optional[bool] = field(
        default=True,
        metadata={"help": "An opiton to shuffle the data"})
    def __post_init__(self):
        dataset_name = self.dataset_name
        dataset_config_name = self.dataset_config_name
        dataset_language = self.dataset_language
        streaming = self.streamingname
        random_seed = self.random_seed
        get_valid = self.get_valid
        validation_split_percentage = self.validation_split_percentage
        shuffer = self.shuffer

    @abstractclassmethod
    def load_dataset(self) -> dataset:
        pass

    def load_valid(self) -> validation_set:
        pass

@dataclass
class preprocess_vi(preprocess):
    def __post_init__(self):
        super().__post_init__()

    def load_dataset(self):
        datasets = load_dataset(dataset_name, dataset_language, split=dataset_config_name, streaming=streaming)
        return datasets

    

def main():
    args = DataArguments(
        dataset_name='wiki_lingua',
        dataset_config_name = 'train',
        dataset_language = 'vietnamese',
        streaming = True)
    preprocess_instance = preprocess_vi(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_language=args.dataset_language,
        streaming=args.streaming)
    datasets = preprocess_instance.load_dataset()
    print(next(iter(datasets)))
    print("Run all complete !!!")

if __name__ == "__main__":
    main()