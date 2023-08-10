from dataclasses import field, dataclass
from typing import Optional
from datasets import load_dataset
from abc import ABC, abstractmethod
import datasets

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
    num_sample: Optional[int] = field(
        default=10000,
        metadata={"help": "An option number of sameple for validation set, default = 0.2"})
    shuffle: Optional[bool] = field(
        default=True,
        metadata={"help": "An opiton to shuffle the data"})
    def __post_init__(self):
        dataset_name = self.dataset_name
        dataset_config_name = self.dataset_config_name
        dataset_language = self.dataset_language
        streaming = self.streaming
        random_seed = self.random_seed
        shuffle = self.shuffle

    @abstractmethod

    def load_dataset(self) -> datasets.iterable_dataset.IterableDataset:
        pass

    def load_valid(self) -> datasets.iterable_dataset.IterableDataset:
        pass

@dataclass
class preprocess_vi(DataArguments):
    def __post_init__(self):
        super().__post_init__()

    def load_dataset(self):
        datasets = load_dataset(
            self.dataset_name, 
            self.dataset_language, 
            split=self.dataset_config_name, 
            streaming=self.streaming)
        return datasets

    def load_valid(self):
        datasets = load_dataset(
            self.dataset_name, 
            self.dataset_language, 
            split=self.dataset_config_name, 
            streaming=self.streaming)
        if self.shuffle == True:
            shuffled_dataset = datasets.shuffle(
                seed=self.random_seed, buffer_size=self.num_sample)
            return shuffled_dataset
        return datasets
    

def main():
    wikilingua = preprocess_vi(
        dataset_name='wiki_lingua',
        dataset_config_name = 'train',
        dataset_language = 'vietnamese',
        streaming = True)
    train_set = wikilingua.load_dataset()
    valid_set = wikilingua.load_valid()
    print(next(iter(train_set)))
    print(next(iter(train_set)))
    print(next(iter(valid_set)))
    
    print("Run all complete !!!")

if __name__ == "__main__":
    main()