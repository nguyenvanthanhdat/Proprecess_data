from dataclasses import field, dataclass
from typing import Optional
from datasets import load_dataset
from abc import ABC, abstractmethod
import datasets
from utils.span_corruption import *
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from utils.model_utils import tokenize_function, restructure_example
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
    batch_size: int = field(
        default= 8,
        metadata={"help": "An option number in batch size"})
    before_mask_input_length: Optional[int] = field(
        default=None,
        metadata={"help": "The length before mask"})
    target_length: Optional[int] = field(
        default=None,
        metadata={"help": "The length target to recover"})

    def __post_init__(self):
        # check condition before init
        pass

    @abstractmethod

    def load_dataset(self) -> datasets.iterable_dataset.IterableDataset:
        datasets = load_dataset(
            self.dataset_name, 
            self.dataset_language, 
            split=self.dataset_config_name, 
            streaming=self.streaming)
        return datasets

    def load_valid(self) -> datasets.iterable_dataset.IterableDataset:
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

    def data_collator_for_t5mlm(self):
        expanded_inputs_length, self.target_length = compute_input_and_target_lengths(
            inputs_length=128,
            noise_density=0.15,
            mean_noise_span_length=3.0)
        data_collator = DataCollatorForT5MLM(
            tokenizer=get_tokenizer('google/t5-v1_1-base'),
            noise_density=0.15,
            mean_noise_span_length=3.0,
            input_length=128,
            target_length=self.target_length,
            pad_token_id=0)
        return data_collator

@dataclass
class preprocess_vi(DataArguments):
    def __post_init__(self):
        super().__post_init__()
    
    def load_dataset(self):
        return super().load_dataset()

    def load_valid(self):
        return super().load_valid()

    def data_collator_for_t5mlm(self):
        return super().data_collator_for_t5mlm()
    

def main():
    bs = 4
    wikilingua = preprocess_vi(
        dataset_name='wiki_lingua',
        dataset_config_name = 'train',
        dataset_language = 'vietnamese',
        streaming = True)
    train_set = wikilingua.load_dataset()
    valid_set = wikilingua.load_valid()
    data_collator = wikilingua.data_collator_for_t5mlm()
    retrain_set = train_set.map(restructure_example, batched=True)
    retrain_set = retrain_set.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': get_tokenizer('google/t5-v1_1-base'),
                    'in_length': wikilingua.before_mask_input_length,
                },
                remove_columns=['article'],
            )
    
    train_set = train_set.shuffle(buffer_size=10_000, seed=42)
    
    
    train_loader = DataLoader(
        retrain_set,
        collate_fn=data_collator,
        batch_size=bs,
        # num_workers=12,
        pin_memory=True,
        drop_last=False)
    for i in train_loader:
        break
    print("Run all complete !!!")

if __name__ == "__main__":
    main()
