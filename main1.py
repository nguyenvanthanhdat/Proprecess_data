from dataclasses import field, dataclass
from typing import Optional

@dataclass
class DataArguments:
    """
    Arguments for setup data
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of dataset"}
    )
    # dataset_config_name
    # dataset_languge
    # validation_split_percentage
    # streaming
    # shuffle
    def __post_init__(self):
        # check the input
        pass

@dataclass
class preprocess_vi(DataArguments):
    def load_dataset(self):
        dataset_name = self.dataset_name
        dataset_config_name = self.dataset_config_name
        dataset_languge = self.dataset_languge
        streaming = self.streaming
        datasets = load_dataset('wiki_lingua', 'vietnamese', split='train', streaming=True)

def main():
    print(DataArguments.dataset_name.metadata["help"])

if __name__ == "__main__":
    main()