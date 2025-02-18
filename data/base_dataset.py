from dataclasses import dataclass
from config import Config


if __name__ == '__main__':
    base_dataset = BaseDataset(100, 1000)
    print(base_dataset)  # BaseDataset(identity_nums=100, sample_nums=1000)