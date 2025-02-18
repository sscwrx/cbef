from dataclasses import dataclass
from abc import ABC, abstractmethod 
from pathlib import Path 
from typing import Tuple
@dataclass
class BaseConfig: 
    """Base configuration class."""

    """where the output files will be saved"""
    output_dir: str = Path("./output")

    def __str__(self):
        """just for pretty print() """
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
    