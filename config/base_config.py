from dataclasses import dataclass
from abc import ABC, abstractmethod 
from pathlib import Path 
from typing import Tuple, Type, Any
@dataclass
class BaseConfig: 
    """Base configuration class."""

    _target:Type 

    def setup(self,**kwargs)->Any:
        return self._target(self,**kwargs) 
    
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
