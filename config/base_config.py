from dataclasses import dataclass
from abc import ABC, abstractmethod 
from pathlib import Path 

@dataclass
class BaseConfig: 
    """Base configuration class."""

    """where the output files will be saved"""
    output_dir: str = Path("./output")
    