from typing import Dict,List
from method.base_method import MethodConfig
from method.bio_hash import BioHashConfig

method_names = ["BioHash","AVET","Bi_AVET","In_AVET","C_IOM"]

methods_config = {
    "BioHash":BioHashConfig(),
}