from typing import Dict,List
from method.base_method import MethodConfig
from method.bio_hash import BioHashConfig
from method.bi_avet import BiAVETConfig
from method.in_avet import InAVETConfig
from method.swg import SWGConfig


method_names: List[str] = ["BioHash","AVET","Bi_AVET","In_AVET","C_IOM"]

METHOD_CONFIGS: Dict[str,MethodConfig] = {
    "BioHash":BioHashConfig(),
    "BiAVET":BiAVETConfig(),
    "InAVET":InAVETConfig(),
    "SWG":SWGConfig()}