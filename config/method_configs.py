from typing import Dict,List,Tuple
from method.base_method import MethodConfig
from method.bio_hash import BioHashConfig
from method.bi_avet import BiAVETConfig
from method.in_avet import InAVETConfig
from method.swg import SWGConfig
from verification.verify import VerifierConfig


method_names: List[str] = ["BioHash","AVET","Bi_AVET","In_AVET","C_IOM"]

METHOD_CONFIGS: Dict[str,Tuple[MethodConfig,VerifierConfig]] = {
    "SWG": (SWGConfig(),VerifierConfig(measure="hamming")),
    "BioHash": (BioHashConfig(),VerifierConfig(measure="hamming")),
    "AVET": (SWGConfig(),VerifierConfig(measure="cosine")),
    "Bi_AVET": (BiAVETConfig(),VerifierConfig(measure="hamming")),
    "In_AVET": (InAVETConfig(),VerifierConfig(measure="jaccard")),
    "C_IOM": (SWGConfig(),VerifierConfig(measure="cosine"))
}