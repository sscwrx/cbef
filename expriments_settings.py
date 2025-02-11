from enum import Enum
from AVET_imp import bi_avet

class Measure(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    HAMMING = "hamming"
    JACCARD = "jaccard"


class Method(Enum):
    AVET = {"measure":Measure.COSINE,
            "method_name" : bi_avet}
    BI_AVET = {"measure":Measure.HAMMING}
    IN_AVET = {"measure":Measure.JACCARD}
    Bio_hash = {"measure":Measure.HAMMING}
    C_IoM = {"measure":Measure.EUCLIDEAN} 

class DataType(Enum):
    FACE = {"data_type":"face",
            "datasets":["FEI","LFW", "ColorFeret", "CASIA-WebFace"],
            "identity_number":127,
            "sample_number":12}
    
    FINGERPRINT = {"data_type":"fingerprint",
                   "datasets":["FVC2002/Db1_a","FVC2002/Db2_a","FVC2002/Db3_a",
                               "FVC2004/Db1_a","FVC2004/Db2_a","FVC2004/Db3_a"],
                   "identity_number":100,
                   "sample_number":5}

class Metrics(Enum):
    EER = "eer"
    THR = "thr" 

if __name__ == "__main__":
    # method  = Method()
    print(Method.AVET.value.get("measure"))
    print(Method.C_IoM.value.get("measure"))

    print(DataType.FACE.value.get("data_type"))
    print()
    a = Method.AVET.value.get("method_name")
    a