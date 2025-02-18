from base_method import BaseMethod, MethodConfig



class BioHashConfig(MethodConfig):
    """Configuration class for BioHash method."""

    """The desired length of the resulting BioHashCode."""
    bh_len: int = 40

class BioHash(BaseMethod):
    config: BioHashConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def process_feature( feature_vector, seed=1):
        """ Creates a BioHash by projecting the input biometric feature vector onto a set of randomly generated basis vectors and then binarising the resulting vector

        **Parameters:**

        feat_vec (array): The extracted fingervein feture vector

        bh_len (int): The desired length (i.e., number of bits) of the resulting BioHash

        user_seed (int): The seed used to generate the user's specific random projection matrix

        **Returns:**

        biohash (array): The resulting BioHash, which is a protected, binary representation of the input feature vector

        """

        np.random.seed(user_seed) # re-seed the random number generator according to the user's specific seed
        rand_mat = np.random.rand(len(feat_vec), bh_len) # generate matrix of random values from uniform distribution over [0, 1] 
        orth_mat, _ = np.linalg.qr(rand_mat, mode='reduced') # orthonormalise columns of random matrix, mode='reduced' returns orth_mat with size len(feat_vec) x bh_len    
        biohash = np.dot(feat_vec, orth_mat)
        thresh = np.mean(biohash) # threshold by which to binarise vector of dot products to generate final BioHash
        biohash = np.where(biohash > 0, 1, 0)
        return biohash
