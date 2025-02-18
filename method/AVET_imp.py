import numpy as np 


def absolute_value_equations_transform(x: np.ndarray,seed:int=1) -> np.ndarray:
    assert len(x.shape) == 1, "Input must be a 1D array,please use np.squeeze() to convert it." 

    n = np.floor(len(x)/2).astype(int)
    
    u = x[:n].copy()
    v = x[n:2*n].copy()

    assert u.shape == v.shape, f"u v must have the same shape. u.shape = {u.shape}, v.shape = {v.shape}"
    if np.all(np.equal(u,v)): v=v + 1e-6

    rng = np.random.default_rng(seed)
    R = rng.normal(loc=0,scale=(1/ np.sqrt(n)),size=(n,n))
    A = rng.normal(loc=0,scale=(1/ np.sqrt(n)),size=(n,n))
    B = rng.normal(loc=0,scale=(1/ np.sqrt(n)),size=(n,n))  
    y =  A @ u + B @  np.abs(R @ v)

    return y , R , A , B

def bi_avet(x,seed=1):
    return np.where( absolute_value_equations_transform(x,seed)[0] > 0, 1, 0)

def in_avet(x,k=300,g=16,seed=1):
    assert len(x.shape) == 1, "Input must be a 1D array,please use np.squeeze() to convert it." 

    n = np.floor(len(x)/2).astype(int)
    
    u = x[:n].copy()
    v = x[n:2*n].copy()

    assert u.shape == v.shape, f"u v must have the same shape. u.shape = {u.shape}, v.shape = {v.shape}"
    if np.all(np.equal(u,v)): v=v + 1e-6

    rng = np.random.default_rng(seed)
    output = np.zeros(k,)
    
    R = rng.normal(loc=0,scale=(1/ np.sqrt(n)),size=(k,n,n))
    A = rng.normal(loc=0,scale=(1/ np.sqrt(n)),size=(k,g,n))
    B = rng.normal(loc=0,scale=(1/ np.sqrt(n)),size=(k,g,n))  

    for i in range(k):
        y =  A[i] @ u + B[i] @  np.abs(R[i] @ v)
        output[i] = np.argmax(y)
    return output.astype(np.int32)
def biohash( feat_vec, bh_len=40, user_seed=1):
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


if __name__ == "__main__":
    x = np.random.rand(299,)
    print(len(x))
    print(biohash(x,bh_len=len(x)).shape)
    print(biohash(x))