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

if __name__ == "__main__":
    x = np.random.rand(299,)
    print(bi_avet(x).shape)
    print(in_avet(x))