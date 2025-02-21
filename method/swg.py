import code
from matplotlib.pylab import permutation
from numpy._typing._array_like import NDArray
from method.base_method import BaseMethod, MethodConfig
from dataclasses import dataclass, field
from typing import Type,List 
import numpy as np 
import scipy as sp 
from scipy.linalg import hadamard

@dataclass
class SWGConfig(MethodConfig):

    _target: Type = field(default_factory= lambda: SWG)
    method_name:str = "SWG"
    N:int = 512 
    """ the length of the feature vector and shape of the Walsh matrix"""
    w:int = 2 # the shift length
    n:int = 4 # the number of the hash functions
    r:int = 256 # the number of the rows of the Walsh matrix



class SWG(BaseMethod):
    config: SWGConfig

    def __init__(self, config):
        self.config = config
        self._walshmatrix = self._walsh(self.config.N)
        self.seed:int = 1

    def set_seed(self, seed:int):
        """Set the global seed, generate or update the parameters."""
        self.seed = seed
        np.random.seed(self.seed)
        self.part_walsh:NDArray = np.zeros((self.config.r,self.config.N),dtype=np.int8)
        self.n_part_walsh_vector:NDArray = np.zeros(self.config.n*self.config.r)
        self.permutation_n_part_walsh_vector:NDArray[np.float64] = np.empty(self.config.n*self.config.r,dtype=int)
        # 随机生成的参数
        self.perm:NDArray = self._randperm(self.config.N, self.config.r)
        self.permutation_seed = np.random.permutation(self.config.n*self.config.r)

    @property
    def get_seed(self):
        return self.seed
    
    def _gen_gray_code(self,N:int)->NDArray:
        """Generate N-bit Gray code."""
        sub_gray = np.array([[0], [1]])
        for n in range(2, N + 1):
            top_gray: NDArray = np.hstack((np.zeros((2**(n-1), 1)), sub_gray))
            bottom_gray:NDArray = np.hstack((np.ones((2**(n-1), 1)), sub_gray[::-1]))
            sub_gray:NDArray = np.vstack((top_gray, bottom_gray))
        return sub_gray

    def _walsh(self,m) -> NDArray:
        """Generate an m x m Walsh matrix."""
        N = int(np.log2(m))
        x = hadamard(m)  # 使用 scipy.linalg.hadamard 生成 Hadamard 矩阵
        walsh_matrix = np.zeros((m, m), dtype=int)
        graycode = self._gen_gray_code(N)
        nh1 = np.zeros((m, N), dtype=int)
        
        for i in range(m):
            q = graycode[i, :]
            nh = 0
            for j in range(N, 0, -1):
                nh1[i, j-1] = q[j-1] * 2**(j-1)
            nh = np.sum(nh1[i, :])
            walsh_matrix[i, :] = x[nh, :]
        return walsh_matrix
    
    def _randperm(self,max:int, select:int):
        a = np.arange(max)
        np.random.shuffle(a)#np.random.shuffle()第一次，设置完之后每次生成的permatrix都是一样的。
        return a[:select]
    
    def process_feature(self, feature_vector: NDArray) -> NDArray:
        
        for j in range(self.config.r):
            self.part_walsh[j,:] = self._walshmatrix[self.perm[j],:]

        code_list: List[int] = [0]*(self.config.n*self.config.r)
        # 开始生成哈希码
        walsh_vector = np.dot(self.part_walsh, feature_vector)

        for k in range(self.config.n):
            self.n_part_walsh_vector[k*self.config.r:(k+1)*self.config.r] = walsh_vector

        self.permutation_n_part_walsh_vector =self.n_part_walsh_vector[self.permutation_seed]

        for t in range(self.config.n*self.config.r):
            if self.permutation_n_part_walsh_vector[t] < self.permutation_n_part_walsh_vector[(t+self.config.w) % (self.config.n*self.config.r)]:
                code_list[t] = 0
            else:
                code_list[t] = 1

        binary_hashcode = np.array([code_list])
        binary_hashcode =np.squeeze(binary_hashcode)

        # binary_hashcode:NDArray[np.int8] = np.array([code_list[i:i+8] for i in range(0,self.config.n*self.config.r,8)],dtype=np.int8)
        # print(binary_hashcode.shape)
        return binary_hashcode


