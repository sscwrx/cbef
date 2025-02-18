import numpy as np
from sliding_window import SlidingWindowMechanism
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path 
from tqdm import tqdm
import scipy as sp
from copy import deepcopy
from base_method import BaseMethod, MethodConfig
from dataclasses import dataclass
from typing import Optional,List,Union,Literal
@dataclass
class ChaosBasedIoMConfig(MethodConfig):
    """ Configuration class for ChaosBasedIoM """

    """ feature vector dimension"""
    dimension: int = 512
    """ sliding window size """
    window_size: int = 3
    """ initial conditions of the chaotic system """
    intial_conditions: tuple = (0.98, 0.21, 0.46)
    """ length of the chaos sequence """
    sequence_length: int = 10000
    """ output file path of the chaos sequence """
    sequence_outout: Path = MethodConfig.output_dir / "sequence.npy"
    """ use min or max index of the sliding window """
    IoM_type: Literal["min","max"] = "min"

    """ memristor parameters same as the paper except for the dt."""
    dt: float = 0.001
    R_off:float = 20e3
    R_on:float = 1e2
    M0:float = 16e3
    u_v:float = 1e-14 
    D:float= 1e-8 
    beta:float = ((R_on - R_off)* u_v * R_on) / D**2
    N_3:float = - (R_off - M0)**2 / (2 * beta)
    N_4:float = - (R_on - M0)**2 / (2 * beta)
    N_5:float =  (R_off**2 - M0**2) / (2 * beta)
    N_6:float =  (R_on**2 - M0**2) / (2 * beta)

    a:float = 35.0
    b:float = 3.0


class ChaosBasedIoM(BaseMethod):
    """ Chaos-based IoM protection method """

    config: ChaosBasedIoMConfig

    def __init__(self,config: ChaosBasedIoMConfig):
        self.config = config
        # 生成混沌序列
        try:
            self.x_I, self.y_I, self.z_I = np.load(self.config.sequence_outout, allow_pickle=True).item().values()
            print("Load sequence from file.")
        except Exception as e:
            print("Generate sequence.") 
            self.x_I, self.y_I, self.z_I = self._generate_chaos_sequences(self.sequence_length)


            np.save(self.config.sequence_outout, {"u": self.x_I, "v": self.y_I, "w": self.z_I}) 

        self.Pi_n = self._find_dimension_indices(self.x_I)

    def _memristor_nonlinear_function(self, epsilon):

        if epsilon < self.N_5:
            return (epsilon - self.N_3) / self.R_off 
        elif  self.N_5<= epsilon <= self.N_6:
            return  (np.sqrt(2*self.beta *epsilon + self.M0**2) - self.M0) / self.beta 
        elif self.N_6 < epsilon:
            return (epsilon - self.N_4) / self.R_on 
        else:
            raise Exception( f"The input value of memristor_nonlinear_funciton is invalid")
    def _memristor_chaotic_map(self,last_u, last_v, last_w):
        u_dot = self.a * (last_v - last_u)
        h = self._memristor_nonlinear_function(-np.abs(last_u))
        v_dot = (self.c - self.a) * self.alpha * h - last_u *  last_w + self.c * last_v 
        w_dot = last_u * last_v - self.b * last_w 
        
        return u_dot, v_dot, w_dot 

    def _generate_chaos_sequences(self, length):
        """
        生成三个混沌序列
        
        Args:
            length (int): 序列长度
            
        Returns:
            tuple: (x_I, y_I, z_I) 三个混沌序列
        """

        u_seq = np.zeros(length)
        v_seq = np.zeros(length)
        w_seq = np.zeros(length)
        
        u_seq[0] = self.u0
        v_seq[0] = self.v0
        w_seq[0] = self.w0

        # 生成混沌序列
        for i in tqdm(range(1, length),desc="Generating chaos sequences"):
            u_dot,v_dot,w_dpt = self._memristor_chaotic_map(u_seq[i - 1], v_seq[i - 1], w_seq[i - 1])
            u_seq[i] = u_seq[i - 1] + u_dot * self.dt
            v_seq[i] = v_seq[i - 1] + v_dot * self.dt
            w_seq[i] = w_seq[i - 1] + w_dpt * self.dt
            # print(f" i = {i}, u = {u_seq[i]}, v = {v_seq[i]}, w = {w_seq[i]}")
        # 生成整数序列（从1到dimension）
        x_I = np.floor(((u_seq + v_seq + 100) % 1) * 10**16) % (self.dimension + 1)
        y_I = np.floor(((u_seq + w_seq + 100) % 1) * 10**16) % (self.dimension + 1)
        z_I = np.floor(((v_seq + w_seq + 100) % 1) * 10**16) % (self.dimension + 1)
        
        # 保存结果

        return x_I.astype(int), y_I.astype(int), z_I.astype(int)
    
    def plot_chaos_sequences(self, length,type="3d"):
        """
        绘制混沌序列,3d图
        
        Args:
            length (int): 序列长度
        """
        import matplotlib.pyplot as plt

        # 生成混沌序列
        dt = 0.001
        u_seq = np.zeros(length)
        v_seq = np.zeros(length)
        w_seq = np.zeros(length)
        
        # 设置初始值
        u_seq[0] = self.u0
        v_seq[0] = self.v0
        w_seq[0] = self.w0
        
        # 生成混沌序列
        for i in range(1, length):
            u_dot,v_dot,w_dot = self._memristor_chaotic_map(u_seq[i-1], v_seq[i-1], w_seq[i-1])
            u_seq[i] = u_seq[i-1] + u_dot * dt
            v_seq[i] = v_seq[i-1] + v_dot * dt
            w_seq[i] = w_seq[i-1] + w_dot * dt

        # 创建3D图
        if type == "3d":
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制轨迹
            ax.plot(u_seq, v_seq, w_seq, 'b-', lw=0.5)
            
            ax.set_xlabel('u')
            ax.set_ylabel('v')
            ax.set_zlabel('w')
            ax.set_title('Memristor Chaos System Phase Space')
            
            plt.show()
        elif type == "2d": # u w
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.plot(u_seq, w_seq, 'b-', lw=0.5)
            ax.set_xlabel('u')
            ax.set_ylabel('w')
            ax.set_title('Memristor Chaos System Phase Space')
            plt.show()
    def _find_dimension_indices(self, x_I):
        """
        找到x_I中等于dimension的所有索引
        
        Args:
            x_I (array): 混沌序列
            
        Returns:
            array: 满足条件的索引数组
        """
        indices = np.where(x_I == self.dimension)[0]
        assert len(indices)  >=self.dimension, f"Pi_n length {len(indices)} is not great than dimension {self.dimension},please update chaos sequence length larger."
        return indices
    
    def _generate_chaos_seed(self,i=1):
        """
        生成随机混沌种子 Ⅲ.B C_s = G( Pi_n(i) ) 
        假设 G 是对P_n的一个随机排列
        Args:
            Pi_n (array): 索引数组
            i (int): random chaos seed的索引
            
        Returns:
            C_s(array(int)): 随机混沌种子 (n,)  n<=d 
        """
        # 随机 permutation
        assert len(self.Pi_n) >= i, f"Pi_n length {len(self.Pi_n)} is not great than i {i},please update chaos sequence length larger."
        rng = np.random.default_rng(seed=self.Pi_n[i])
        C_s = rng.permutation(np.arange(0,self.dimension))
    
        # print(C_s)
        return C_s

    
    def _process_shuffle_indices(self, y_I):
        """
        处理洗牌索引序列
        
        Args:
            y_I (array): 初始索引序列
            
        Returns:
            array: 处理后的完整索引序列（长度为dimension）
        """
        # algorithm (6)
        # 确保所有索引都在1到dimension之间，并去除重复值
        unique_indices = np.unique(y_I[(y_I >= 1) & (y_I <= self.dimension)])
        valid_indices = np.sort(unique_indices)
        # 如果valid_indices超过了dimension，只保留前dimension个
        if len(valid_indices) > self.dimension:
            valid_indices = valid_indices[:self.dimension]

        # algorithm (7)
        all_indices = np.arange(1, self.dimension + 1)  
        missing_indices = np.setdiff1d(all_indices, valid_indices)
        
        # 如果合并后的长度会超过dimension，只使用部分missing_indices
        remaining_slots = self.dimension - len(valid_indices)
        if remaining_slots > 0:
            missing_indices = missing_indices[:remaining_slots]
            
        # 合并并返回完整的索引序列
        result = np.concatenate([valid_indices, missing_indices])

        # 1: d/2 , d/2 : d 调换

        
        # 确保结果长度正确
        assert len(result) == self.dimension, f"处理后的索引序列长度({len(result)})与维度({self.dimension})不匹配"
        return result
    
    def _generate_facecode(self, feature_vector, y_I):
        """
        生成FaceCode
        
        Args:
            feature_vector (array): 原始特征向量
            y_I (array): 洗牌索引序列
            
        Returns:
            array: 生成的FaceCode
        """
        if len(feature_vector) != self.dimension:
            raise ValueError(f"特征向量维度({len(feature_vector)})与预设维度({self.dimension})不匹配")
            
        # 使用处理后的y_I对特征向量重新排列 # algorithm (8)
        processed_y_I = self._process_shuffle_indices(y_I)
        # - 1 是因为y_I中的元素是从1开始的。 
        # + 1 是因为 : 切片是左闭右开区间，要完整覆盖到d/2 
        left_part = feature_vector[ processed_y_I[0 : self.dimension//2 + 1 : 1 ] -1] # 1: d//2 
        reversed_right_part = feature_vector[ processed_y_I[self.dimension : self.dimension//2 :-1] -1] # d: d//2 
        facecode = np.concatenate([left_part,reversed_right_part])
        assert len(facecode) == self.dimension, f"faceCode长度({len(facecode)})与维度({self.dimension})不匹配" 
        return facecode

    def protect_feature(self, feature_vector,seed_index=1):
        """
        完整的特征保护过程
        
        Args:
            feature_vector (array): 原始特征向量
            seed_index (int): 混沌种子索引
            IoM_type (str): 滑动窗口类型    "min" or "max"
            
        Returns:
            tuple: (FaceCode, chaos_seed(C_s), protected_template) 保护后的特征、混沌种子和受保护模板
        """

        # 生成混沌种子 C_s = G( Pi_n(i) )   Pi_n 是一个数组，包含所有的索引，这些索引是 x_I 中 数值等于dimension的元素的位置
        chaos_seed = self._generate_chaos_seed(i=seed_index)
        # print("chaos_seed:",chaos_seed)
        # 生成FaceCode
        facecode = self._generate_facecode(feature_vector, self.y_I)
        
        # 根据chaos_seed 使用滑动窗口生成受保护模板
        # print(chaos_seed)
        protected_template = self.sliding_window.process_feature(facecode,chaos_seed,IoM_type=self.config.IoM_type)
        return protected_template


class SlidingWindowMechanism:
    def __init__(self, window_size):
        """
        初始化滑动窗口机制
        
        Args:
            window_size (int): 滑动窗口大小k
        """
        self.window_size = window_size
        
    def _get_windows(self, feature_vector):
        """
        获取所有滑动窗口
        
        Args:
            feature_vector (array): 输入的特征向量
            
        Returns:
            list: 所有滑动窗口列表
        """
        windows = []
        for i in range(len(feature_vector) - self.window_size + 1):
            window = feature_vector[i:i + self.window_size]
            windows.append(window)
        return windows
    
    def _find_min_indices(self, windows):
        """
        计算每个窗口的最小值索引
        
        Args:
            windows (list): 滑动窗口列表
            
        Returns:
            array: 最小值索引序列
        """
        min_indices = []
        for window in windows:
            # 找到窗口内最小值的索引（从0开始）
            min_idx = np.argmin(window)
            min_indices.append(min_idx)
        return np.array(min_indices)
    
    def _find_max_indices(self, windows):
        """
        计算每个窗口的最大值索引
        
        Args:
            windows (list): 滑动窗口列表
            
        Returns:
            array: 最大值索引序列
        """
        max_indices = []
        for window in windows:
            # 找到窗口内最大值的索引（从0开始）
            max_idx = np.argmax(window)
            max_indices.append(max_idx)
        return np.array(max_indices)

    
    def process_feature(self, feature_vector,chao_seed:np.ndarray,IoM_type="min"):
        """
        处理特征向量，生成最小值索引序列
        
        Args:
            feature_vector (array): 输入的特征向量
            
        Returns:
            array: 最小值索引序列
        """
        # 根据混沌种子打乱特征向量
        S_y = feature_vector[chao_seed]

        # 创建滑动窗口，获取每个窗口的最小值索引 
        windows = self.get_windows(S_y)
        if IoM_type == "min":
            min_indices = self.find_min_indices(windows)
            return min_indices
        elif IoM_type == "max":
            max_indices = self.find_max_indices(windows)
            return max_indices
        else:
            raise ValueError("IoM_type参数错误")


if __name__ == "__main__":
    config = ChaosBasedIoMConfig()