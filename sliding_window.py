import numpy as np

class SlidingWindowMechanism:
    def __init__(self, window_size):
        """
        初始化滑动窗口机制
        
        Args:
            window_size (int): 滑动窗口大小k
        """
        self.window_size = window_size
        
    def get_windows(self, feature_vector):
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
    
    def find_min_indices(self, windows):
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
    
    def find_max_indices(self, windows):
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
    
    def calculate_matching(self, indices1, indices2):
        """
        计算两个索引序列的匹配度
        
        Args:
            indices1 (array): 第一个索引序列
            indices2 (array): 第二个索引序列
            
        Returns:
            array: 最小值的匹配结果
        """
        if len(indices1) != len(indices2):
            raise ValueError("索引序列长度不匹配")
            
        # 计算最小值匹配度
        M_min = np.where(indices1 == indices2, 1, 0)
        
        return M_min
    
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

