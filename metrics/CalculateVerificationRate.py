from typing import Tuple, List
import numpy as np

def calculateVerificationRate(threshold: float, genuine: List[float], impostor: List[float]) -> Tuple[float, float, float, float]:
    """计算验证率相关指标
    
    Args:
        threshold: 判定阈值
        genuine: 真匹配相似度列表
        impostor: 假匹配相似度列表
    
    Returns:
        Tuple[float, float, float, float]: (验证率TSR, 错误接受率FAR, 错误拒绝率FRR, 真实接受率GAR)
    """
    # 将列表转换为numpy数组以进行向量化计算
    genuine_arr = np.array(genuine)
    impostor_arr = np.array(impostor)
    
    # 计算错误接受率FAR (False Accept Rate)
    fa_count = float(np.sum(impostor_arr >= threshold))
    far = float((fa_count / len(impostor)) )
    
    # 计算错误拒绝率FRR (False Reject Rate)和真实接受率GAR (Genuine Accept Rate)
    fr_count = float(np.sum(genuine_arr < threshold))
    ga_count = float(np.sum(genuine_arr > threshold))
    
    frr = float((fr_count / len(genuine)) )
    gar = float((ga_count / len(genuine)) )
    
    # 计算总错误率TER (Total Error Rate)和验证率TSR (True Success Rate)
    ter = float((fa_count + fr_count) / (len(genuine) + len(impostor)))
    tsr = float((1 - ter) )
    
    return tsr, far, frr, gar

def computePerformance(genuine: List[float], impostor: List[float], step: float, verbose: bool = False) -> Tuple[float, float, List[float], List[float]]:
    """计算性能指标和最优阈值
    
    Args:
        genuine: 真匹配相似度列表
        impostor: 假匹配相似度列表
        step: 阈值搜索步长
        verbose: 是否打印详细信息
    
    Returns:
        Tuple[float, float, List[float], List[float]]: (等错误率EER, 最优阈值, FAR列表, GAR列表)
    """
    # 确定阈值搜索范围
    start = float(min(np.min(genuine), np.min(impostor)))
    stop = float(max(np.max(genuine), np.max(impostor)))
    thresholds = np.arange(start, stop + step, step)
    
    # 存储各指标列表
    metrics = [calculateVerificationRate(float(thr), genuine, impostor) for thr in thresholds]
    tsr_list, far_list, frr_list, gar_list = zip(*metrics)
    
    # 将列表转换为numpy数组以便计算
    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    
    # 找出FAR和FRR最接近的最优阈值
    diff = np.abs(frr_arr - far_arr)
    optimal_idx = int(np.argmin(diff))
    optimal_threshold = float(thresholds[optimal_idx])
    
    # 计算等错误率EER
    eer = float((frr_arr[optimal_idx] + far_arr[optimal_idx]) / 2)
    
    if verbose:
        print(f'验证率 (Verification Rate): {tsr_list[optimal_idx]:.6f}')
        print(f'真实接受率 (Genuine Accept Rate): {gar_list[optimal_idx]:.6f}')
        print(f'错误接受率 (False Accept Rate): {far_arr[optimal_idx]:.6f}')
        print(f'错误拒绝率 (False Reject Rate): {frr_arr[optimal_idx]:.6f}')
        print(f'等错误率 (Equal Error Rate): {eer:.6f}')
    
    # 将元组转换为列表
    far_list = list(map(float, far_list))
    gar_list = list(map(float, gar_list))
    
    return eer, optimal_threshold, far_list, gar_list
