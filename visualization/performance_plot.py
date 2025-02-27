import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
from datetime import datetime
from scipy.interpolate import make_interp_spline

def plot_roc_curve(self, far_list, gar_list, title=None, experiment_num=None):
    """绘制ROC曲线并保存到实验输出目录
    
    Args:
        far_list: 假阳性率列表(FAR)
        gar_list: 真阳性率列表(GAR)
        title: 图表标题(可选)
        experiment_num: 实验编号(可选)
    
    Returns:
        保存的图像路径
    """
    # 创建图表目录
    plots_dir = self.config._get_base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保数据按FAR排序
    sorted_indices = np.argsort(far_list)
    far_sorted = np.array(far_list)[sorted_indices]
    gar_sorted = np.array(gar_list)[sorted_indices]
    
    # 计算AUC
    roc_auc = auc(far_sorted, gar_sorted)
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    plt.plot(far_sorted, gar_sorted, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('Genuine Acceptance Rate (GAR)', fontsize=12)
    
    # 设置标题
    if title is None:
        method_name = self.config.method_config.method_name
        dataset_name = self.config.dataset_config.dataset_name
        title = f'ROC Curve: {method_name} on {dataset_name}'
    
    if experiment_num is not None:
        title += f' (Experiment {experiment_num})'
    
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_prefix = f"exp{experiment_num}_" if experiment_num is not None else ""
    filename = f"{exp_prefix}roc_{self.config.method_config.method_name}_{self.config.dataset_config.dataset_name}_{timestamp}.png"
    save_path = plots_dir / filename
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 记录到日志
    self.logger.info(f"ROC curve saved to {save_path}")
    
    return save_path

def plot_score_distributions(self, genuine_similarity_list, impostor_similarity_list, title=None, experiment_num=None, n_bins=100):
    """绘制真假匹配分数分布的直方图和平滑曲线
    
    Args:
        genuine_similarity_list: 真匹配相似度列表
        impostor_similarity_list: 假匹配相似度列表
        title: 图表标题(可选)
        experiment_num: 实验编号(可选)
        n_bins: 直方图的柱数(默认100)
    
    Returns:
        保存的图像路径
    """
    # 创建图表目录
    plots_dir = self.config._get_base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算直方图数据
    genuine_counts, genuine_bins = np.histogram(genuine_similarity_list, bins=n_bins)
    impostor_counts, impostor_bins = np.histogram(impostor_similarity_list, bins=n_bins)
    
    # 计算直方图的中心点
    genuine_bin_centers = (genuine_bins[:-1] + genuine_bins[1:]) / 2
    impostor_bin_centers = (impostor_bins[:-1] + impostor_bins[1:]) / 2
    
    # 创建平滑曲线
    genuine_xnew = np.linspace(genuine_bin_centers.min(), genuine_bin_centers.max(), 300)
    genuine_spline = make_interp_spline(genuine_bin_centers, genuine_counts, k=3)
    genuine_ynew = genuine_spline(genuine_xnew)
    
    impostor_xnew = np.linspace(impostor_bin_centers.min(), impostor_bin_centers.max(), 300)
    impostor_spline = make_interp_spline(impostor_bin_centers, impostor_counts, k=3)
    impostor_ynew = impostor_spline(impostor_xnew)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    plt.hist(genuine_similarity_list, bins=n_bins, alpha=0.3, label='Genuine Scores', color='blue')
    plt.hist(impostor_similarity_list, bins=n_bins, alpha=0.3, label='Impostor Scores', color='red')
    
    # 绘制平滑曲线
    plt.plot(genuine_xnew, genuine_ynew, label='Genuine Distribution', color='blue', linewidth=2)
    plt.plot(impostor_xnew, impostor_ynew, label='Impostor Distribution', color='red', linewidth=2)
    
    # 设置图表属性
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # 设置标题
    if title is None:
        method_name = self.config.method_config.method_name
        dataset_name = self.config.dataset_config.dataset_name
        title = f'Score Distributions: {method_name} on {dataset_name}'
    
    if experiment_num is not None:
        title += f' (Experiment {experiment_num})'
    
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_prefix = f"exp{experiment_num}_" if experiment_num is not None else ""
    filename = f"{exp_prefix}dist_{self.config.method_config.method_name}_{self.config.dataset_config.dataset_name}_{timestamp}.png"
    save_path = plots_dir / filename
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 记录到日志
    self.logger.info(f"Score distributions plot saved to {save_path}")
    
    return save_path