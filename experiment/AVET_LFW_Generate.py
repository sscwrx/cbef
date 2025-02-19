import numpy as np
import scipy as sp
import time
import os
from tqdm import tqdm
from method.AVET import absolute_value_equations_transform,bi_avet, in_avet,biohash
from pathlib import Path
from metrics.performance.performance_metrics import perform_matching
def generate_protected_templates(data_type="face",dataset="LFW",seed=1,method="avet"):
    # 初始化保护系统（特征维度为512，与原始代码相同）
    assert data_type in ["face", "fingerprint"], "data_type must be 'face' or 'fingerprint'"
    assert method in ["baseline","bio_hash","avet", "bi_avet", "in_avet"], "method must be 'avet' or 'bi_avet' or 'in_avet'"

    if data_type == "face":
        # 初始化计时
        start_time = time.time()
        # 生成受保护的模板
        for i in tqdm(range(127)):
            for j in range(12):
                try:
                    # 加载特征向量
                    path = f"./embeddings/{dataset}/{i+1}_{j+1}.mat" 
                    # 为FEI修订
                    if dataset == "FEI":
                        path = f"./embeddings/{dataset}/{i+1}_{j}.mat"
                    if not os.path.exists(path):
                        print(f"Warning: File not found: {path}")
                        continue
                        
                    mat_file = sp.io.loadmat(path)
                    face_vector = mat_file['preOut']  # 与原始代码保持一致
                    face_vector = np.squeeze(face_vector)

                    # 生成受保护模板
                    if method == "avet":
                        protected_template,_,_,_ = absolute_value_equations_transform(face_vector,seed)
                    elif method == "bi_avet":
                        protected_template = bi_avet(face_vector,seed)
                    elif method == "in_avet":
                        protected_template = in_avet(face_vector,k=300,g=16,seed=seed)
                    elif method == "bio_hash":
                        protected_template = biohash(face_vector, bh_len=40, user_seed=seed)     
                    elif method == "baseline":
                        protected_template = face_vector               
                    # 保存结果
                    output_path = f"./protectedTemplates/{dataset}/{i+1}_{j+1}"
                    np.savez(output_path, 
                            protected_template=protected_template,
                            seed=seed)
                    
                except Exception as e:
                    print(f"Error processing {i+1}_{j+1}: {str(e)}")
        
        # 计算平均时间
        end_time = time.time()
        mean_time = (end_time - start_time) / 1524  # 总共127*12=1524个模板
        print('生成1524个受保护模板的平均时间是：', mean_time)
        return mean_time
    elif data_type == "fingerprint":

        # 初始化计时
        start_time = time.time()

        
        # 生成受保护的模板
        for i in tqdm(range(0,100)):
            for j in range(0,5):
                # 加载特征向量
                path = f"./embeddings/{dataset}/{i+1}_{j+4}.mat" # dataset value is like "FVC2002/Db1_a" 
                assert os.path.exists(path), f"File not found: {path}"
                    
                mat_file = sp.io.loadmat(path)
                fingerprint_vector = mat_file['Ftemplate']  # 与原始代码保持一致


                # 归一化
                fingerprint_vector = np.squeeze(fingerprint_vector)
                # fingerprint_vector = fingerprint_vector / np.linalg.norm(fingerprint_vector)

                # 生成受保护模板
                if method == "avet":
                    protected_template,_,_,_ = absolute_value_equations_transform(fingerprint_vector,seed)
                elif method == "bi_avet":
                    protected_template = bi_avet(fingerprint_vector,seed)
                elif method == "in_avet":
                    protected_template = in_avet(fingerprint_vector,k=300,g=16,seed=seed)
                elif method == "bio_hash":
                    protected_template = biohash(fingerprint_vector, bh_len=40, user_seed=seed)
                elif method == "baseline":
                    protected_template = fingerprint_vector  
                # 保存结果
                output_path = f"./protectedTemplates/{dataset}/{i+1}_{j+4}"
                np.savez(output_path, 
                        protected_template=protected_template,
                        seed=seed)

        # 计算平均时间
        end_time = time.time()
        mean_time = (end_time - start_time) / 500 
        print('生成500个受保护模板的平均时间是：', mean_time)
        return mean_time

if __name__ == '__main__':
    generate_protected_templates(data_type="fingerprint",dataset="FVC2002/Db1_a",seed=1,method="baseline")
