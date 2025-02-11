import itertools
import numpy as np
import time
import CalculateVerificationRate
from AVET_imp import absolute_value_equations_transform
import scipy as sp
def calculate_template_similarity(template1, template2,measure="cosine"):
    """计算两个受保护模板之间的相似度"""
    if len(template1) != len(template2):
        raise ValueError(f"模板长度不匹配, {len(template1)} != {len(template2)}")

    if measure == "cosine":
        # cosine similarity 
        similarity = np.dot(template1, template2) / (np.linalg.norm(template1) * np.linalg.norm(template2))
    elif measure == "euclidean":
        # eculidean
        similarity =  - sp.spatial.distance.euclidean(template1, template2)
    elif measure == "hamming":
        # hamming similarity 
        similarity = np.sum(template1 == template2) / len(template1) 
    elif measure == "jaccard":
        # jaccard similarity in_avet
        # assert isinstance(template1[0],np.int32), f"template element must be np.int32 type, got {type(template1[0])}"
        # similarity = np.sum(np.bitwise_and(template1,template2)) / np.sum(np.bitwise_or(template1,template2))
        
        # distance = sp.spatial.distance.jaccard(template1, template2)
        # similarity = 1 - distance
        match = np.abs(template1 - template2)
        total_zero_num = np.count_nonzero(match == 0)
        similarity = total_zero_num / (template1.__len__() + template2.__len__() - total_zero_num)
        
    return similarity

def perform_matching(data_type='face',dataset="LFW",template_path="./protectedTemplates/LFW/",verbose=False,measure="cosine"):
    if data_type == "face":

        """执行模板匹配过程"""
        # 生成真匹配和假匹配的组合
        genuine_combinations = list(itertools.combinations(range(1, 13), 2))
        impostor_combinations = list(itertools.combinations(range(1, 128), 2))
        
        genuine_similarity_list = []
        impostor_similarity_list = []
        

        
        # 执行真匹配（同一用户的不同样本）
        start_time1 = time.time()
        for i in range(127):
            for comb in genuine_combinations:
                # 加载第一个模板
                data1 = np.load(f"{template_path}/{i+1}_{comb[0]}.npz")
                template1 = data1['protected_template']
                
                # 加载第二个模板
                data2 = np.load(f"{template_path}/{i+1}_{comb[1]}.npz")
                template2 = data2['protected_template']
                # 计算相似度
                similarity = calculate_template_similarity(template1, template2,measure=measure)
                genuine_similarity_list.append(similarity)
                
        end_time1 = time.time()
        mean_time_genuine = (end_time1 - start_time1) / 8382
        print("8382次真匹配的平均时间：", mean_time_genuine)
        
        # 执行假匹配（不同用户之间的匹配）
        start_time2 = time.time()
        for comb in impostor_combinations:
            # 加载第一个模板
            data1 = np.load(f"{template_path}/{comb[0]}_1.npz")
            template1 = data1['protected_template']
            
            # 加载第二个模板
            data2 = np.load(f"{template_path}/{comb[1]}_1.npz")
            template2 = data2['protected_template']
            
            # 计算相似度
            similarity = calculate_template_similarity(template1, template2,measure=measure)
            impostor_similarity_list.append(similarity)
            
        end_time2 = time.time()
        mean_time_impostor = (end_time2 - start_time2) / 8001
        print("8001次假匹配的平均时间：", mean_time_impostor)
        
        # 计算EER和阈值

        EER, thr = CalculateVerificationRate.computePerformance(
            genuine_similarity_list, 
            impostor_similarity_list, 
            0.001,
            verbose=verbose
        )
        
        return EER, thr
    elif data_type == "fingerprint":

        genuine_combinations = itertools.combinations(list(range(4, 9)), 2)#真匹配，每个用户的类内匹配，C52 * 100= 1000
        impostor_combinations = itertools.combinations(list(range(1, 101)), 2)  # 假匹配，每个用户的第一个i_4.npy模板与其它所有用户的第一个模板j_4.npy做假匹配，C100 2 = 4950
        genuine_similarity_list = []
        impostor_similarity_list = []

        # 执行真匹配（同一用户的不同样本）
        start_time1 = time.time()
        for i in range(100):
            for comb in genuine_combinations:
                data1 = np.load(f"{template_path}/{i+1}_{comb[0]}.npz")
                template1 = data1['protected_template']

                data2 = np.load(f"{template_path}/{i+1}_{comb[1]}.npz")
                template2 = data2['protected_template'] 

                similarity = calculate_template_similarity(template1, template2,measure=measure)
                genuine_similarity_list.append(similarity)
        start_time2 = time.time()
        mean_time_genuine = (start_time2 - start_time1)/1000
        print("1000次真匹配的平均时间：", mean_time_genuine)

        start_time1 = time.time()
        # 执行假匹配
        for comb in impostor_combinations:
            # 加载第一个模板
            data1 = np.load(f"{template_path}/{comb[0]}_4.npz")
            template1 = data1['protected_template']
            
            # 加载第二个模板
            data2 = np.load(f"{template_path}/{comb[1]}_4.npz")
            template2 = data2['protected_template']

            # 计算相似度
            similarity = calculate_template_similarity(template1, template2,measure=measure)
            impostor_similarity_list.append(similarity)

        end_time2 = time.time()
        mean_time_impostor = (end_time2 - start_time2) / 4950
        print("4950次假匹配的平均时间：", mean_time_impostor) 
                
        # 计算EER和阈值

        EER, thr = CalculateVerificationRate.computePerformance(
            genuine_similarity_list, 
            impostor_similarity_list, 
            0.001,
            verbose=verbose
        )
        return EER, thr
if __name__ == '__main__':
    times = 5
    seed = [1,2,3,4,5]
    data_type = ["fingerprint","face"]
    datasets = {
        "fingerprint":["FVC2002/Db1_a","FVC2002/Db2_a","FVC2002/Db3_a",
                       "FVC2004/Db1_a","FVC2004/Db2_a","FVC2004/Db3_a"],
        "face":["FEI","LFW", "ColorFeret", "CASIA-WebFace"]
    }

    measure = "jaccard"
    eer_list = []
    optimal_thr_list = []

    # print(test_template)   
    for data_type in data_type:
        for dataset in datasets[data_type]:
            eer_list.append(EER)
            optimal_thr_list.append(thr)
            for i in range(times):
                print(f"({i+1}/5) Matching evaluation for {dataset}... ")
                EER, thr = perform_matching(data_type=data_type,
                                            dataset=dataset,
                                            template_path=f"./protectedTemplates/{dataset}/",
                                            verbose=True,
                                            measure=measure)
            eer_list.append(EER)
            optimal_thr_list.append(thr)
    print("#" * 100)
    print(f"\nFinal results for {dataset} (min):")
    print(f"Mean EER: {np.mean(eer_list)}")
    print(f"Mean Optimal Threshold: {np.mean(optimal_thr_list)}")
    print(f"Standard Deviation of EER: {np.std(eer_list)}")
    print(f"Standard Deviation of Optimal Threshold: {np.std(optimal_thr_list)}")
    print(f"Max EER: {np.max(eer_list)}")
    print(f"Min EER: {np.min(eer_list)}")
    print(f"Max Optimal Threshold: {np.max(optimal_thr_list)}")
    print(f"Min Optimal Threshold: {np.min(optimal_thr_list)}")
    print("#" * 100)