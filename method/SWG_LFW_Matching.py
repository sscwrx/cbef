import itertools
import numpy as np
import time
import SWG.SWG_CalculateVerificationRate
genuine_combinations = itertools.combinations(list(range(1, 13)), 2)
genuine_combinations_list = []
for c in genuine_combinations:
    genuine_combinations_list.append(c)

impostor_combinations = itertools.combinations(list(range(1, 128)), 2)
impostor_combinations_list = []
for c in impostor_combinations:
    impostor_combinations_list.append(c)

def read_binary_file_as_binary_string(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
        # 将每个字节转换为8位二进制字符串
        return ''.join(format(byte, '08b') for byte in binary_data)
def SWG_Matching(HashCode_path):
    genuine_similarity_list = []
    impostor_similarity_list = []
    start_time1 = time.time()
    for i in range(127):
        for j in range(genuine_combinations_list.__len__()):
            file = genuine_combinations_list[j]
            bin_file_path1 = HashCode_path + str(i+1) + '_' + str(file[0]) + '.bin'
            HashCode1 = np.array(list(read_binary_file_as_binary_string(bin_file_path1)), dtype=np.uint8)
            bin_file_path2 = HashCode_path + str(i+1) + '_' + str(file[1]) + '.bin'
            HashCode2 = np.array(list(read_binary_file_as_binary_string(bin_file_path2)), dtype=np.uint8)
            match = np.abs(HashCode1 - HashCode2)
            total_zero_num = np.count_nonzero(match == 0)
            similiraty = total_zero_num / (HashCode1.__len__() + HashCode2.__len__() - total_zero_num)
            genuine_similarity_list.append(similiraty)
    end_time1 = time.time()
    mean_time_genuine = (end_time1 - start_time1)/8382
    #print(np.min(genuine_similarity_list))
    #print(np.max(genuine_similarity_list))
    print("8382次真匹配的平均时间：", mean_time_genuine)
    start_time2 = time.time()
    for k in range(impostor_combinations_list.__len__()):
        file = impostor_combinations_list[k]
        bin_file_path1 = HashCode_path + str(file[0]) + '_1.bin'
        HashCode1 = np.array(list(read_binary_file_as_binary_string(bin_file_path1)), dtype=np.uint8)
        bin_file_path2 = HashCode_path + str(file[1]) + '_1.bin'
        HashCode2 = np.array(list(read_binary_file_as_binary_string(bin_file_path2)), dtype=np.uint8)
        match = np.abs(HashCode1 - HashCode2)
        total_zero_num = np.count_nonzero(match == 0)
        similiraty = total_zero_num / (HashCode1.__len__() + HashCode2.__len__() - total_zero_num)
        impostor_similarity_list.append(similiraty)
    end_time2 = time.time()
    mean_time_impostor = (end_time2 - start_time2)/8001
    #print(np.min(impostor_similarity_list))
    #print(np.max(impostor_similarity_list))
    print("8001次假匹配的平均时间：", mean_time_impostor)
    EER, thr = SWG.SWG_CalculateVerificationRate.computePerformance(genuine_similarity_list, impostor_similarity_list, 0.001)
    #EER, z, x, y = SWG.SWG_CalculateVerificationRate.computePerformance(genuine_similarity_list, impostor_similarity_list, 0.005)
    # 计算genuine_similarity_list的均值和方差
    genuine_mean = np.mean(genuine_similarity_list)
    #print("genuine_similarity_list的均值:", genuine_mean)
    genuine_variance = np.var(genuine_similarity_list)
    #print("genuine_similarity_list的方差:", genuine_variance)
    # 计算impostor_similarity_list的均值和方差
    impostor_mean = np.mean(impostor_similarity_list)
    #print("impostor_similarity_list的均值:", impostor_mean)
    impostor_variance = np.var(impostor_similarity_list)
    #print("impostor_similarity_list的方差:", impostor_variance)
    DI = np.abs(genuine_mean - impostor_mean) / np.sqrt((genuine_variance+impostor_variance)/2)
    print("DI:", DI)
    #return EER, z, x, y
    return EER, thr, DI