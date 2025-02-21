import time
import numpy as np
import scipy as sp
import SWG_LFW_Matching
N = 512#沃尔什矩阵和人脸特征向量的维度
walshmatrix = sp.io.loadmat('walshmtx512.mat')
walsh = walshmatrix['WalshMtx']
def randperm(max, select):
    a = np.arange(max)
    np.random.shuffle(a)#np.random.shuffle()第一次，设置完之后每次生成的permatrix都是一样的。
    return a[:select]
def SWG(w, n, r, Db, HashCode_path):
    part_walsh = np.zeros((r, N), dtype=np.int8)  # 生成一个零矩阵part_walsh来存储1个维度为r * N的部分walsh矩阵
    n_part_walsh_vector = np.zeros(n * r)
    permutation_n_part_walsh_vector = np.empty(n * r, dtype=int)
    perm = randperm(N, r)  # 生成一个随机索引序列，取沃尔什矩阵的对应索引行。
    # print('随机索引序列', perm)
    for j in range(r):
        part_walsh[j, :] = walsh[perm[j], :]  # 根据生成的随机序列索引perm取出部分沃尔什矩阵part_walsh
    permutation_seed = np.random.permutation(n * r)  # np.random.permutation(n)函数产生n范围内的随机整数，且里面的数是0到n-test1。
    list = [0] * (n * r)
    for i in range(127):
        for j in range(12):
            file_path = HashCode_path + str(i + 1) + '_' + str(j + 1) + '.bin'
            with open(file_path, 'wb') as file:
                file.write(b'')
    start_time = time.time()
    for i in range(127):
        for j in range(12):
            mat_file_path = '../data/' + Db + '/' + str(i + 1) + '_' + str(j + 1)
            mat_file = sp.io.loadmat(mat_file_path)
            face_vector = mat_file['preOut'][0]
            walsh_vector = np.dot(part_walsh, face_vector)
            #print(walsh_vector)
            for k in range(n):
                n_part_walsh_vector[k*r:(k+1)*r] = walsh_vector
            permutation_n_part_walsh_vector = n_part_walsh_vector[permutation_seed]
            #print(permutation_n_part_walsh_vector)
            for t in range(n*r):
                if t < n*r-w:
                    if permutation_n_part_walsh_vector[t] < permutation_n_part_walsh_vector[t+w]:
                        list[t] = 0
                    else:
                        list[t] = 1
                if t >= n*r-w:
                    if permutation_n_part_walsh_vector[t] < permutation_n_part_walsh_vector[(t+w) % n*r]:
                        list[t] = 0
                    else:
                        list[t] = 1
            Hashcode = ''.join([str(num) for num in list])
            binary_HashCode = bytes(int(Hashcode[i:i + 8], 2) for i in range(0, n * r, 8))
            with open(HashCode_path + str(i + 1) + '_' + str(j + 1) + '.bin', 'wb') as file:
                file.write(binary_HashCode)
    end_time = time.time()
    mean_time = (end_time - start_time) / 1524
    print('生成1524个OPH哈希码的平均时间是：', mean_time)
    return mean_time
if __name__ == '__main__':
    w=2
    n=4
    r=256
    Db = 'LFW'
    #HashCode_path = 'HashCode/SWG/LFW/n=4_r=256_w=2_5/'
    HashCode_path = 'HashCode/SWG/LFW/'
    #SWG(w, n, r, Db, HashCode_path)
    SWG_LFW_Matching.SWG_Matching(HashCode_path)
