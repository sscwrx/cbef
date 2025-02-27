from typing import Tuple,List
import numpy as np

def caculateVerificationRate(thr, genuine, impostor):
    #计算错误接受率FAR:False Accept Rate
    FA  = 0
    for i in range(impostor.__len__()):
        if impostor[i] >= thr:
            FA = FA + 1
    FAR = (FA / impostor.__len__())*100
    #计算错误拒绝率FRR:False Reject Rate和真实接受率Genuine Accept Rate
    FR = 0
    GA = 0
    for j in range(genuine.__len__()):
        if genuine[j] < thr:
            FR = FR + 1
        elif genuine[j] > thr:
            GA = GA + 1
    FRR = (FR / genuine.__len__())*100
    GAR = (GA / genuine.__len__())*100
    #计算等错误率
    TER = (FA + FR)/(genuine.__len__() + impostor.__len__())
    #计算验证率Verification Rate
    TSR = (1 - TER)*100
    #print(TSR, FAR, FRR, GAR)
    return TSR, FAR, FRR, GAR

def computePerformance(genuine, impostor, step,verbose=False)-> Tuple[float,float,List[int],List[int]]:
    #查找迭代的起始值和终止值，以找到最优阈值。
    start = np.min(genuine)
    if np.min(impostor) < start:
        start = np.min(impostor)
    stop = max(impostor)
    if np.max(genuine) > stop:
        stop = np.max(genuine)
    #计算每个阈值的TSR、FAR、FRR和GAR
    mTSR = [0]
    mFAR = [0]
    mFRR = [0]
    mGAR = [0]
    y = []
    x = []
    z = []
    s = []
    mTSR[0], mFAR[0], mFRR[0], mGAR[0] = caculateVerificationRate(start, genuine, impostor)
    thr_list = np.arange(start + step, stop, step)
    #print(thr_list)
    for i in range(thr_list.__len__()):
        TSR, FAR, FRR, GAR = caculateVerificationRate(thr_list[i], genuine, impostor)
        mTSR.append(TSR)
        mFAR.append(FAR)
        mFRR.append(FRR)
        mGAR.append(GAR)
        y.append(FRR)
        x.append(FAR)
        z.append(GAR)
        s.append(TSR)
        if verbose:
            print('阈值thr：%.6f, FAR：%.6f, FRR：%.6f, TSR：%.6f' % (thr_list[i], FAR, FRR, TSR))

    #找出FAR和FRR最接近的最优阈值
    optimal_thr_index = np.argmin(np.abs(np.array(y) - np.array(x)))
    optimal_thr = thr_list[optimal_thr_index]
    EER = (y[optimal_thr_index] + x[optimal_thr_index]) / 2
    if verbose:
        print('验证率Verification Rate：%.6f' % (s[optimal_thr_index]))
        print('真实接受率Genuine Accept Rate：%.6f' % (z[optimal_thr_index]))
        print('错误接受率False Accept Rate：%.6f' % (x[optimal_thr_index]))
        print('错误拒绝率False Reject Rate：%.6f' % (y[optimal_thr_index]))
        print('等错误率Equal Error Rate：%.6f' % (EER))


    return  EER, optimal_thr,mFAR,mGAR


