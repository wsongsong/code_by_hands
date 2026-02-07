import numpy as np

y_true =  [0,   0,   1,   1,   0,   1,   0,   1,   1,   1]
y_score = [0.1, 0.4, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9]

def auc(y_true, y_score):
    # 将y_true和y_score按照y_score从大到小排序
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    # 计算真正例和假正例的数量
    P = np.sum(y_true)
    N = len(y_true) - P
    
    # 计算TPR和FPR
    TPR = np.cumsum(y_true_sorted) / P
    FPR = np.cumsum(1 - y_true_sorted) / N
    
    # 计算AUC
    auc_value = np.trapz(TPR, FPR)
    
    return auc_value

def auc2(y_true, y_score):
    ranks = enumerate(sorted(zip(y_true, y_score), key=lambda x:x[-1]), start=1)
    pos_ranks = [x[0] for x in ranks if x[1][0]==1]
    M = sum(y_true)
    N = len(y_true)-M
    auc = (sum(pos_ranks)-M*(M+1)/2)/(M*N)
    return auc


print("AUC:", auc(y_true, y_score), auc2(y_true, y_score))