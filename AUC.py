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

    print("TPR:", TPR)
    print("FPR:", FPR)  
    
    # 计算AUC
    auc_value = np.trapz(TPR, FPR)
    
    return auc_value

print("AUC:", auc(y_true, y_score))