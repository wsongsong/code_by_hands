import numpy as np

def compute_f1(y_true, y_pred):
    """
    计算F1分数的函数
    参数:
    y_true -- 真实标签的numpy数组
    y_pred -- 预测标签的numpy数组
    返回:
    f1_score -- 计算得到的F1分数
    """
    # 计算真阳性、假阳性和假阴性
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # 计算精确率和召回率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
f1 = compute_f1(y_true, y_pred)
print(f"F1 Score: {f1}")