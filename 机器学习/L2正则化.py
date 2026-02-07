import numpy as np
# 设定随机种子，使得每次生成结果一致
np.random.seed(42)
X = np.random.rand(100, 3)
Y = X.dot(np.array([10, 20, 30])) + 40 + np.random.randn(100) * 5
w = np.zeros(3)
b = 0.0
lr = 0.01
epochs = 10000
lambd = 0.1
for _ in range(epochs):
    y_pred = X @ w + b
    error = y_pred - Y
    dw = (X.T @ error) / len(Y) + lambd * w
    db = np.sum(error) / len(Y)
    w -= lr * dw
    b -= lr * db
    loss = (1/(2*len(Y))) * np.sum(error ** 2) + (lambd / 2) * np.sum(w ** 2)
X_test = np.array([[1, 2, 3]])
Y_test_pred = X_test @ w + b
print("Predicted value for X_test:", Y_test_pred)
print("Learned parameters: w =", w, ", b =", b)