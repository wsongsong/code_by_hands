import numpy as np

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义逻辑回归类
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
# 创建示例数据
np.random.seed(42)
X = np.random.randn(100, 2)  # 100个样本，2个特征
# 创建标签：如果x1 + x2 > 0则为1，否则为0
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 创建模型
model = LogisticRegression(learning_rate=0.1, num_iterations=10000)

# 训练模型
model.fit(X, y)

# 预测
predictions = model.predict(X)

# 计算准确率
accuracy = np.mean(predictions == y)
print(f"准确率: {accuracy:.2%}")

# 查看学到的参数
print(f"权重: {model.weights}")
print(f"偏置: {model.bias}")