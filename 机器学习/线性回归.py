import numpy as np
# 设定随机种子，使得每次生成结果一致
np.random.seed(42)

# 设定X为100行2列的随机数组，每行代表一个样本，每列代表一个特征
X = np.random.rand(100, 2)

# 设定Y为X的线性组合（10*x1 + 20*x2）加上一些随机噪声，模拟真实数据
Y = X.dot(np.array([10,20]).reshape(-1,1)) + np.random.randn(100, 1)

# 增加截距项（bias），在X的第一列添加一列全1
X = np.hstack((np.ones((X.shape[0], 1)), X))

# 定义计算成本函数
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - Y))
    return cost

# 定义梯度下降函数
def gradient_descent(X, Y, theta, learning_rate, iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - Y
        gradients = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, Y, theta)

    return theta, cost_history

theta = np.random.randn(3, 1)
alpha = 0.01
num_iterations = 10000
theta_final, cost_history = gradient_descent(X, Y, theta, alpha, num_iterations)
print("Final parameters (theta):")
print(theta_final)
print("Final cost:")
print(cost_history[-1])
print(np.array([1,2,3]).dot(theta_final))
