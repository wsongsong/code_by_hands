import numpy as np

np.random.seed(42)
# linspace生成100个点，从0到10
X = np.linspace(0, 10, 100)

Y = 3 * X + 4 + np.random.randn(100) * 5

w = 0.0
b = 0.0
learning_rate = 0.01
num_iterations = 10000
m = len(X)
for _ in range(num_iterations):
    Y_pred = w * X + b
    errors = Y_pred - Y
    dw = (1/m) * np.dot(errors, X)
    db = (1/m) * np.sum(errors)
    w -= learning_rate * dw
    b -= learning_rate * db
    loss = (1/(2*m)) * np.sum(errors ** 2)

# quick test
X_test = np.array([1, 2, 3])
Y_test_pred = w * X_test + b
print("Predicted values for X_test:", Y_test_pred)
print("Learned parameters: w =", w, ", b =", b)