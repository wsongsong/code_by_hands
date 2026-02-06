import numpy as np

np.random.seed(42)
X = 2 * np.random.rand(100, 2)
Y = X.dot(np.array([3,100]).reshape(-1,1)) + np.random.randn(100, 1)

def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - Y))
    return cost

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

theta = np.random.randn(2, 1)
alpha = 0.01
num_iterations = 1000
theta_final, cost_history = gradient_descent(X, Y, theta, alpha, num_iterations)
print("Final parameters (theta):")
print(theta_final)
print("Final cost:")
print(cost_history[-1])
print(np.array([1,2]).dot(theta_final))
