import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # 计算测试点与所有训练点的距离,并找到最近的k个点,然后统计这k个点的标签，选择出现次数最多的标签作为预测结果
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)
# Example usage
if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    X_test = np.array([[1.5, 2.5], [7.5, 6.5]])
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)