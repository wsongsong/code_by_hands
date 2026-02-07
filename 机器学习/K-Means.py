import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 初始化质心,随机选择k个样本点作为初始质心
        indices = np.random.choice(n_samples, self.k, replace=False)
        centroids = X[indices]
        
        for _ in range(self.max_iters):
            # 分配标签,计算每个样本到质心的距离并分配到最近的质心
            # linalg.norm计算欧氏距离,axis=2表示对最后一个维度计算,newaxis用于扩展维度以便广播
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # 计算新质心
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                if np.sum(labels == i) > 0:
                    new_centroids[i] = np.mean(X[labels == i], axis=0)
            
            # 检查收敛
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break
                
            centroids = new_centroids
            
        self.centroids = centroids
        return labels
    
    def predict(self, X):
        if self.centroids is None:
            raise ValueError("请先调用fit方法训练模型")
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
# 生成示例数据
np.random.seed(42)
X1 = np.random.randn(100, 2) + np.array([5, 5])
X2 = np.random.randn(100, 2) + np.array([-5, -5])
X3 = np.random.randn(100, 2) + np.array([5, -5])
X = np.vstack((X1, X2, X3))


# 训练KMeans
kmeans = KMeans(k=3, max_iters=100)
labels = kmeans.fit(X)

print(f"质心形状: {kmeans.centroids.shape}")
print(f"标签形状: {labels.shape}")
print(f"唯一标签: {np.unique(labels)}")