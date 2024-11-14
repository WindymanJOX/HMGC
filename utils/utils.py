import torch
import numpy as np
from sklearn.cluster import KMeans

def one_hot_encode(num_classes, class_idx):
        return torch.eye(num_classes)[class_idx.to(torch.device('cpu'))]

def label2correlaiton(label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        correlaiton = torch.eq(label_i, label_j).float()

        return correlaiton

def g_ancor(X, mask):
        box_size = 32
        if X.shape[1] <= box_size**2:
                return X, mask
        h = int(np.sqrt(X.shape[1]))
        idx_ancor = torch.tensor(list(range(0, int(X.shape[1]), (h//box_size)**2)))
        X_ancor = X[:, idx_ancor, :]
        _mask = mask[:, idx_ancor]
        return X_ancor, _mask

def extract_representative_samples(feature_map, mask, num_samples_per_class=100):
    # 初始化变量
    _, num_classes, h, w = mask.shape
    _, c, _, _ = feature_map.shape
    class_centers = []

    for cls in range(num_classes):
        # 找到属于当前类别的所有像素位置
        positions = np.where(mask[0, cls, :, :] == 1)
        if len(positions[0]) == 0:
            # 如果没有找到任何属于当前类别的像素，跳过
            continue

        # 提取属于当前类别的所有像素向量
        samples = feature_map[0, :, positions[0], positions[1]].T

        # 如果样本数量少于所需数量，直接使用所有样本
        if samples.shape[0] <= num_samples_per_class:
            class_center = np.mean(samples, axis=0)
            class_centers.append(class_center)
            continue

        # 使用K-means聚类方法提取代表性样本
        kmeans = KMeans(n_clusters=num_samples_per_class, random_state=0)
        kmeans.fit(samples)
        representative_samples = kmeans.cluster_centers_
        
        # 计算类特征中心
        class_center = np.mean(representative_samples, axis=0)
        class_centers.append(class_center)

    return np.array(class_centers)