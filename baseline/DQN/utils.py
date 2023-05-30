from collections import defaultdict
import os
import random
import time
import tqdm
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as td


class EvalDataset(td.Dataset):
    def __init__(self, positive_data, item_num, positive_mat, negative_samples=99):
        super(EvalDataset, self).__init__()
        self.positive_data = np.array(positive_data)
        self.item_num = item_num
        self.positive_mat = positive_mat
        self.negative_samples = negative_samples
        
        self.reset()
        
    def reset(self):
        print("Resetting dataset")
        data = self.create_valid_data()
        labels = np.zeros(len(self.positive_data) * (1 + self.negative_samples))
        labels[::1+self.negative_samples] = 1
        self.data = np.concatenate([
            np.array(data), 
            np.array(labels)[:, np.newaxis]], 
            axis=1
        )

    def create_valid_data(self):
        valid_data = []
        for user, positive in self.positive_data:
            valid_data.append([user, positive])
            for i in range(self.negative_samples):
                negative = np.random.randint(self.item_num)
                while (user, negative) in self.positive_mat:
                    negative = np.random.randint(self.item_num)
                    
                valid_data.append([user, negative])
        return valid_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, label = self.data[idx]
        output = {
            "user": user,
            "item": item,
            "label": np.float32(label),
        }
        return output


#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.4, min_sigma=0.4, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.tensor([action + ou_state]).float()


class Prioritized_Buffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, user, memory, action, reward, next_user, next_memory, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((user, memory, action, reward, next_user, next_memory, done))
        else:
            self.buffer[self.pos] = (user, memory, action, reward, next_user, next_memory, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        batch       = list(zip(*samples))
        user        = np.concatenate(batch[0])
        memory      = np.concatenate(batch[1])
        action      = batch[2]
        reward      = batch[3]
        next_user   = np.concatenate(batch[4])
        next_memory = np.concatenate(batch[5])
        done        = batch[6]

        return user, memory, action, reward, next_user, next_memory, done

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def get_beta(idx, beta_start=0.4, beta_steps=100000):
    return min(1.0, beta_start + idx * (1.0 - beta_start) / beta_steps)

def preprocess_data(data_dir, train_rating):
    # 随机抽取2000个items
    # data = pd.read_csv(os.path.join(data_dir, train_rating),usecols=[0, 1, 2])
    data = pd.read_csv("/Users/xuxiaoan/Downloads/ml-100k/u.data", usecols=[0, 1, 2], engine='python', header=None, sep='\t')
    data.columns = ['userId','movieId','rating']
    random.seed(1)
    items = random.sample(data["movieId"].unique().tolist(), 800)
    data = data[data["movieId"].isin(items)]

    # 将movieId映射
    unique_movieids = data['movieId'].unique()
    mapping = {}
    count = 0
    for movieid in unique_movieids:
        if movieid not in mapping:
            mapping[movieid] = count
            count += 1
    reverse_mapping = {i[1]: i[0] for i in mapping.items()}
    data["movieId"] = [mapping[i] for i in data["movieId"]]

    # userId映射
    data["userId"] = data["userId"] - 1

    # 获取user和item个数
    user_num = data['userId'].max() + 1
    item_num = data['movieId'].max() + 1

    # 对于抽取后的data划分
    # train_data = data.sample(frac=0.8, random_state=16)
    # test_data = data.drop(train_data.index).values.tolist()
    # train_data = train_data.values.tolist()

    # 稀疏矩阵存储训练集和测试集
    train_data = data.sample(frac=0.6, random_state=16)
    valid_test_data = data.drop(train_data.index)
    valid_data = valid_test_data.sample(frac=0.5, random_state=16)
    test_data = valid_test_data.drop(valid_data.index)
    train_data = train_data.values.tolist()
    valid_data = valid_data.values.tolist()
    test_data = test_data.values.tolist()

    # 稀疏矩阵存储训练集和测试集
    train_mat = defaultdict(float)
    valid_mat = defaultdict(float)
    test_mat = defaultdict(float)
    for user, item, rating in train_data:
        train_mat[user, item] = rating
    for user, item, rating in valid_data:
        valid_mat[user, item] = rating
    for user, item, rating in test_data:
        test_mat[user, item] = rating
    train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(train_matrix, train_mat)
    valid_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(valid_matrix, valid_mat)
    test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(test_matrix, test_mat)

    # 评分记录不少于20的user作为appropriate_users
    appropriate_users = np.arange(user_num)[(train_matrix.tocsr().getnnz(1) >= 20)]
    
    return (train_data, train_matrix, valid_data, valid_matrix, test_data, test_matrix,
            user_num, item_num, appropriate_users, mapping, reverse_mapping)

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def hit_metric(recommended, actual):
    return int(actual in recommended)

def dcg_metric(recommended, actual):
    if actual in recommended:
        index = recommended.index(actual)
        return np.reciprocal(np.log2(index + 2))
    return 0




