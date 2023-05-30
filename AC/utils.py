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
from scipy.spatial import distance
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import stats


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
        labels[::1 + self.negative_samples] = 1
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


# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.4, min_sigma=0.4, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.tensor([action + ou_state]).float()


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data_pointer = 0
        self.n_entries = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype = object)

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])


class ReplayTree:
    def __init__(self, capacity):
        self.capacity = capacity # the capacity for memory replay
        self.tree = SumTree(capacity)
        self.abs_err_upper = 1.

        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 0.01
        self.abs_err_upper = 1.

    def __len__(self):
        return self.tree.total()

    def push(self, error, sample):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, batch_size):
        pri_segment = self.tree.total() / batch_size

        priorities = []
        batch = []
        idxs = []

        is_weights = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            prob = p / self.tree.total()

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return zip(*batch), idxs, is_weights

    def batch_update(self, tree_idx, abs_errors):
        '''Update the importance sampling weight
        '''
        abs_errors += self.epsilon

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class Prioritized_Buffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
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

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        user = np.concatenate(batch[0])
        memory = np.concatenate(batch[1])
        action = batch[2]
        reward = batch[3]
        next_user = np.concatenate(batch[4])
        next_memory = np.concatenate(batch[5])
        done = batch[6]

        return user, memory, action, reward, next_user, next_memory, done

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def get_beta(idx, beta_start=0.4, beta_steps=100000):
    return min(1.0, beta_start + idx * (1.0 - beta_start) / beta_steps)


def preprocess_data(data_dir, train_rating):
    data = pd.read_csv(os.path.join(data_dir, train_rating), usecols=[0, 1, 2])

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


def pairwise_distances_fig(embs):
    embs = embs.detach().cpu().numpy()
    similarity_matrix_cos = distance.cdist(embs, embs, "cosine")
    similarity_matrix_euc = distance.cdist(embs, embs, "euclidean")

    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(121)
    cax = ax.matshow(similarity_matrix_cos)
    fig.colorbar(cax)
    ax.set_title("Cosine")
    ax.axis("off")

    ax = fig.add_subplot(122)
    cax = ax.matshow(similarity_matrix_euc)
    fig.colorbar(cax)
    ax.set_title("Euclidian")
    ax.axis("off")

    fig.suptitle("Action pairwise distances")
    plt.close()
    return fig


def pairwise_distances(embs):
    fig = pairwise_distances_fig(embs)
    fig.show()


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def smooth_gauss(arr, var):
    return ndimage.gaussian_filter1d(arr, var)


class Plotter:
    def __init__(self, loss, style):
        self.loss = loss
        self.style = style
        self.smoothing = lambda x: smooth_gauss(x, 4)

    def set_smoothing_func(self, f):
        self.smoothing = f

    def plot_loss(self):
        row_num = 0
        for row in self.style:
            row_num += 1
            train_or_test = "train" if row_num == 1 else "test"
            fig, axes = plt.subplots(1, len(row), figsize=(16, 6))
            if len(row) == 1:
                axes = [axes]
            for col in range(len(row)):
                key = row[col]
                axes[col].set_title(key)
                if train_or_test == "train":
                    axes[col].plot(
                        self.loss[train_or_test]["step"],
                        self.smoothing(self.loss[train_or_test][key]),
                        "b-",
                        label="train",
                    )
                else:
                    axes[col].plot(
                        self.loss[train_or_test]["step"],
                        self.loss[train_or_test][key],
                        "r-.",
                        label="test",
                    )
            plt.legend()
            plt.show()

    def log_loss(self, key, item, test=False):
        kind = "train"
        if test:
            kind = "test"
        self.loss[kind][key].append(item)

    def log_losses(self, losses, test=False):
        for key, val in losses.items():
            self.log_loss(key, val, test)

    @staticmethod
    def kde_reconstruction_error(
            ad, gen_actions, true_actions, device=torch.device("cpu")
    ):
        def rec_score(actions):
            return (
                ad.rec_error(torch.tensor(actions).to(device).float())
                .detach()
                .cpu()
                .numpy()
            )

        true_scores = rec_score(true_actions)
        gen_scores = rec_score(gen_actions)

        true_kernel = stats.gaussian_kde(true_scores)
        gen_kernel = stats.gaussian_kde(gen_scores)

        x = np.linspace(0, 1000, 100)
        probs_true = true_kernel(x)
        probs_gen = gen_kernel(x)
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        ax.plot(x, probs_true, "-b", label="true dist")
        ax.plot(x, probs_gen, "-r", label="generated dist")
        ax.legend()
        return fig

    @staticmethod
    def plot_kde_reconstruction_error(*args, **kwargs):
        fig = Plotter.kde_reconstruction_error(*args, **kwargs)
        fig.show()
