from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

import pointops_cuda


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx

furthestsampling = FurthestSampling.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None: new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)

knnquery = KNNQuery.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, input, idx):
        """
        input: input: (n, c), idx : (m, nsample)
        output: (m, nsample, c)
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = torch.cuda.FloatTensor(m, nsample, c)
        pointops_cuda.grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (m, c, nsample)
        output: (n, c), None
        """
        n = ctx.n
        idx, = ctx.saved_tensors
        m, nsample, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        return grad_input, None

grouping = Grouping.apply


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    # 只用了坐标，新坐标和每个新坐标周围要sample的数把原来的点进行分组
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)
    else:
        return grouped_feat


class Subtraction(Function):
    @staticmethod
    def forward(ctx, input1, input2, idx):
        """
        input: input1: (n, c), input2: (n, c), idx: (n, nsample)
        output:  (n, nsample, c)
        """
        assert input1.is_contiguous() and input2.is_contiguous()
        n, c = input1.shape; nsample = idx.shape[-1]
        output = torch.cuda.FloatTensor(n, nsample, c).zero_()
        pointops_cuda.subtraction_forward_cuda(n, nsample, c, input1, input2, idx, output)
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, nsample, c)
        output: grad_input1: (n, c), grad_input2: (n, c)
        """
        idx, = ctx.saved_tensors
        n, nsample, c = grad_output.shape
        grad_input1 = torch.cuda.FloatTensor(n, c).zero_()
        grad_input2 = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.subtraction_backward_cuda(n, nsample, c, idx, grad_output, grad_input1, grad_input2)
        return grad_input1, grad_input2, None

subtraction = Subtraction.apply


class Aggregation(Function):
    @staticmethod
    def forward(ctx, input, position, weight, idx):
        """
        input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx: (n, nsample)
        output: (n, c)
        """
        assert input.is_contiguous() and position.is_contiguous() and weight.is_contiguous()
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.aggregation_forward_cuda(n, nsample, c, w_c, input, position, weight, idx, output)
        ctx.save_for_backward(input, position, weight, idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, c)
        output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight : (n, nsample, c')
        """
        input, position, weight, idx = ctx.saved_tensors
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grad_position = torch.cuda.FloatTensor(n, nsample, c).zero_()
        grad_weight = torch.cuda.FloatTensor(n, nsample, w_c).zero_()
        pointops_cuda.aggregation_backward_cuda(n, nsample, c, w_c, input, position, weight, idx, grad_output, grad_input, grad_position, grad_weight)
        return grad_input, grad_position, grad_weight, None

aggregation = Aggregation.apply


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, 3), (n, 3)
    dist_recip = 1.0 / (dist + 1e-8) # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


class Interpolation(Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset, new_offset, k=3):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, k), (n, k)
        dist_recip = 1.0 / (dist + 1e-8) # (n, k)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm # (n, k)

        n, c, m = new_xyz.shape[0], input.shape[1], input.shape[0]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.interpolation_forward_cuda(n, c, k, input, idx, weight, output)
        ctx.m, ctx.k = m, k
        ctx.save_for_backward(idx, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        m, k = ctx.m, ctx.k
        idx, weight = ctx.saved_tensors
        n, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(m, c).zero_()
        pointops_cuda.interpolation_backward_cuda(n, c, k, grad_output, idx, weight, grad_input)
        return None, None, grad_input, None, None, None

interpolation2 = Interpolation.apply



# class KMEANS:
#     def __init__(self, n_clusters=10, max_iter=None, verbose=True, device=torch.device("cpu")):

#         # self.n_cluster = n_clusters
#         self.n_clusters = n_clusters
#         self.labels = None
#         self.dists = None  # shape: [x.shape[0],n_cluster]
#         self.centers = None
#         self.variation = torch.Tensor([float("Inf")]).to(device)
#         self.verbose = verbose
#         self.started = False
#         self.representative_samples = None
#         self.max_iter = max_iter
#         self.count = 0
#         self.device = device

#     def fit(self, x):
#         # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
#         init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
#         # print(init_row.shape)    # shape 10
#         init_points = x[init_row]
#         # print(init_points.shape) # shape (10, 2048)
#         self.centers = init_points
#         while True:
#             # 聚类标记
#             self.nearest_center(x)
#             # 更新中心点
#             self.update_center(x)
#             if self.verbose:
#                 print(self.variation, torch.argmin(self.dists, (0)))
#             if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
#                 break
#             elif self.max_iter is not None and self.count == self.max_iter:
#                 break

#             self.count += 1

#         return self.representative_sample()

#     def nearest_center(self, x):
#         labels = torch.empty((x.shape[0],)).long().to(self.device)
#         # print(labels.shape)  # shape (250000)
#         dists = torch.empty((0, self.n_clusters)).to(self.device)
#         # print(dists.shape)   # shape (0, 10)
#         for i, sample in tqdm(enumerate(x)):
#             # print(self.centers.shape) # shape(10, 2048)
#             # print(sample.shape)       # shape 2048
#             dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
#             # print(dist.shape)         # shape 10
#             labels[i] = torch.argmin(dist)
#             # print(labels.shape)       # shape 250000
#             # print(labels[:10])
#             dists = torch.cat([dists, dist.unsqueeze(0)], (0))
#             # print(dists.shape)        # shape (1,10)
#             # print('*')
#         self.labels = labels           # shape 250000
#         if self.started:
#             self.variation = torch.sum(self.dists - dists)
#         self.dists = dists              # 250000, 10
#         self.started = True

#     def update_center(self, x):
#         centers = torch.empty((0, x.shape[1])).to(self.device) # shape (0, 250000)
#         for i in range(self.n_clusters):
#             mask = self.labels == i
#             cluster_samples = x[mask]
#             centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))  # 10, 2048
#         self.centers = centers  # shape (10, 2048)

#     def representative_sample(self):
#         # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
#         # print(self.dists.shape)
#         self.representative_samples = torch.argmin(self.dists, 1)
#         # print(self.representative_samples.shape)  # shape 250000
#         # print('*')
#         return self.representative_samples


# def time_clock(matrix, device):
#     a = time.time()
#     k = KMEANS(max_iter=10,verbose=False,device=device)
#     classifier_n_labels_result = k.fit(matrix)
#     b = time.time()
#     return (b-a)/k.count


# def choose_device(cuda=False):
#     if cuda:
#         device = torch.device("cuda:0")
#     else:
#         device = torch.device("cpu")
#     return device


# if __name__ == "__main__":
#     # import matplotlib.pyplot as plt
#     import pandas as pd
#     import torch
#     import pickle
#     from tqdm import tqdm

#     device = choose_device(True)

#     # read data
#     train_ava_data_list = list(pd.read_csv('train_val_dataset.txt', header=None, sep=' ', index_col=False)[0])
#     test_ava_data_list = list(pd.read_csv('test_dataset_19929.txt', header=None, sep=' ', index_col=False)[0])
#     all_data_list = train_ava_data_list + test_ava_data_list
#     print(len(all_data_list))
#     all_data_tensor = torch.empty(0, 2048).cuda()
#     for i in tqdm(all_data_list):
#         elem_path = '/home/flyingbird/Data/feature_extract/feature_2048/train/' + str(i)
#         with open(elem_path, 'rb') as f:
#             elem_tensor = pickle.load(f, encoding='latin1')
#         all_data_tensor = torch.cat((all_data_tensor, torch.Tensor(elem_tensor).unsqueeze(0).cuda()), dim=0)
#         # print(all_data_tensor.shape)
#         # print('*')
#     print(all_data_tensor.shape)

#     # knn
#     k = KMEANS(max_iter=10, verbose=False, device=device)
#     classifier_result = k.fit(all_data_tensor.cuda())
#     # print(classifier_result[:10])
#     print(classifier_result.shape)
#     classifier_result = classifier_result.cpu().numpy()

#     # save result (img_id : label)
#     dict = {0:all_data_list, 1:classifier_result}
#     pd.DataFrame(dict).to_csv('k_means_all_ava_data_label.txt', sep=' ', header=None, index=False)


#     # speed = time_clock(matrix, device)
#     # print(speed)
#     # cpu_speeds.append(speed)
#     # l1, = plt.plot(2048, cpu_speeds,color = 'r',label = 'CPU')

#     # device = choose_device(True)
#     #
#     # gpu_speeds = []
#     # for i in tqdm([20, 100, 500, 2000, 8000, 20000]):
#     #     matrix = torch.rand((250000, i)).to(device)
#     #     speed = time_clock(matrix,device)
#     #     gpu_speeds.append(speed)
#     # l2, = plt.plot([20, 100, 500, 2000, 8000, 20000], gpu_speeds, color='g',label = "GPU")

#     # plt.xlabel("num_features")
#     # plt.ylabel("speed(s/iter)")
#     # plt.title("Speed with cuda")
#     # plt.legend(handles = [l1],labels = ['GPU'],loc='best')
#     # plt.savefig("speed.jpg")
