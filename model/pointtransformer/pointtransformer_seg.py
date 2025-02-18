import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from lib.pointops.functions import pointops
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np




class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.linear_pos = nn.Linear(out_planes, 3, bias=False)
        # self.bn_pos = nn.BatchNorm1d(3)
        # self.relu_pos = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b); o=[ 9391, 18782, 28173, 37564, 46955, 56346, 65737, 75128]

        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o

        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]
       
    # def forward(self, pxo):
    #     p, x, o = pxo  # (n, 3), (n, c), (b)
    #     if self.stride != 1:
    #         n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
    #         for i in range(1, o.shape[0]):
    #             count += (o[i].item() - o[i-1].item()) // self.stride
    #             n_o.append(count)
    #         n_o = torch.cuda.IntTensor(n_o)
    #         idx = pointops.furthestsampling(p, o, n_o)  # (m)
    #         n_p = p[idx.long(), :]  # (m, 3)
    #         x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
    #         x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
    #         x = self.pool(x).squeeze(-1)  # (m, c)
    #         p, o = n_p, n_o
    #     else:
    #         x = self.relu(self.bn(self.linear(x)))  # (n, c)
    #     # print("p.shape = ", p.shape)
    #     # print("x.shape = ", x.shape)
    #     # print("o = ", o)

    #     return [p, x, o]


class LinearFunc(Module):
    def __init__(self, in_planes, out_planes):
        super(LinearFunc,self).__init__()
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # x = self.relu(self.bn(self.linear(x)))
        x = self.linear(x)
        # x = x[:,:3].contiguous()
        return x

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_support, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nsupport = num_support
        self.linear = nn.Linear(in_features*num_support, out_features)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        output_list = []
        bs = input.shape[0] // adj.shape[-1]
        for ix in range(bs):
            input_single_st, input_single_ed = adj.shape[-1] * ix, adj.shape[-1] * (ix+1)
            input_single = input[input_single_st:input_single_ed]
            cheb_x = input_single.unsqueeze(2)
            for adj_ix in range(1, self.nsupport):
                x1 = torch.mm(adj[adj_ix], input_single)
                cheb_x = torch.cat((cheb_x, x1.unsqueeze(2)), 2)
            cheb_x = cheb_x.reshape([input_single.shape[0], -1])
            output_single = self.linear(cheb_x)
            # print("self.linear.weight.grad: ", self.linear.weight.grad)
            if self.bias is not None:
                output_single = output_single + self.bias
            output_list.append(output_single)
        return torch.cat(output_list, dim=0)
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # self.depthwise = nn.Linear(in_channels, in_channels)
        # self.bn_d = nn.BatchNorm1d(in_channels)
        # self.relu_d = nn.ReLU(inplace=True)

        # self.pointwise = nn.Linear(in_channels, out_channels)
        # self.bn_p = nn.BatchNorm1d(out_channels)
        # self.relu_p = nn.ReLU(inplace=True)


    def forward(self, x):
        # out = self.relu_d(self.bn_d(self.depthwise(x)))
        out = self.pointwise(self.depthwise(x))

        return out


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=106, gcn_num=2, cheb_order=7, resolution=9391, dropout=0.5, training=False):
        super().__init__()
        self.c = c
        self.gcn_num = gcn_num
        self.in_planes, planes = c, [32, 64, 128, 256, 512]

        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        # stride, nsample = [1, 4, 4, 4, 4], [8, 8, 8, 8, 8]


        self.dropout = dropout
        self.training = training
        self.resolution = resolution

        self.gc0 = GraphConvolution(k, 64, cheb_order)
        self.gch = GraphConvolution(64, 64, cheb_order)
        self.gc1 = GraphConvolution(64, k, cheb_order)
        self.gc_pt = GraphConvolution(k, k, 2)
        # self.pointnet_3d = PointNetDenseCls(k)
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        self.pos_bn = nn.BatchNorm1d(resolution)
        self.ln_gcn = nn.LayerNorm(k)
        self.ln_pt = nn.LayerNorm(k)

        self.conv = DepthwiseSeparableConv(2, 1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        self.smooth = nn.Conv1d(k, k, kernel_size=3,stride=1,padding=1)
        self.avgpool = nn.AvgPool1d(kernel_size=3,stride=1,padding=1)
        self.smooth1 = nn.Sequential(nn.Conv1d(k, k, kernel_size=3,stride=1,padding=1), nn.BatchNorm1d(k), nn.ReLU(inplace=True),nn.Conv1d(k, k, kernel_size=3,stride=1,padding=1))
        self.smooth2 = nn.Conv1d(k, k, kernel_size=3,stride=1,padding=1)
        self.smooth_pt = nn.Conv1d(k, k, kernel_size=3,stride=1,padding=1)



    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))

        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo, adj):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)

        adj_order1 = adj[1]



        global x_seed
        bs = x0.shape[0]//o0[0]
        x_seed = x0.clone()
        x_seed = torch.exp(x_seed/0.02)
        sumNorm =  torch.sum(x_seed,axis=1)
        sumNorm = sumNorm.unsqueeze(1)
        x_seed = x_seed/sumNorm
        list_res = []
        y = F.relu(self.gc0(x_seed, adj))
        list_res.append(y)
        y = F.dropout(y, self.dropout, training=self.training)
        for layer_ix in range(self.gcn_num):
            y = F.relu(self.gch(y, adj))
            list_res.append(y)
        for i in range(self.gcn_num):
            y += list_res[i]
        y = self.gc1(y, adj)

        # y = self.ln_gcn(y)

        p1, x1, o1 = self.enc1([p0, y, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        # x = self.smooth_pt(x.reshape(-1, self.resolution, x.shape[-1]).transpose(-1,-2)).transpose(-1,-2).reshape(-1, x.shape[-1])

        # x = x.reshape(-1, self.resolution, x.shape[-1]).transpose(-1,-2)

        # x = torch.matmul(x, adj_order1).transpose(-1,-2)
        # x = x.reshape(-1, x.shape[-1])


        x = self.ln_pt(x).squeeze()

        output_stack = torch.stack((x,y),dim=2)
        output_stack = self.maxpool(output_stack).squeeze()

        # # output_stack = self.smooth(output_stack.reshape(-1, self.resolution, output_stack.shape[-1]).transpose(-1,-2)).transpose(-1,-2)
        # output_stack0 = output_stack.reshape(-1, self.resolution, output_stack.shape[-1]).transpose(-1,-2)
        # # output_stack = self.avgpool(output_stack0)
        # output_stack = self.smooth(output_stack0)
        # output_stack = output_stack.transpose(-1,-2)

        return output_stack, x, y

def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model
