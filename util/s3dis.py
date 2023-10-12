import os

import torch
import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare, generateCor2, data_sparse
from util.data_util import small2small
from nilearn.surface import load_surf_data


import scipy.sparse as sp
from util.data_util import sparse_mx_to_torch_sparse_tensor, chebyshev_polynomials

class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        data_list = sorted(os.listdir(data_root))
        # data_list = [item[:-4] for item in data_list if 'Area_' in item]
        data_list = [item for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop

def load_adj():
    adjsm = np.load("/nfs2/users/zj/v1/pointnet/point-transformer/adjsm.npy",allow_pickle=True)
    adjsm = sp.csr_matrix(adjsm.all())
    adjsm = sparse_mx_to_torch_sparse_tensor(adjsm)
    T_k = chebyshev_polynomials(adjsm, 4)

    T_k = torch.FloatTensor(np.array(T_k))
    T_k = T_k.cuda(non_blocking=True)
    return T_k

def load_morph_data(data_path, choice):
    output = []
    len_data = len(choice)
    for morph_path in data_path:
        point_set_morph = load_surf_data(morph_path)
        #resample
        point_set = point_set_morph[choice]
        point_set = point_set - np.mean(point_set) # center
        dist = np.max(np.abs(point_set))
        point_set = point_set / dist #scale
        output.append(point_set)
    output = np.stack(output).T
    return output    # thickness, sulc, curv, area, volume
    


class swDataset(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=None, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        trainset_percent = 0.5
        with open(os.path.join(data_root,"datalist_sw_t1.txt"),"r") as f:
            sub_list = f.readlines()
            sub_list = [ix[:-1].split() for ix in sub_list]
        func_root = "/nfs2/users/zj/southwest/sw0117_func5/"
        datapath_L = [[os.path.join(data_root,"{}/{}/{}/fsaverage_surf/{}.L.white.10k_fs.surf.gii".format(ix[0],ix[1],ix[1],ix[1])), \
            os.path.join(data_root,"{}/{}/{}/fsaverage_surf/{}.L.thickness.10k_fsaverage.shape.gii".format(ix[0],ix[1],ix[1],ix[1])), \
            os.path.join(data_root,"{}/{}/{}/fsaverage_surf/{}.L.sulc.10k_fsaverage.shape.gii".format(ix[0],ix[1],ix[1],ix[1])), \
            os.path.join(data_root,"{}/{}/{}/fsaverage_surf/{}.L.curv.10k_fsaverage.shape.gii".format(ix[0],ix[1],ix[1],ix[1])), \
            os.path.join(data_root,"{}/{}/{}/fsaverage_surf/{}.L.area.10k_fsaverage.shape.gii".format(ix[0],ix[1],ix[1],ix[1])), \
            os.path.join(data_root,"{}/{}/{}/fsaverage_surf/{}.L.volume.10k_fsaverage.shape.gii".format(ix[0],ix[1],ix[1],ix[1])), \
            os.path.join(func_root,"{}_{}.L.10k_fsaverage.func.gii".format(ix[0],ix[1]))] for ix in sub_list]
        if split == "train":
            self.sub_list = sub_list[:int(len(sub_list)*trainset_percent)]
        elif split == "eval":
            self.sub_list = sub_list[int(len(sub_list)*trainset_percent):]
        else:
            self.sub_list = sub_list

        self.label = np.load("/nfs2/users/zj/v1/Brainnetome/mesh_label_all_L_fsaverage10k.npy")
        self.label = small2small(self.label, "L")
        self.choice = np.loadtxt("/nfs2/users/zj/v1/Brainnetome/metric_index_L_fsaverage10k.txt").astype(int)

        for idx, item in enumerate(self.sub_list):
            if not os.path.exists("/dev/shm/{}".format("_".join(item))):

                point_set_white = load_surf_data(datapath_L[idx][0])
                #resample
                point_set = point_set_white[0][self.choice, :]
                point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
                dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
                point_set = point_set / dist #scale
                morph_data = load_morph_data(datapath_L[idx][1:-1], self.choice)   # thickness, sulc, curv, area, volume

                func_set = load_surf_data(datapath_L[idx][-1])[:, self.choice]
                # func_set = func_set.T
                func_set_cor = generateCor2(self.label,func_set)
                # initAtlas=np.expand_dims(np.argmax(func_set_cor,axis=1),axis=1)
                # data_total =np.concatenate([point_set, morph_data, initAtlas],axis=1)

                data_total =np.concatenate([point_set, morph_data, func_set_cor],axis=1)


                sa_create("shm://{}".format("_".join(item)), data_total)
        self.data_idx = np.arange(len(self.sub_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):

        data_idx = self.sub_list[idx % len(self.sub_list)]
        data = SA.attach("shm://{}".format("_".join(data_idx))).copy()

        coord, feat = data[:, :3], data[:, 8:]
        initAtlas=np.expand_dims(np.argmax(feat,axis=1),axis=1)
        label = self.label
        mask = np.squeeze(initAtlas) == label
        # transform is crop, hue_tuning, etc. Default none, todo: Mask is not included in transform.
        coord, feat, label, mask = data_prepare(coord, feat, label, mask, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        # print("coord.shape = {}, feat.shape = {}, label.shape = {},".format(coord.shape, feat.shape, label.shape))

        # stft_band = 8
        # with torch.no_grad():
        #     x0_ft = torch.stft(feat, stft_band)
        #     total_samples = x0_ft.shape[0]
        #     self.bs = total_samples // 9391
        #     x0_ft = x0_ft.reshape(self.bs, 9391, -1)
        #     x0_fc = torch.matmul(x0_ft,x0_ft.transpose(-1,-2)).reshape(total_samples, -1)

        return coord, feat, label, mask

    def __len__(self):
        return len(self.data_idx) * self.loop


# class swDataset(data.Dataset):
#     def __init__(self,
#                  dataset_path,
#                  npoints=2500,
#                  trainset_percent=0.8,
#                  classification=False,
#                  class_choice=None,
#                  split='train',
#                  data_augmentation=True):
#         self.npoints = npoints
#         self.root = dataset_path
#         # self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')

#         self.data_augmentation = data_augmentation
#         self.classification = classification
#         self.seg_classes = {}
#         with open(os.path.join(self.root,"datalist_sw_t1.txt"),"r") as f:
#             sub_list = f.readlines()
#             sub_list = [ix[:-1].split() for ix in sub_list]
#         datapath_L = [os.path.join(self.root,"{}/{}/{}.L.white.10k_fs.surf.gii".format(ix[0],ix[1],ix[1])) for ix in sub_list]
#         if split == "train":
#             self.datapath = datapath_L[:int(len(datapath_L)*trainset_percent)]
#         elif split == "eval":
#             self.datapath = datapath_L
#         else:
#             self.datapath = datapath_L[int(len(datapath_L)*trainset_percent):]
#         self.seg = np.load("/home/zjlab/data_disk/panbl/v1/Brainnetome/mesh_label_all_L_fsaverage10k.npy")
#         self.seg = small2small(self.seg, "L")

#         # datapath_R = [os.path.join(self.root,"{}/{}/{}.R.white.10k_fs.surf.gii".format(ix[0],ix[1],ix[1])) for ix in sub_list]
#         # self.datapath = [x for y in zip(datapath_L, datapath_R) for x in y]
#         self.choice = np.loadtxt("/home/zjlab/data_disk/panbl/v1/Brainnetome/metric_index_L_fsaverage10k.txt").astype(int)
#         self.num_seg_classes = 106

#         print(self.seg_classes, self.num_seg_classes)

    # def __getitem__(self, index):
    #     fn = self.datapath[index]

    #     # point_set = surface.load_surf_data(fn)[0].astype(np.float32)
    #     point_set = load_surf_data(fn)
    #     seg = self.seg
    #     # choice = np.random.choice(len(seg), self.npoints, replace=True)
    #     choice = self.choice
    #     #resample
    #     point_set = point_set[choice, :]
    #     # seg = seg[choice]

    #     point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
    #     dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    #     point_set = point_set / dist #scale

    #     if self.data_augmentation:
    #         theta = np.random.uniform(0,np.pi*2)
    #         rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    #         point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
    #         point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

    #     point_set = torch.from_numpy(point_set)
    #     seg = torch.from_numpy(seg).to(torch.long)

    #     return point_set, seg, fn

#     def __len__(self):
#         return len(self.datapath)
