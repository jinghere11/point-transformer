import numpy as np
import random
import SharedArray as SA

import torch
import torch.nn.functional as F

from util.voxelize import voxelize

from nilearn import surface
import nibabel as nib

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def small2small(small_label, hemisphere):

    small_label[small_label == -1] = 0
    if hemisphere == 'L':
        small_label = (small_label+1)//2
    elif hemisphere == 'R':
        small_label = small_label//2
    return np.array(small_label)

def meshto32klabel(meshlabel, hemisphere, trans=True):
    file = "/nfs2/users/zj/v1/Brainnetome/fsaverage.{}.BN_Atlas.10k_fsaverage.label.gii".format(hemisphere)
    label_model = surface.load_surf_data(file)
    label = np.zeros(len(label_model))
    if trans:
        if hemisphere == 'R':
            meshlabel = meshlabel*2
        elif hemisphere == 'L':
            meshlabel = meshlabel*2-1
            meshlabel[meshlabel==-1] = 0
    path = "/nfs2/users/zj/v1/Brainnetome/metric_index_{}_fsaverage10k.txt".format(hemisphere)
    select_ind = np.loadtxt(path).astype(int)
    label[select_ind] = meshlabel
    return label

def saveGiiLabel(data, hemisphere, savepath):
    '''
    save data as gii format
    template_path: the path template
    the length of data is not required same to the template but should match the relevent surface points' size
    path and save_name is the saving location of gii file
    '''
 
    # BNA
    label = meshto32klabel(data, hemisphere, trans=True).astype('int32')
    template_path = "/nfs2/users/zj/v1/Brainnetome/fsaverage.{}.BN_Atlas.10k_fsaverage.label.gii".format(hemisphere)

    # # scheafer
    # label = np.zeros(len(original_label.get_fdata()))
    original_label = nib.load(template_path)
    a = nib.gifti.gifti.GiftiDataArray(label.astype('int32'),intent='NIFTI_INTENT_LABEL')
    new_label = nib.gifti.gifti.GiftiImage( meta = original_label.meta, labeltable = original_label.labeltable)
    new_label.add_gifti_data_array(a)
    nib.save(new_label, savepath)
    return None


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def data_sparse(x, k = 0.8):      #L:9391 R:9409
    topk, indices = x.topk(k = int(k * x.shape[1]), dim = 1)
    res = torch.autograd.Variable(0*x)
    res = res.scatter(1,indices,topk)
    # print("x.shape = ", x.size())
    # print("res.shape = ", res.size())
    return res


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):

    if transform:
        coord, feat, label = transform(coord, feat, label)
    # if voxel_size:
    #     coord_min = np.min(coord, 0)
    #     coord -= coord_min
    #     uniq_idx = voxelize(coord, voxel_size)
    #     coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    # if voxel_max and label.shape[0] > voxel_max:
    #     init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
    #     crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
    #     coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    # # norm in space, not in time dimension
    # feat = F.softmax(feat, dim=0)
    feat = data_sparse(feat)
    label = torch.LongTensor(label)
    return coord, feat, label

def calROIFCRS(sSeries,tSeries):
    sSeries = sSeries-np.mean(sSeries,0)
    normss = np.sqrt(np.sum(sSeries**2,0))
    normss[normss<=0] = 1e-9
    sSeries = sSeries/normss
    
    tSeries = tSeries-np.mean(tSeries,0)
    normts = np.sqrt(np.sum(tSeries**2,0))
    normts[normts<=0] = 1e-9
    tSeries = tSeries/normts
    
    corr_mat = np.matmul(sSeries.transpose(),tSeries)
    return corr_mat

def generateCor2(atlas_roi,tc_matrix_clean_trun):
    timePoints,verticalsNum = np.shape(tc_matrix_clean_trun)
    
    maxAtlas = int(np.max(atlas_roi))
    minAtlas = int(np.min(atlas_roi))
    averageRef = np.zeros([timePoints,maxAtlas-minAtlas+1])
    corrList = np.zeros([verticalsNum,maxAtlas-minAtlas+1])

    #cal mean according to atlas
    for i in range(maxAtlas-minAtlas+1):
        index = np.argwhere(np.squeeze(atlas_roi) == i+minAtlas)
        index = np.transpose(np.squeeze(index))
        index = index.tolist()
        region_feature = tc_matrix_clean_trun[:,index]
        averageRef[:,i] = np.mean(region_feature,1)
    corrList = calROIFCRS(tc_matrix_clean_trun,averageRef)
    return corrList

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # import pdb
    # pdb.set_trace()
    adj = adj.to_dense()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized

    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]).todense())
    t_k.append(scaled_laplacian.todense())

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k
