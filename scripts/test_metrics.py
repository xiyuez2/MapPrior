import numpy as np
# import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import glob
import argparse
from sklearn import metrics
from tqdm import tqdm

def take_threshold(data,threshold = ()):
    # data: batch_size, 6, w, h
    for i in range(6):
        data[:,i] = data[:,i] > threshold[i]
    return data

class eval_data(Dataset):
    def __init__(self, save_dir,prefix_res,prefix_gt):
        self.save_dir = save_dir
        self.prefix_res = prefix_res
        self.prefix_gt = prefix_gt
        res_len = len(glob.glob(save_dir + '/' + prefix_res + '*.npy'))
        gt_len = len(glob.glob(save_dir + '/' + prefix_gt + '*.npy'))
        if not gt_len == res_len:
            raise ValueError('gt and res must have same num of files')
        self.lens = gt_len

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        res = np.load(self.save_dir + '/' + self.prefix_res + str(idx) +'.npy')
        gt = np.load(self.save_dir + '/' + self.prefix_gt + str(idx) +'.npy')

        return res, gt

def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="",
        help="dir to save the res when testing",
    )
    parser.add_argument(
        "-g",
        "--gt_prefix",
        type=str,
        default="gt_",
        help="prefix of file name used for gt",
    )
    parser.add_argument(
        "-r",
        "--res_prefix",
        type=str,
        default="res_",
        help="result of file name used for gt",
    )
    opt, unknown = parser.parse_known_args()
    return opt


def IOUs(gt,pre,thes=0.5):
    # compute IOU for each classes of the map and mIOU
    # gt is the ground truth and pre is the corresponding prediction from the model
    # both gt and pre is an array of shape [num_samples, map_classes, w, h]
    IOUs = []
    print('computing IOU...')
    threshold = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    for i in tqdm(range(6)):
        cur_pre = pre[:,i,:,:] > threshold[i]
        cur_gt = gt[:,i,:,:] > thes
        iou = np.sum(cur_pre & cur_gt) / np.sum(cur_pre | cur_gt)
        IOUs.append(iou)
        del cur_pre, cur_gt
    print("===============IOUs=============")
    print("drivable_area | ped_crossing | walkway | stop_line | carpark_area | divider | mean")
    print('%.4f        | %.4f       | %.4f  | %.4f    | %.4f       | %.4f  | %.4f' % (IOUs[0],IOUs[1],IOUs[2],IOUs[3],IOUs[4],IOUs[5],np.mean(IOUs)))
    IOUs.append(np.mean(IOUs))
    print(IOUs)
    return IOUs
def ACC(gt,pre,thes=0.5):
    ACCs = []
    print('computing ACC...')
    for i in tqdm(range(6)):
        cur_pre = pre[:,i,:,:] > thes
        cur_gt = gt[:,i,:,:] > thes
        acc = np.mean(cur_pre == cur_gt)
        del cur_pre, cur_gt
        ACCs.append(acc)
    print("===============ACCs=============")
    print("drivable_area | ped_crossing | walkway | stop_line | carpark_area | divider | mean")
    print('%.4f        | %.4f       | %.4f  | %.4f    | %.4f       | %.4f  | %.4f' % (ACCs[0],ACCs[1],ACCs[2],ACCs[3],ACCs[4],ACCs[5],np.mean(ACCs)))
    ACCs.append(np.mean(ACCs))
    print(ACCs)
    return ACCs


def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    gamma = (1/len(X))**0.5
    XX = metrics.pairwise.polynomial_kernel(X, X, gamma = gamma, degree=degree, coef0=coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, gamma = gamma, degree=degree, coef0=coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, gamma = gamma, degree=degree, coef0=coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def MMDs(gts,res):
    gts = (gts > 0.5).reshape((len(gts),-1))#[idx]
    res = (res > 0.5).reshape((len(gts),-1))#[idx]

    print(np.shape(gts),np.shape(res))
    print("=========mmd_scores==========")
    print("linear:")
    print(mmd_linear(gts,res))

    # rbf and poly kernels are disabled for now 
    # they need more cpu mem
    # print("rbf")
    # print(mmd_rbf(gts,res))
    
    # print("poly")
    # print(mmd_poly(gts,res))



def metrics_main(save_dir,prefix_res = 'res_',prefix_gt = 'gt_'):
    # the main function to calcuate metrics.
    dataset = eval_data(save_dir,prefix_res,prefix_gt)
    dataloader = DataLoader(dataset=dataset, batch_size=1,num_workers=4)
    res_total = np.zeros((len(dataset),6,200,200))
    gt_total = np.zeros((len(dataset),6,200,200))
    print('reading data...')
    for i,data_batch in enumerate(tqdm(dataloader)):
        res_total[i], gt_total[i] = data_batch[0][0],data_batch[1][0]
    IOUs(gt_total,res_total)
    # ACC(gt_total,res_total) accuracy is unnecessary
    MMDs(gt_total,res_total)



if __name__ == '__main__':
    parser = get_parser()
    # parser.save_dir: the data path where you saved your samples 
    # parser.res_prefix: the prefix for sampling results when saving samples
    # parser.gt_prefix: the prefix for ground truth when saving samples
    metrics_main(parser.save_dir,parser.res_prefix,parser.gt_prefix)


