import os
import urllib.request
import sys
import zipfile
import gzip
import tarfile
import bz2
import datetime
import numpy as np
import torch
# import cv2
import random
import torch.distributed as dist
from multiprocessing import Pool
from datetime import timedelta
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler


# gisette dataset
url_gisette = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/'
url_Gisette = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/'
FILENAME_X = 'gisette_processed_x.npy'
FILENAME_Y = 'gisette_processed_y.npy'

# download & extract
def url_download(url, name):
    urllib.request.urlretrieve(url, name)
    return

def extract_zip(location, filename):
    zip_file = zipfile.ZipFile(os.path.join(location, filename))
    for names in zip_file.namelist():
        zip_file.extract(names, location)
    zip_file.close()

def extract_gz(location, filename):
    f_name = filename.replace(".gz", "")
    g_file = gzip.GzipFile(os.path.join(location, filename))
    open(os.path.join(location, f_name), "wb+").write(g_file.read())
    g_file.close()

def extract_tar(location, filename):
    tar = tarfile.open(os.path.join(location, filename))
    names = tar.getnames()
    for name in names:
        tar.extract(name, location)
    tar.close()

def extract_bz2(location, filename):
    filepath = os.path.join(location, filename)
    zipfile = bz2.BZ2File(filepath)
    data = zipfile.read()
    newfilepath = filepath[:-4]
    open(newfilepath, 'wb').write(data)

extract_map = {
    'zip':extract_zip,
    'gz':extract_gz,
    'tar':extract_tar,
    'bz2':extract_bz2
}

def download_extract(url, cache_location, download_name, extract=None, extract_name='dfgfdg'):
    if not os.path.exists(os.path.join(cache_location, download_name)) and not os.path.exists(os.path.join(cache_location, extract_name)):
        print('Downloading '+download_name)
        url_download(url+download_name,
            os.path.join(cache_location, download_name))
        print(download_name+' downloaded')

    if not os.path.exists(os.path.join(cache_location, extract_name)):
        if extract is not None:
            print('extracting '+download_name)
            extract_map[extract](cache_location, download_name)
            print(download_name+' extracted')

'''
def load_image_gray(img_path):
    img = cv2.imread(img_path, 0)
    return img
'''

def write_res(filepath, res):
    res = np.array(res)
    np.save(filepath, res)

def var(list):
    avg = 0
    for i in range(len(list)):
        avg += list[i]
    avg /= len(list)
    var = 0
    for i in range(len(list)):
        var += (list[i] - avg) * (list[i] - avg)
    var /= len(list)
    return avg, var

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

##################### preprocess ##########################
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup(args):
    args = args.__dict__.copy()

    return args


################## multiprocessing ########################
def launch_a_process(rank, args, target, minutes=720):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args['master_ip'], master_port=args['master_port'])
    dist.init_process_group(backend='gloo',
                            init_method=dist_init_method,
                            # If you have a larger dataset, you will need to increase it.
                            timeout=timedelta(minutes=minutes),
                            world_size=args['num_processes'],
                            rank=rank)
    assert torch.distributed.get_rank() == rank
    target(rank, args)

def synchronize(num_processes):
    if num_processes > 1:
        dist.barrier()


################## Data Loader ###########################
class PageLoader(Dataset):
    def __init__(self, dataset, large_batch, small_batch, p, block=1, drop_last=True):
        self.dataset = dataset
        self.large_batch = large_batch
        self.small_batch = small_batch
        self.p = p
        self.drop_last = drop_last
        self.block = block
        self.sampler = RandomSampler(dataset)
        self.cnt = 0
        
    
    def __iter__(self):
        data = torch.Tensor()
        label = torch.Tensor().int()
        flag_toss = True
        
        for idx in self.sampler:
            if flag_toss:
              toss = np.random.binomial(1, self.p)
              my_bsz = self.large_batch if toss == 1 else self.small_batch
              flag_toss = False
              if self.cnt < self.block:
                my_bsz = self.large_batch
                self.cnt += 1
            data = torch.cat([data, self.dataset[idx][0].unsqueeze(0)])
            label = torch.cat([label, torch.tensor(self.dataset[idx][1]).unsqueeze(0)])
            while data.size(0) >= my_bsz:
                if data.size(0) == my_bsz:
                    yield (data, label)
                    data = torch.Tensor()
                    label = torch.Tensor().int()
                else:
                    return_data, data = data.split([my_bsz, data.size(0) - my_bsz])
                    return_label, label = label.split([my_bsz, label.size(0) - my_bsz])
                    yield (return_data, return_label)
                flag_toss = True
        if data.size(0) > 0 and not self.drop_last:
            yield (data, label)