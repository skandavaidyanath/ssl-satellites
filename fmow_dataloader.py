from numpy.lib.function_base import select
import pandas as pd
import numpy as np
import warnings
import random
import pickle
from glob import glob
import os
import rasterio
from rasterio.enums import Resampling
import torch
from tqdm import tqdm

from torch.utils.data.dataset import Dataset
from PIL import Image

from moco.loader import RandomDropBands

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
#warnings.filterwarnings("ignore")

class fMoWRGBDataset(Dataset):
    '''fMoW RGB Dataset'''
    def __init__(self, csv_path, transforms=None):
        """
        Args:
            csv_path (string): path to csv file (Works for /atlas/u/pliu1/housing_event_pred/data/fmow-csv/fmow-train.csv fmow-rgb)
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transforms
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        # note that img_as_tensor is a list of 2 images if TwoCropTranform is used
        return (img_as_tensor, single_image_name, single_image_label)

    def __len__(self):
        return self.data_len


class fMoWMultibandDataset(Dataset):
    '''fMoW Multiband Dataset'''
    def __init__(self, 
                 csv_path, 
                 transforms=None,
                 resize=64):
        """ Initialize the dataset.
        
        Args:
            csv_file (string): Path to the csv file with annotations.  (Works for /atlas/u/pliu1/housing_event_pred/data/fmow-sentinel-filtered-csv/train.csv fmow-sentinel)
            transform (callable, optional): Optional transform to be applied
                on tensor images.
            resize: Size to load images as
        """
        self.data_info = pd.read_csv(csv_path)
        self.indices = self.data_info.index.unique().to_numpy()
        self.data_len = len(self.indices)
        self.transforms = transforms
        self.resize = resize
        self.categories = pickle.load(open('fmow-category-labels.pkl', 'rb'))
        
    def __len__(self):
        return self.data_len
    
    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            #img = data.read(
            #    out_shape=(data.count, self.resize, self.resize),
            #    resampling=Resampling.bilinear
            #)
            img = data.read()
        return img

    def __getitem__(self, idx):
        index = self.indices[idx]
        selection = self.data_info.loc[index]
        
        image_path = selection["image_path"]
        image = torch.FloatTensor(self.open_image(image_path))
        category = selection["category"] 
        label = self.categories[category]
        if self.transforms:
            # note that images is a list of 2 images if TwoCropTranform is used
            images = self.transforms(image)
            
        return (images, image_path, label)
    
    
class fMoWJointDataset(Dataset):
    '''fMoW Joint Dataset'''
    def __init__(self, 
                 csv_path, 
                 sentinel_transforms=None,
                 rgb_transforms=None,
                 joint_transform='either'):
        """ Initialize the dataset.
        
        Args:
            csv_file (string): Path to the csv file with annotations.  (Works for /atlas/u/pliu1/housing_event_pred/data/fmow-sentinel-filtered-csv/train.csv fmow-sentinel)
            sentinel_transforms (callable, optional): Optional transform to be applied
                on tensor images.
            rgb_transforms (callable, optional): Optional transform to be applied
                on tensor images.
            joint_transform: 'either' or 'drop' or 'both' or 'rgb' or 'sentinel'
        """
        self.data_info = pd.read_csv(csv_path)
        self.data_info = self.data_info[self.data_info['fmow_path'].notna()]
        self.indices = self.data_info.index.unique().to_numpy()
        self.data_len = len(self.indices)
        self.sentinel_transforms = sentinel_transforms
        self.rgb_transforms = rgb_transforms
        self.joint_transform = joint_transform
        self.categories = pickle.load(open('fmow-category-labels.pkl', 'rb'))
        
    def __len__(self):
        return self.data_len
    
    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()
        return img

    def __getitem__(self, idx):
        index = self.indices[idx]
        selection = self.data_info.loc[index]
        
        sentinel_image_path = selection["image_path"]
        rgb_image_path = selection["fmow_path"]
        
        sentinel_image = torch.FloatTensor(self.open_image(sentinel_image_path))
        rgb_image = Image.open(rgb_image_path)
        
        if self.sentinel_transforms:
            # TwoCropsTransform may be used here
            sentinel_images = self.sentinel_transforms(sentinel_image)
        if self.rgb_transforms:
            # TwoCropsTransform may be used here
            rgb_images = self.rgb_transforms(rgb_image)
        
        if isinstance(sentinel_images, list):
            assert self.joint_transform in ['either', 'drop', 'both']
            assert sentinel_images[0].shape[1:] == rgb_images[0].shape[1:]
            assert sentinel_images[1].shape[1:] == rgb_images[1].shape[1:]
            joint_images = [torch.cat([sentinel_images[i], rgb_images[i]], dim=0) for i in range(2)]
        else:
            assert self.joint_transform in ['both', 'rgb', 'sentinel', 'drop']
            assert sentinel_images.shape[1:] == rgb_images.shape[1:]
            joint_images = torch.cat([sentinel_images, rgb_images], dim=0)
        
        if self.joint_transform == 'either':
            rgb_mask = torch.stack([torch.ones(joint_images[0].shape[1:]) if i in range(sentinel_images[0].shape[0]) else torch.zeros(joint_images[0].shape[1:]) for i in range(joint_images[0].shape[0])])
            sentinel_mask = torch.stack([torch.zeros(joint_images[0].shape[1:]) if i in range(sentinel_images[0].shape[0]) else torch.ones(joint_images[0].shape[1:]) for i in range(joint_images[0].shape[0])])
            
            pair_type = np.random.randint(0,4)
            if pair_type == 0:
                ## Sentinel, Sentinel
                joint_images = [image * rgb_mask for image in joint_images]
            if pair_type == 1:
                ## Sentinel, RGB
                joint_images = [joint_images[0] * rgb_mask, joint_images[1] * sentinel_mask]
            if pair_type == 2:
                ## RGB, Sentinel
                joint_images = [joint_images[0] * sentinel_mask, joint_images[1] * rgb_mask]
            if pair_type == 3:
                ## RGB, RGB
                joint_images = [image * sentinel_mask for image in joint_images]
        elif self.joint_transform == 'drop':
            if isinstance(joint_images, list):
                joint_images = [RandomDropBands()(image) for image in joint_images]
            else:
                joint_images = RandomDropBands()(joint_images)
        elif self.joint_transform == 'rgb':
            sentinel_mask = torch.stack([torch.zeros(joint_images.shape[1:]) if i in range(sentinel_images.shape[0]) else torch.ones(joint_images.shape[1:]) for i in range(joint_images.shape[0])])
            joint_images = joint_images * sentinel_mask
        elif self.joint_transform == 'sentinel':
            rgb_mask = torch.stack([torch.ones(joint_images.shape[1:]) if i in range(sentinel_images.shape[0]) else torch.zeros(joint_images.shape[1:]) for i in range(joint_images.shape[0])])
            joint_images = joint_images * rgb_mask
        elif self.joint_transform == 'both':
            ## Nothing to do here
            pass
        
        category = selection["category"] 
        label = self.categories[category]
            
        return (joint_images, [sentinel_image_path, rgb_image_path], label)



if __name__ == '__main__':
    
    import torchvision.transforms as transforms
    from moco.loader import TwoCropsTransform
    
#     d = fMoWJointDataset('/atlas/u/pliu1/housing_event_pred/data/fmow-sentinel-filtered-csv/val.csv',
#                             TwoCropsTransform(transforms.RandomResizedCrop(4),
#                                             transforms.RandomResizedCrop(4)),
#                             TwoCropsTransform(transforms.Compose([transforms.RandomResizedCrop(4), transforms.ToTensor()]),
#                                            transforms.Compose([transforms.RandomResizedCrop(4), transforms.ToTensor()])),
#                             'drop')
    
    d = fMoWJointDataset('/atlas/u/pliu1/housing_event_pred/data/fmow-sentinel-filtered-csv/val.csv',
                            transforms.RandomResizedCrop(4),
                            transforms.Compose([transforms.RandomResizedCrop(4), transforms.ToTensor()]),
                            'both')
    
   
    print(d[0][0])
    print('**********')
    print(d[1][0])
    print('***********')
    print(d[2][0])
    """
    import torchvision.transforms as transforms
    from tqdm import tqdm
    d1 = fMoWMultibandDataset('/atlas/u/pliu1/housing_event_pred/data/fmow-sentinel-filtered-csv/val.csv')
    for item in tqdm(d1):
        pass
    print(d1.categories)
    pickle.dump(d1.categories, open('fmow_category_labels.pkl', 'wb'))
    #print(d1[0][0].shape, d1[0][1], d1[0][2])
    #print(d1[0][0].sum())
    
    # Calculate the channel stats
    df = pd.read_csv('/atlas/u/pliu1/housing_event_pred/data/fmow-sentinel-filtered-csv/train.csv')
    indices = df.index.unique().to_numpy()
    num_channels = 13

    SEED = 1
    NUM_SAMPLES = 2500
    BINS = 200
    LOG = True

    def open_image(image_path):
        #with rasterio.open(image_path) as data:
        #    image = data.read()
        data = rasterio.open(image_path)
        image = data.read()
        data.close()
        del data
        return image

    def channel_values(index):
        image_path = df.loc[index]["image_path"]
        image = open_image(image_path)
        channel_values = image.reshape(num_channels, -1)
        if LOG:
            return np.log(channel_values+1e-10)
        return channel_values
    """

    """
    from multiprocessing.pool import ThreadPool
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import os

    random.seed(SEED)
    random.shuffle(indices)
    pool = ThreadPool()
    #pool = multiprocessing.Pool()
    results = list(tqdm(pool.imap(channel_values, indices[:NUM_SAMPLES]), total=len(indices[:NUM_SAMPLES])))
    pool.close()

    if not os.path.exists(f'channel_plots/bins={BINS}_log={LOG}'):
        os.makedirs(f'channel_plots/bins={BINS}_log={LOG}')

    results = np.concatenate(results, 1)
    for i in range(results.shape[0]):
        plt.hist(results[i], bins=BINS)
        plt.savefig(f'channel_plots/bins={BINS}_log={LOG}/channel_{i}.png')
        plt.clf()
        plt.close()
    """
    """
    def channel_stats(index):
        image_path = df.loc[index]["image_path"]
        image = open_image(image_path)
        if LOG:
            image = np.log(image+1e-10)
        pixel_num = image.size/num_channels
        channel_sum = np.sum(image, axis=(1, 2))
        channel_sum_squared = np.sum(np.square(image), axis=(1, 2))
        height = image.shape[1]
        width = image.shape[2]
        del image
        return (image_path, pixel_num, channel_sum, channel_sum_squared, height, width)

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool()
    results = tqdm(pool.imap(channel_stats, indices), total=len(indices))
    pool.close()
    
    pixel_num = 0
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)
    heights, widths = [], []

    for result in results:
        pixel_num += result[1]
        channel_sum += result[2]
        channel_sum_squared += result[3]
        heights.append(result[4])
        widths.append(result[5])

    channel_means = channel_sum / pixel_num
    channel_stds = np.sqrt(channel_sum_squared / pixel_num - np.square(channel_means))

    if LOG:
        stats = {'log_channel_means': channel_means, 
                'log_channel_stds': channel_stds,
                'mean_height': np.mean(heights),
                'mean_width': np.mean(widths),
                'median_height': np.median(heights),
                'median_width': np.median(widths)}
        print(stats)
        pickle.dump(stats, open('./fmow-multiband-log-stats.pkl', 'wb'))
        
    else:
        stats = {'channel_means': channel_means, 
                'channel_stds': channel_stds,
                'mean_height': np.mean(heights),
                'mean_width': np.mean(widths),
                'median_height': np.median(heights),
                'median_width': np.median(widths)}
        print(stats)
        pickle.dump(stats, open('./fmow-multiband-stats.pkl', 'wb'))
    """





    
        