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

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
#warnings.filterwarnings("ignore")

class fMoWRGBDataset(Dataset):
    '''fMoW RGB Dataset'''
    def __init__(self, csv_path, transforms=None):
        """
        Args:
            csv_path (string): path to csv file (Works for /atlas/u/buzkent/patchdrop/data/fMoW/train_62classes.csv fmow-rgb)
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
        self.categories = {}
        
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
        if category in self.categories:
            label = self.categories[category]
        else:
            self.categories[category] = len(self.categories)
            label = self.categories[category]
        if self.transforms:
            # note that images is a list of 2 images if TwoCropTranform is used
            images = self.transforms(image)
            
        return (images, image_path, label)



if __name__ == '__main__':
    
    import torchvision.transforms as transforms
    from tqdm import tqdm
    d1 = fMoWMultibandDataset('/atlas/u/pliu1/housing_event_pred/data/fmow-sentinel-filtered-csv/val.csv', transforms.Resize(32))
    location_ids = set()
    categories = set()
    for item in tqdm(d1):
        location_ids.add(item[2])
        categories.add(item[3])
    print(len(location_ids))
    print(len(categories))
    #print(d1[0][0].shape, d1[0][1], d1[0][2])
    #print(d1[0][0].sum())
    """
    
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





    
        