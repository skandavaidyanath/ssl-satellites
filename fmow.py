import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import rasterio
from rasterio.enums import Resampling
from PIL import Image
import time
from glob import glob
import json
from datetime import datetime
import random
import time

from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)

class FMOWDataset(Dataset):
    categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", 
                  "aquaculture", "archaeological_site", "barn", "border_checkpoint", 
                  "burial_site", "car_dealership", "construction_site", "crop_field", 
                  "dam", "debris_or_rubble", "educational_institution", "electric_substation", 
                  "factory_or_powerplant", "fire_station", "flooded_road", "fountain", 
                  "gas_station", "golf_course", "ground_transportation_station", "helipad", 
                  "hospital", "impoverished_settlement", "interchange", "lake_or_pond", 
                  "lighthouse", "military_facility", "multi-unit_residential", 
                  "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", 
                  "parking_lot_or_garage", "place_of_worship", "police_station", "port", 
                  "prison", "race_track", "railway_bridge", "recreational_facility", 
                  "road_bridge", "runway", "shipyard", "shopping_mall", 
                  "single-unit_residential", "smokestack", "solar_farm", "space_facility", 
                  "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", 
                  "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", 
                  "wind_farm", "zoo"]
    label_types = ['value', 'one-hot']
    
    '''fMoW Dataset'''
    def __init__(self, 
                 csv_file, 
                 years=[*range(2000, 2021)], 
                 categories=None,
                 pre_transform=None,
                 transform=None,
                 label_type='one-hot',
                 resize=64):
        """ Initialize the dataset.
        
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Base directory with wikidata folders and images.
            years (list, optional): List of years to take images from, None to not filter
            categories (list, optional): List of categories to take images from, None to not filter
            pre_transform (callable, optional): Optional transformation to be applied to individual images
                immediately after loading in. If None, each image in a time series is resized to
                to the dimensions of the first image. If specified, transformation should take images to 
                the same dimensions so they can be stacked together
            transform (callable, optional): Optional transform to be applied
                on tensor images.
            label_type (string): 'values' for single regression label, 'one-hot' for one hot labels
            resize: Size to load images as
        """
        self.df = pd.read_csv(csv_file) \
                .sort_values(['category', 'location_id', 'timestamp']) \
                .set_index(['category', 'location_id'])
        
        # Filter by category
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]
        
        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]
        
        self.indices = self.df.index.unique().to_numpy()
        
        self.pre_transform = pre_transform
        self.transform = transform
        
        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:', 
                ', '.join(self.label_types))
        self.label_type = label_type
        
        self.resize = resize
        
    def __len__(self):
        return len(self.indices)
    
    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read(
                out_shape=(data.count, self.resize, self.resize),
                resampling=Resampling.bilinear
            )
            
        return img
            
        
    def __getitem__(self, idx):
        ''' Gets timeseries info for images in one area
        
        Args: 
            idx: Index of loc in dataset
            
        Returns dictionary containing:
            'images': Images in tensor of dim Batch_SizexLengthxChannelxHeightxWidth
                Channel Order: R,G,B, NIR
            'labels': Labels of each image. Depends on 'label_type'
                'regression': First year is -0.5, second year is 0.5, so on. 
                'one-hot': Returns one-hot encoding of years
                'classification': for class labels by year where '0' is no construction (#labels = years)
            'years': Years of each image in timeseries
            'id': Id of image location
            'type': Type of image location
            'is_annotated': True if annotations for dates are provided
            'year_built': Year built as labeled in dataset
        '''
        index = self.indices[idx]
        selection = self.df.loc[index]

        image_paths = selection['image_path']
        
        #images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = [torch.FloatTensor(self.open_image(img_path)) / 255 for img_path in image_paths]

        if self.pre_transform:
            images = torch.cat([self.pre_transform(image.unsqueeze(0)) for image in images])
        else:
            shape = images[0].shape[-2:]
            resize = transforms.Resize(shape)
            resized = [resize(image.unsqueeze(0)) for image in images]
            images = torch.cat(resized)
        
        category, location_id = index
        
        if self.label_type == 'value':
            labels = int(self.categories.index(category))
        elif self.label_type == 'one-hot':
            labels = torch.FloatTensor(np.array(self.categories) == category)
        else:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following: ' + 
                ', '.join(self.label_types))

        if self.transform:
            images = self.transform(images)
            
        sample = {
            'images': images, 
            'labels': labels, 
            'categories': category, 
            'location_ids': location_id, 
            'image_ids': selection['image_id'].to_list(),
            'timestamps': selection['timestamp'].to_list(),
        }
        return sample
    
def fmow_collate_fn(samples):
    batch = {}
    if isinstance(samples[0]['images'], torch.Tensor):
        batch['images'] = torch.stack([sample['images'] for sample in samples])
        
    # SimCLR
    elif isinstance(samples[0]['images'], list):
        batch['images'] = torch.stack([torch.stack(sample['images']) for sample in samples])
        batch['images'] = torch.split(batch['images'], 1, dim=1)
        batch['images'] = [torch.squeeze(sample, dim=1) for sample in batch['images']]
        
    label_elem = samples[0]['labels']
    if isinstance(label_elem, torch.Tensor):
        batch['labels'] = torch.stack([sample['labels'] for sample in samples])
    elif isinstance(label_elem, float):
        batch['labels'] = torch.Tensor([sample['labels'] for sample in samples], dtype=torch.float64)
    elif isinstance(label_elem, int):
        batch['labels'] = torch.Tensor([sample['labels'] for sample in samples]).long()
        
    batch['categories'] = [sample['categories'] for sample in samples]
    batch['location_ids'] = [sample['location_ids'] for sample in samples]
    batch['image_ids'] = [sample['image_ids'] for sample in samples]
    batch['timestamps'] = [sample['timestamps'] for sample in samples]
    return batch

class FMOWIndividualImageDataset(Dataset):
    categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", 
                  "aquaculture", "archaeological_site", "barn", "border_checkpoint", 
                  "burial_site", "car_dealership", "construction_site", "crop_field", 
                  "dam", "debris_or_rubble", "educational_institution", "electric_substation", 
                  "factory_or_powerplant", "fire_station", "flooded_road", "fountain", 
                  "gas_station", "golf_course", "ground_transportation_station", "helipad", 
                  "hospital", "impoverished_settlement", "interchange", "lake_or_pond", 
                  "lighthouse", "military_facility", "multi-unit_residential", 
                  "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", 
                  "parking_lot_or_garage", "place_of_worship", "police_station", "port", 
                  "prison", "race_track", "railway_bridge", "recreational_facility", 
                  "road_bridge", "runway", "shipyard", "shopping_mall", 
                  "single-unit_residential", "smokestack", "solar_farm", "space_facility", 
                  "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", 
                  "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", 
                  "wind_farm", "zoo"]
    label_types = ['value', 'one-hot']
    
    '''fMoW Dataset'''
    def __init__(self, 
                 csv_file, 
                 years=[*range(2000, 2021)], 
                 categories=None,
                 transform=None,
                 label_type='one-hot',
                 resize=64):
        """ Initialize the dataset.
        
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Base directory with wikidata folders and images.
            years (list, optional): List of years to take images from, None to not filter
            categories (list, optional): List of categories to take images from, None to not filter
            pre_transform (callable, optional): Optional transformation to be applied to individual images
                immediately after loading in. If None, each image in a time series is resized to
                to the dimensions of the first image. If specified, transformation should take images to 
                the same dimensions so they can be stacked together
            transform (callable, optional): Optional transform to be applied
                on tensor images.
            label_type (string): 'values' for single regression label, 'one-hot' for one hot labels
            resize: Size to load images as
        """
        self.df = pd.read_csv(csv_file) \
                .sort_values(['category', 'location_id', 'timestamp']) 
        
        # Filter by category
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]
        
        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]
        
        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform
        
        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:', 
                ', '.join(self.label_types))
        self.label_type = label_type
        
        self.resize = resize
        
    def __len__(self):
        return len(self.df)
    
    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read(
                out_shape=(data.count, self.resize, self.resize),
                resampling=Resampling.bilinear
            )
            
        return img
            
        
    def __getitem__(self, idx):
        ''' Gets timeseries info for images in one area
        
        Args: 
            idx: Index of loc in dataset
            
        Returns dictionary containing:
            'images': Images in tensor of dim Batch_SizexLengthxChannelxHeightxWidth
                Channel Order: R,G,B, NIR
            'labels': Labels of each image. Depends on 'label_type'
                'regression': First year is -0.5, second year is 0.5, so on. 
                'one-hot': Returns one-hot encoding of years
                'classification': for class labels by year where '0' is no construction (#labels = years)
            'years': Years of each image in timeseries
            'id': Id of image location
            'type': Type of image location
            'is_annotated': True if annotations for dates are provided
            'year_built': Year built as labeled in dataset
        '''
        selection = self.df.iloc[idx]

        image_paths = selection['image_path']
        
        #images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = torch.FloatTensor(self.open_image(selection['image_path'])) / 255
        
        labels = self.categories.index(selection['category'])

        if self.transform:
            images = self.transform(images)
            
        sample = {
            'images': images, 
            'labels': labels, 
            'image_ids': selection['image_id'],
            'timestamps': selection['timestamp']
        }
        return sample


class FMOWPairedDataset(Dataset):
    categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", 
                  "aquaculture", "archaeological_site", "barn", "border_checkpoint", 
                  "burial_site", "car_dealership", "construction_site", "crop_field", 
                  "dam", "debris_or_rubble", "educational_institution", "electric_substation", 
                  "factory_or_powerplant", "fire_station", "flooded_road", "fountain", 
                  "gas_station", "golf_course", "ground_transportation_station", "helipad", 
                  "hospital", "impoverished_settlement", "interchange", "lake_or_pond", 
                  "lighthouse", "military_facility", "multi-unit_residential", 
                  "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", 
                  "parking_lot_or_garage", "place_of_worship", "police_station", "port", 
                  "prison", "race_track", "railway_bridge", "recreational_facility", 
                  "road_bridge", "runway", "shipyard", "shopping_mall", 
                  "single-unit_residential", "smokestack", "solar_farm", "space_facility", 
                  "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", 
                  "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", 
                  "wind_farm", "zoo"]
    label_types = ['value', 'one-hot']
    
    '''fMoW Dataset'''
    def __init__(self, 
                 csv_file, 
                 years=[*range(2000, 2021)], 
                 categories=None,
                 pre_transform=None,
                 transform=None,
                 label_type='one-hot',
                 resize=64):
        """ Initialize the dataset.
        
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Base directory with wikidata folders and images.
            years (list, optional): List of years to take images from, None to not filter
            categories (list, optional): List of categories to take images from, None to not filter
            pre_transform (callable, optional): Optional transformation to be applied to individual images
                immediately after loading in. If None, each image in a time series is resized to
                to the dimensions of the first image. If specified, transformation should take images to 
                the same dimensions so they can be stacked together
            transform (callable, optional): Optional transform to be applied
                on tensor images.
            label_type (string): 'values' for single regression label, 'one-hot' for one hot labels
            resize: Size to load images as
        """
        self.df = pd.read_csv(csv_file) \
                .sort_values(['category', 'location_id', 'timestamp']) \
                .set_index(['category', 'location_id'])
        
        # Filter by category
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]
        
        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]
        
        self.indices = self.df.index.unique().to_numpy()
        
        self.pre_transform = pre_transform
        self.transform = transform
        
        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:', 
                ', '.join(self.label_types))
        self.label_type = label_type
        
        self.resize = resize
        
    def __len__(self):
        return len(self.indices)
    
    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read(
                out_shape=(data.count, self.resize, self.resize),
                resampling=Resampling.bilinear
            )
            
        return img
            
        
    def __getitem__(self, idx):
        ''' Gets timeseries info for images in one area
        
        Args: 
            idx: Index of loc in dataset
            
        Returns dictionary containing:
            'images': Images in tensor of dim Batch_SizexLengthxChannelxHeightxWidth
                Channel Order: R,G,B, NIR
            'labels': Labels of each image. Depends on 'label_type'
                'regression': First year is -0.5, second year is 0.5, so on. 
                'one-hot': Returns one-hot encoding of years
                'classification': for class labels by year where '0' is no construction (#labels = years)
            'years': Years of each image in timeseries
            'id': Id of image location
            'type': Type of image location
            'is_annotated': True if annotations for dates are provided
            'year_built': Year built as labeled in dataset
        '''
        index = self.indices[idx]
        selection = self.df.loc[index]

        image_paths = selection['image_path']
        
        #images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = [torch.FloatTensor(self.open_image(img_path)) / 255 for img_path in image_paths]

        if self.pre_transform:
            images = torch.cat([self.pre_transform(image.unsqueeze(0)) for image in images])
        else:
            shape = images[0].shape[-2:]
            resize = transforms.Resize(shape)
            resized = [resize(image.unsqueeze(0)) for image in images]
            images = torch.cat(resized)
        
#         category, location_id = index
        
#         if self.label_type == 'value':
#             labels = int(self.categories.index(category))
#         elif self.label_type == 'one-hot':
#             labels = torch.FloatTensor(np.array(self.categories) == category)
#         else:
#             raise ValueError(
#                 f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following: ' + 
#                 ', '.join(self.label_types))
        
        image1 = images
        image2 = images
        if self.transform:
            image1, shift1 = self.transform(image1)
            image2, shift2 = self.transform(image2)
            
#         sample = {
#             'image1': image1, 
#             'image2': image2, 
#             'labels': labels, 
#             'categories': category, 
#             'location_ids': location_id, 
#             'image_ids': selection['image_id'].to_list(),
#             'timestamps': selection['timestamp'].to_list(),
#         }
        return image1, image2, shift1, shift2


class FMOWTPDataset(Dataset):
    categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", 
                  "aquaculture", "archaeological_site", "barn", "border_checkpoint", 
                  "burial_site", "car_dealership", "construction_site", "crop_field", 
                  "dam", "debris_or_rubble", "educational_institution", "electric_substation", 
                  "factory_or_powerplant", "fire_station", "flooded_road", "fountain", 
                  "gas_station", "golf_course", "ground_transportation_station", "helipad", 
                  "hospital", "impoverished_settlement", "interchange", "lake_or_pond", 
                  "lighthouse", "military_facility", "multi-unit_residential", 
                  "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", 
                  "parking_lot_or_garage", "place_of_worship", "police_station", "port", 
                  "prison", "race_track", "railway_bridge", "recreational_facility", 
                  "road_bridge", "runway", "shipyard", "shopping_mall", 
                  "single-unit_residential", "smokestack", "solar_farm", "space_facility", 
                  "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", 
                  "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", 
                  "wind_farm", "zoo"]
    label_types = ['value', 'one-hot']
    
    '''fMoW Dataset'''
    def __init__(self, 
                 csv_file,
                 years=[*range(2000, 2021)], 
                 categories=None,
                 transform=None,
                 label_type='one-hot',
                 resize=64):
        """ Initialize the dataset.
        
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Base directory with wikidata folders and images.
            years (list, optional): List of years to take images from, None to not filter
            categories (list, optional): List of categories to take images from, None to not filter
            pre_transform (callable, optional): Optional transformation to be applied to individual images
                immediately after loading in. If None, each image in a time series is resized to
                to the dimensions of the first image. If specified, transformation should take images to 
                the same dimensions so they can be stacked together
            transform (callable, optional): Optional transform to be applied
                on tensor images.
            label_type (string): 'values' for single regression label, 'one-hot' for one hot labels
            resize: Size to load images as
        """
        self.df = pd.read_csv(csv_file) \
                .sort_values(['category', 'location_id', 'timestamp']) 
        
        # Filter by category
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]
        
        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]
        
        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform
        
        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:', 
                ', '.join(self.label_types))
        self.label_type = label_type
        
        self.resize = resize
        
    def __len__(self):
        return len(self.df)
    
    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read(
                out_shape=(data.count, self.resize, self.resize),
                resampling=Resampling.bilinear
            )
            
        return img
            
        
    def __getitem__(self, idx):
        ''' Gets timeseries info for images in one area
        
        Args: 
            idx: Index of loc in dataset
            
        Returns dictionary containing:
            'images': Images in tensor of dim Batch_SizexLengthxChannelxHeightxWidth
                Channel Order: R,G,B, NIR
            'labels': Labels of each image. Depends on 'label_type'
                'regression': First year is -0.5, second year is 0.5, so on. 
                'one-hot': Returns one-hot encoding of years
                'classification': for class labels by year where '0' is no construction (#labels = years)
            'years': Years of each image in timeseries
            'id': Id of image location
            'type': Type of image location
            'is_annotated': True if annotations for dates are provided
            'year_built': Year built as labeled in dataset
        '''
        selection = self.df.iloc[idx]

        image_path = selection['image_path'].replace(".jpg", "_crop_0.jpg")
        
        #images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        image = torch.FloatTensor(self.open_image(image_path)) / 255
        
        t1 = selection['timestamp']
        t1 = datetime.fromisoformat(t1[:-1])
        
#         labels = self.categories.index(selection['category'])
        
        split = image_path.rsplit('/', 1)
        base_path = split[0]
        fname = split[1]
#         print(split, base_path, fname)
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit('_', 1)
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
#         print(image_path, regexp)
        temporal_files = glob(regexp)
        temporal_files.remove(image_path)
#         print(temporal_files)
        if temporal_files == []:
            single_image_name_2 = image_path
        else:
            single_image_name_2 = random.choice(temporal_files)
        image2 = torch.FloatTensor(self.open_image(single_image_name_2)) / 255
        
        json_file2 = os.path.join(base_path, single_image_name_2.replace("_crop_0.jpg", ".json"))
        t2 = json.load(open(json_file2, "r"))["timestamp"]
        t2 = datetime.fromisoformat(t2[:-1])
        
        shift1 = torch.FloatTensor([(t2-t1).days])/365
        shift2 = torch.FloatTensor([(t1-t2).days])/365
        

        if self.transform:
            image = self.transform(image)
            image2 = self.transform(image2)
            
#         sample = {
#             'images': images, 
#             'labels': labels, 
#             'image_ids': selection['image_id'],
#             'timestamps': selection['timestamp']
#         }
        return image, image2, shift1, shift2

class CustomDatasetFromImagesTemporal(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
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
        single_image_name_1 = self.image_arr[index]

        splt = single_image_name_1.rsplit('/', 1)
        base_path = splt[0]
        fname = splt[1]
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit('_', 1)
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
        temporal_files = glob(regexp)
        temporal_files.remove(single_image_name_1)
#         print(temporal_files)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
        else:
            single_image_name_2 = random.choice(temporal_files)

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_tensor_1 = self.transforms(img_as_img_1)

        img_as_img_2 = Image.open(single_image_name_2)
        img_as_tensor_2 = self.transforms(img_as_img_2)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return ([img_as_tensor_1, img_as_tensor_2], single_image_label)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    dataset_params = {
        'csv_file': f"/atlas/u/pliu1/housing_event_pred/data/fmow-train_unsupervised.csv", # TODO specify split
        'years':[*range(2015, 2021)], 
        'categories': None,
        'transform': None,
        'label_type': 'value',
        'resize': 64
    }
    train_dataset = FMOWTPDataset(**dataset_params)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,shuffle=True, num_workers=16, drop_last=True)
    
#     start_time = time.time()
#     for i, data in enumerate(train_dataloader, 0):
#         image1, image2, shift1, shift2 = data
#         print(i, time.time()-start_time())
#         if i % 100 == 0:
#             print(image1.shape, image2.shape, shift1.shape, shift2.shape)
    
#     train_csv = "/atlas/u/buzkent/patchdrop/data/fMoW/train_62classes.csv"
#     aug = [ transforms.Resize(64),
#             transforms.CenterCrop(64),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                         std=(0.229, 0.224, 0.225))
#           ]
#     transform = transforms.Compose(aug)
#     train_dataset = CustomDatasetFromImagesTemporal(train_csv, transform)
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,shuffle=True, num_workers=16, drop_last=True)
    
    start_time = time.time()
    for i, data in enumerate(train_dataloader, 0):
#         images, label = data
        print(i, time.time()-start_time)
       