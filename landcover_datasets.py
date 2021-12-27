from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import glob
import os
import numpy as np
import pandas as pd
from landcover_data_utils import clip_and_scale_image


class FactbookDataset(Dataset):
    def __init__(self, csv_fn, dists_fn, idxs=None, neighbors=None,
                 neighborhood=1000, test_features=None, single=False):
        """
        csv_fn: Path of dataset csv
        dists_fn: Path of distances npy (242 x 242)
        idxs: Indices of countries to use of length N
        neighbors: Strategy to use for getting neighbors ['uniform', 'closest', 'inverse']
        neighborhood: Size of neighborhood for sampling neighboring country
        test_features: List of feature idxs NOT to use for training
        single: If True, return only anchor patch
        """
        self.df = pd.read_csv(csv_fn)
        self.distances = np.load(dists_fn) # 242x242
        if idxs: self.idxs = idxs
        else: self.idxs = np.arange(self.df.shape[0])
        if neighbors: self.neighbors = neighbors
        else: self.neighbors = 'closest'
        assert self.neighbors in ['closest', 'uniform', 'inverse']
        self.neighborhood = neighborhood
        if test_features: self.test_features = test_features
        # else: self.test_features = [6, 40, 34, 76, 19, 44, 27, 16, 63, 25, 35, 38, 41, 57, 53] # original set (picked by Neal and Sherrie)
        else: self.test_features = [5, 25, 27, 34, 35, 38, 40, 41, 43, 47, 57, 74, 76, 78] #health-related with computed health index (78)
        self.single = single
        # Remove test features
        data_raw = self.df.values
        self.test_data = data_raw[:,self.test_features].astype(np.float)
        data = data_raw[:,[i for i in range(data_raw.shape[1]) if i not in self.test_features]]
        self.data = data[:,5:].astype(np.float)
        self.n_features = self.data.shape[1]
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        # Get anchor country feature vector
        a = self.data[idx]
        if self.single:
            return torch.FloatTensor(a)
        else:
            # Sample neighbor/distant country feature vector
            n_idx, d_idx = self.sample_neighbor(idx)
            n, d = self.data[n_idx], self.data[d_idx]
            # Sample random distant country
            # options = list(self.idxs)
            # _ = options.remove(idx)
            # d_idx = np.random.choice(options)
            # d = self.data[d_idx]
            a, n, d = torch.FloatTensor(a), torch.FloatTensor(n), torch.FloatTensor(d)
            sample = {'anchor': a, 'neighbor': n, 'distant': d}
            return sample
    
    def sample_neighbor(self, a_idx):
        """Samples a neighbor idx using sampling strategy."""
        a_vec = self.distances[a_idx].copy()
        if self.neighbors == 'closest': # get the closest country
            n = a_vec.argsort()[:2] # get indices of 2 smallest elements
            n_idx = n[1] # smallest is itself (d=0)
            d_idxs = list(self.idxs)
            d_idxs.remove(a_idx)
            d_idxs.remove(n_idx)
        elif self.neighbors == 'uniform': # sample uniformly within neighborhood
            neighbors = []
            nh = self.neighborhood
            while not neighbors:
                a_vec_temp = a_vec.copy()
                a_vec_temp[a_vec_temp > nh] = 0 # don't choose places outside neighborhood
                neighbors = list(np.nonzero(a_vec_temp)[0])
                nh *= 2 # double neighborhood
            n_idx = np.random.choice(neighbors)
            d_idxs = list(self.idxs)
            d_idxs.remove(a_idx)
            for neighbor in neighbors:
                d_idxs.remove(neighbor)
        elif self.neighbors == 'inverse': # sample with probability inverse to distance
            a_vec[a_idx] = 1 # so we don't divide by zero
            probs = np.ones_like(a_vec) / a_vec # probabilities for ALL countries
            probs[a_idx] = 0 # don't select yourself
            probs = probs[self.idxs] # get probs for countries in training set
            probs /= probs.sum()
            n_idx = np.random.choice(self.idxs, p=probs)
            d_idx = list(self.idxs) # THIS IS WRONG, WILL FIX LATER
        # Choose d_idx
        d_idx = np.random.choice(d_idxs)
        return n_idx, d_idx


class PatchTripletsDataset(Dataset):
    
    def __init__(self, patch_dir, transform=None, n_triplets=None,
        pairs_only=True, tile_size=50, idx_offset=None):
        self.patch_dir = patch_dir
        self.anchor_files = glob.glob(os.path.join(self.patch_dir, '*'))
        self.transform = transform
        self.n_triplets = n_triplets
        self.pairs_only = pairs_only
        self.tile_size = tile_size
        self.idx_offset = idx_offset

    def __len__(self):
        if self.n_triplets: return self.n_triplets
        else: return len(self.anchor_files) // 3
    
    def __getitem__(self, idx):
        if self.idx_offset: idx += self.idx_offset
        p = np.load(os.path.join(self.patch_dir, '{}anchor.npy'.format(idx)))
        n = np.load(os.path.join(self.patch_dir, '{}neighbor.npy'.format(idx)))
        if self.pairs_only:
            name = np.random.choice(['anchor', 'neighbor', 'distant'])
            d_idx = np.random.randint(0, self.n_triplets)
            d = np.load(os.path.join(self.patch_dir,
                '{}{}.npy'.format(d_idx, name)))
        else:
            d = np.load(os.path.join(self.patch_dir,
                '{}distant.npy'.format(idx)))
        p = np.moveaxis(p, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        
        full_size = p.shape[1]
        assert full_size >= self.tile_size, 'tile size cannot be greater than image size'
        full_center = full_size // 2
        tile_radius = self.tile_size // 2
        p = self.get_center(p, full_center, tile_radius)
        n = self.get_center(n, full_center, tile_radius)
        d = self.get_center(d, full_center, tile_radius)
        sample = {'anchor': p, 'neighbor': n, 'distant': d}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_center(self, p, fc, tr):
        """Gets the center tile from a larger tile."""
        p = p[:, fc-tr:fc+self.tile_size-tr,fc-tr:fc+self.tile_size-tr]
        return p


class PatchTripletsAsPatchDataset(Dataset):

    def __init__(self, patch_dir, anchor_idxs, tile_size=50, transform=None):
        self.patch_dir = patch_dir
        self.anchor_idxs = anchor_idxs
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return 3 * len(self.anchor_idxs)

    def __getitem__(self, idx):
        anchor_types = ['anchor', 'neighbor', 'distant']
        anchor_type = anchor_types[idx % 3]
        p_idx = self.anchor_idxs[idx // 3]
        p = np.load(os.path.join(self.patch_dir, '{}{}.npy'.format(
            p_idx, anchor_type)))
        p = np.moveaxis(p, -1, 0)
        full_size = p.shape[1]
        assert full_size >= self.tile_size, 'tile size cannot be greater than image size'
        full_center = full_size // 2
        tile_radius = self.tile_size // 2
        p = self.get_center(p, full_center, tile_radius)
        if self.transform:
            p = self.transform(p)
        return p

    def get_center(self, p, fc, tr):
        """Gets the center tile from a larger tile."""
        p = p[:, fc-tr:fc+self.tile_size-tr,fc-tr:fc+self.tile_size-tr]
        return p


class PatchDataset(Dataset):

    def __init__(self, tile_dir, tile_idxs, labels=None, transform=None, pad_data=False):
        self.tile_dir = tile_dir
        self.tile_idxs = tile_idxs
        self.labels = labels
        self.transform = transform
        self.pad_data = pad_data

    def __len__(self):
        return len(self.tile_idxs)

    def __getitem__(self, idx):
        p_idx = self.tile_idxs[idx]
        p = np.load(os.path.join(self.tile_dir, '{}tile.npy'.format(p_idx)))
        p = p[:, :, :3]
        p = np.moveaxis(p, -1, 0)
        y = self.labels[p_idx]
        if self.transform:
            p = self.transform(p)
        if self.pad_data:
            padding = torch.zeros(13, p.shape[1], p.shape[2])
            p = torch.cat([padding, p], dim=0)
        return p, y

class UCMercedDataset(Dataset):
    
    def __init__(self, img_dir, class_names, transform=None, n_triplets=20000, pairs_only=True, tile_size=100):
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform
        self.pairs_only = pairs_only
        self.tile_size = tile_size
        self.n_triplets = n_triplets
        print("tile size initiated to: {}".format(tile_size))

    def __len__(self):
        return self.n_triplets
    
    def __getitem__(self, idx):
        # sample a random class and random integer between 00 and 99
        img_class_anchor = np.random.choice(self.class_names)
        img_num = np.random.randint(0,100)
        img_fn = os.path.join(self.img_dir, img_class_anchor + '{:02}.npy'.format(img_num))
        img = np.load(img_fn)
        (w, h, d) = img.shape
        # print("Image shape: {}".format(img.shape))
        max_x = w - self.tile_size
        max_y = h - self.tile_size
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        a = np.array(img[x:x+self.tile_size,y:y+self.tile_size])

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        n = np.array(img[x:x+self.tile_size,y:y+self.tile_size])

        # sample a random class and random integer between 00 and 99
        img_class_distant = np.random.choice(self.class_names)
        img_num = np.random.randint(0,100)
        img_fn = os.path.join(self.img_dir, img_class_distant + '{:02}.npy'.format(img_num))
        img = np.load(img_fn)
        (w, h, d) = img.shape
        # print("Image shape: {}".format(img.shape))
        max_x = w - self.tile_size
        max_y = h - self.tile_size
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        d = np.array(img[x:x+self.tile_size,y:y+self.tile_size])

        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        # print("Anchor shape: {}, neighbor shape: {}, distant shape: {}".format(a.shape, n.shape, d.shape))
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        if self.transform:
            sample = self.transform(sample)
        return sample


### TRANSFORMS ###

class GetBands(object):
    """
    Gets the first X bands of the anchor triplet.
    """
    def __init__(self, bands):
        assert bands >= 0, 'Must get at least 1 band'
        self.bands = bands

    def __call__(self, sample):
        p, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Patches are already in [c, w, h] order
        p, n, d = (p[:self.bands,:,:], n[:self.bands,:,:], d[:self.bands,:,:])
        sample = {'anchor': p, 'neighbor': n, 'distant': d}
        return sample


class GetBandsSinglePatch(object):
    """
    Does what GetBands does except for one anchor.
    """
    def __init__(self, bands):
        assert bands >= 0, 'Must get at least 1 band'
        self.bands = bands

    def __call__(self, p):
        p = p[:self.bands,:,:]
        return p


class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, sample):
        p, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Randomly horizontal flip
        if np.random.rand() < 0.5: p = np.flip(p, axis=2).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=2).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5: p = np.flip(p, axis=1).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=1).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: p = np.rot90(p, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: n = np.rot90(n, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: d = np.rot90(d, k=rotations, axes=(1,2)).copy()
        sample = {'anchor': p, 'neighbor': n, 'distant': d}
        return sample


class RandomFlipAndRotateSinglePatch(object):
    """
    Does what RandomFlipAndRotate except for one anchor.
    """
    def __call__(self, p):
        # Randomly horizontal and vertical flip
        if np.random.rand() < 0.5: p = np.flip(p, axis=2).copy()
        if np.random.rand() < 0.5: p = np.flip(p, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: p = np.rot90(p, k=rotations, axes=(1,2)).copy()
        return p


class ClipAndScale(object):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, sample):
        p, n, d = (clip_and_scale_image(sample['anchor'], self.img_type),
                   clip_and_scale_image(sample['neighbor'], self.img_type),
                   clip_and_scale_image(sample['distant'], self.img_type))
        sample = {'anchor': p, 'neighbor': n, 'distant': d}
        return sample


class ClipAndScaleSinglePatch(object):
    """
    Does what ClipAndScale does except for one anchor.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, p):
        p = clip_and_scale_image(p, self.img_type)
        return p


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        p, n, d = (torch.from_numpy(sample['anchor']).float(),
            torch.from_numpy(sample['neighbor']).float(),
            torch.from_numpy(sample['distant']).float())
        sample = {'anchor': p, 'neighbor': n, 'distant': d}
        return sample


class ToFloatTensorSinglePatch(object):
    """
    Does what ToFloatTensor does except for one anchor.
    """
    def __call__(self, p):
        p = torch.from_numpy(p).float()
        return p


class Normalize(object):
    """
    Normalizes each anchor triplet using the provided channel means and standard
    deviations for that dataset. The param normalize is a list with two
    elements: the first is a tuple containing the channel means, the second is
    a tuple containing the channel stds.
    """
    def __init__(self, normalize):
        self.normalize = normalize
        self.channels = len(self.normalize[0])

    def __call__(self, sample):
        p, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        for i in range(self.channels):
            p[:,:,i] = (p[:,:,i] - self.normalize[0][i]) / self.normalize[1][i]
            n[:,:,i] = (n[:,:,i] - self.normalize[0][i]) / self.normalize[1][i]
            d[:,:,i] = (d[:,:,i] - self.normalize[0][i]) / self.normalize[1][i]
        sample = {'anchor': p, 'neighbor': n, 'distant': d}
        return sample

### TRANSFORMS ###


def triplet_dataloader(img_type, patch_dir, bands=4, augment=True,
    batch_size=4, shuffle=True, num_workers=4, n_triplets=None, tile_size=50,
    pairs_only=True, normalize=None):
    """
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat anchores.
    Turn shuffle to False for producing embeddings that correspond to original
    anchores.
    """
    assert img_type in ['landsat', 'rgb', 'naip']
    transform_list = []
    if img_type in ['landsat', 'naip']: transform_list.append(GetBands(bands))
    transform_list.append(ClipAndScale(img_type))
    if normalize: transform_list.append(Normalize(normalize))
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = PatchTripletsDataset(patch_dir, transform=transform,
        n_triplets=n_triplets, pairs_only=pairs_only, tile_size=tile_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    return dataloader

def UCMerced_dataloader(img_type, img_dir, class_names, bands=3, augment=True,
    batch_size=4, shuffle=True, num_workers=4, n_triplets=None, tile_size=100,
    pairs_only=True, normalize=None):
    """
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat anchores.
    Turn shuffle to False for producing embeddings that correspond to original
    anchores.
    """
    assert img_type in ['landsat', 'rgb', 'naip']
    transform_list = []
    if img_type in ['landsat', 'naip']: transform_list.append(GetBands(bands))
    transform_list.append(ClipAndScale(img_type))
    if normalize: transform_list.append(Normalize(normalize))
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = UCMercedDataset(img_dir, class_names, transform=transform,
        n_triplets=n_triplets, pairs_only=pairs_only, tile_size=tile_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder
    splits_fn = '/atlas/u/kayush/pix2vec/splits.npy'
    y_fn = '/atlas/u/kayush/pix2vec/y_50_100.npy'
    splits = np.load(splits_fn)
    idxs_tr = np.where(splits == 0)[0]
    y = np.load(y_fn)
    le = LabelEncoder()
    le.fit(y)
    labels = le.transform(y)
    dataset = PatchDataset('/atlas/u/kayush/pix2vec/supervised_50_100/', idxs_tr, labels, ToFloatTensorSinglePatch(), True)
    print(dataset[100][0].shape)