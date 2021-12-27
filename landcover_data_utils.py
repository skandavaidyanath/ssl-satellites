import numpy as np
import os
# import gdal
# import imageio
# import cv2
import urllib
from time import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from math import radians, cos, sin, asin, sqrt


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in km.
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
    

def load_all_triplet_patches(patch_dir, n_triplets, bands=None,
    clip_and_scale=False, img_type='naip', tile_size=50, print_every=None):
    """
    Loads all triplet patches saved as numpy arrays in the given patch
    directory. Patches must be indexed from 0 through (n_triplets-1). The
    patches are returned as a list of numpy array. For NAIP and Landsat, set
    bands equal to the number of channels to keep (from 0 through (bands-1)).
    """
    patches = []
    t0 = time()
    tile_radius = tile_size // 2
    for i in range(n_triplets):
        p = np.load(os.path.join(patch_dir, '{}anchor.npy'.format(i)))
        n = np.load(os.path.join(patch_dir, '{}neighbor.npy'.format(i)))
        d = np.load(os.path.join(patch_dir, '{}distant.npy'.format(i)))
        
        full_size = p.shape[0]
        full_center = full_size // 2
        
        p = p[full_center-tile_radius:full_center+tile_size-tile_radius,
              full_center-tile_radius:full_center+tile_size-tile_radius,:]
        n = n[full_center-tile_radius:full_center+tile_size-tile_radius,
              full_center-tile_radius:full_center+tile_size-tile_radius,:]
        d = d[full_center-tile_radius:full_center+tile_size-tile_radius,
              full_center-tile_radius:full_center+tile_size-tile_radius,:]
        if bands: p, n, d = p[:,:,:bands], n[:,:,:bands], d[:,:,:bands]
        if clip_and_scale:
            p, n, d = (clip_and_scale_image(p, img_type=img_type),
                       clip_and_scale_image(n, img_type=img_type),
                       clip_and_scale_image(d, img_type=img_type))
        patches += [p, n, d]
        if print_every is not None and (i + 1) % print_every == 0:
            print('Loaded {}/{} triplets: {:0.3f}s'.format(
                i+1, n_triplets, time()-t0))
    return patches


def compute_pca_on_patches(patches, n_components=2, scale=False):
    """
    Computes PCA embeddings for a list of patches after vectorizing. If scale
    is true, then also scales the patches with StandardScaler before PCA.
    """
    patch_array = np.zeros((len(patches), np.product(patches[0].shape)))
    for idx, patch in enumerate(patches):
        patch_array[idx,:] = patch.ravel()
    if scale:
        patch_array = StandardScaler().fit_transform(patch_array)
    pca_embeddings = PCA(n_components).fit_transform(patch_array)
    return pca_embeddings


def get_bands(img, band0=3, band1=2, band2=1, gamma=2):
    """
    Gets three bands from satellite image for visualization. If sqrt is True,
    then takes the sqrt of each value for better visualization - this should
    be turned on for Landsat imagery.
    """
    X = img[:,:,[band0, band1, band2]]
    X = X ** (1 / gamma)
    # if sqrt:
    #     X = X ** 0.5
    # else:
    #     gammas = [1.8, 1.8, 1.8]
    #     for idx, gamma in enumerate(gammas):
    #         X[:,:,idx] = X[:,:,idx] ** (1 / gamma) 
    return X


def load_satellite_image(img_fn, bands_only=True, clip_and_scale=True):
    """
    Loads either RGB or Landsat satellite image by checking file type.
    """
    _, ext = os.path.splitext(img_fn)
    assert ext in ['.jpg', '.jpeg', '.png', '.tif'], 'Unknown file type'
    if ext == '.tif':
        return load_landsat(img_fn, bands_only, clip_and_scale)
    else:
        return imageio.imread(img_fn)


def clip_and_scale_image(img, img_type='naip', clip_min=0, clip_max=10000):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    if img_type in ['naip', 'rgb']:
        return img / 255
    elif img_type == 'landsat':
        return np.clip(img, clip_min, clip_max) / (clip_max - clip_min)



def load_landsat(img_fn, bands_only=True, clip_and_scale=True):
    """
    Loads Landsat image with gdal, rearranges axes so channels are third,
    returns image as array.
    """
    obj = gdal.Open(img_fn)
    img = obj.ReadAsArray().astype(np.float64)
    del obj # close GDAL dataset
    img = np.moveaxis(img, 0, -1)
    if bands_only: img = img[:,:,:7]
    if clip_and_scale: img = clip_and_scale_image(img, 'landsat')
    return img


def visualize_satellite_image(img, band0=3, band1=2, band2=1, pix_per_inch=200,
                              title=True, noaxes=False, save_fn=None,
                              save_only=False):
    """
    Visualizes three specified bands from Landsat image array.
    """
    w, h, c = img.shape
    assert (w >= pix_per_inch) and (h >= pix_per_inch)
    if c == 3:
        X = img
    else:
        X = get_bands(img, band0, band1, band2, sqrt=True)
    if save_fn is not None:
        plt.imsave(save_fn, X)
    if not save_only:
        plt.figure(figsize=(h // pix_per_inch, w // pix_per_inch))
        plt.imshow(X)
        if title: plt.title('{} rows by {} cols'.format(w, h))
        if noaxes: plt.axis('off')
        plt.show()


def extract_patch(img_padded, x0, y0, patch_radius):
    """
    Extracts a patch from a (padded) image given the row and column of
    the center pixel and the patch radius. E.g., if the patch
    size is 15 pixels per side, then the patch radius should be 7.
    """
    w, h, c = img_padded.shape
    row_min = x0 - patch_radius
    row_max = x0 + patch_radius
    col_min = y0 - patch_radius
    col_max = y0 + patch_radius
    assert row_min >= 0, 'Row min: {}'.format(row_min)
    assert row_max <= w, 'Row max: {}'.format(row_max)
    assert col_min >= 0, 'Col min: {}'.format(col_min)
    assert col_max <= h, 'Col max: {}'.format(col_max)
    patch = img_padded[row_min:row_max+1, col_min:col_max+1, :]
    return patch


def sample_patch_triplet(img_padded, xp, yp, patch_radius=7, neighborhood=50):
    """
    Samples a patch triplet given a padded image, the center pixel of
    the main patch in the padded image, the patch radius, and the size
    of the neighborhood.
    """
    w_pad, h_pad, c = img_padded.shape
    w = w_pad - 2 * patch_radius
    h = h_pad - 2 * patch_radius
    patch = extract_patch(img_padded, xp, yp, patch_radius)
    # Get neighbor patch
    xn = np.random.randint(max(xp-neighborhood, patch_radius),
        min(xp+neighborhood, w+patch_radius))
    yn = np.random.randint(max(yp-neighborhood, patch_radius),
        min(yp+neighborhood, h+patch_radius))
    patch_neighbor = extract_patch(img_padded, xn, yn, patch_radius)
    # Get distant patch
    xd, yd = xp, yp
    while (xd >= xp - neighborhood) and (xd <= xp + neighborhood):
        xd = np.random.randint(0, w) + patch_radius
    while (yd >= yp - neighborhood) and (yd <= yp + neighborhood):
        yd = np.random.randint(0, h) + patch_radius
    patch_distant = extract_patch(img_padded, xd, yd, patch_radius)
    return (patch, patch_neighbor, patch_distant)


def save_patch_triplets(patch_dir, img, n_triplets=1000, patch_size=31,
    neighborhood=100, print_every=None):
    """
    Saves patch triplets as numpy arrays with naming convention:
    - 0patch.npy, 0neighbor.npy, 0distant.npy
    """
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    assert patch_size % 2 == 1
    patch_radius = patch_size // 2
    w, h, c = img.shape
    img_padded = np.pad(img, pad_width=[(patch_radius, patch_radius),
                                        (patch_radius, patch_radius), (0,0)],
                        mode='reflect')
    t0 = time()
    for i in range(n_triplets):
        xp = np.random.randint(0, w) + patch_radius
        yp = np.random.randint(0, h) + patch_radius
        patch, patch_neighbor, patch_distant = sample_patch_triplet(
            img_padded, xp, yp, patch_radius, neighborhood)
        np.save(os.path.join(patch_dir, '{}anchor.npy'.format(i)), patch)
        np.save(os.path.join(patch_dir, '{}neighbor.npy'.format(i)),
            patch_neighbor)
        np.save(os.path.join(patch_dir, '{}distant.npy'.format(i)),
            patch_distant)
        if print_every is not None and ((i+1) % print_every == 0):
            print('Saved {} triplets successfully in {:0.3f}s'.format(
                i+1, time()-t0))


def sample_idx_by_cdl_class(y, cdl_class, class_labels):
    """
    Samples an index uniformly at random for the given CDL class.
    The class can be given as a string or an integer.
    class_labels: A dictionary mapping class names (str) to
    class labels (int)
    """
    if isinstance(cdl_class, str):
        cdl_class = class_labels[cdl_class]
    idx = np.random.choice(np.argwhere(y == cdl_class).squeeze())
    return idx


def get_analogy_nn(idx1, idx2, idx3, embeddings):
    """
    Returns the index of the nearest neighbor to: (z1 - z2) + z3.
    """
    z1, z2, z3 = embeddings[[idx1, idx2, idx3]]
    z4 = (z1 - z2) + z3
    distances = np.linalg.norm(embeddings - z4, axis=1)
    return np.argmin(distances)


def get_k_neighbors(idx, embeddings, k=25):
    """
    Gets the k nearest neighbors in the provided embedding.
    """
    z = embeddings[idx]
    distances = np.linalg.norm(embeddings - z, axis=1)
    ordering = np.argsort(distances)
    topk_idxs = ordering[:k]
    topk_dists = distances[topk_idxs]
    return (topk_idxs, topk_dists)


def get_k_neighbors_with_z(z, embeddings, k=25):
    """
    Gets the k nearest neighbors in the provided embedding.
    """
    distances = np.linalg.norm(embeddings - z, axis=1)
    ordering = np.argsort(distances)
    topk_idxs = ordering[:k]
    topk_dists = distances[topk_idxs]
    return (topk_idxs, topk_dists)


def get_majority_label_and_fraction(full_patch, threshold=0.8):
    """
    Gets the majority label and fraction of majority class for
    NAIP iamge patch.
    """
    cdl_labels = full_patch[:,:,4]
    counts = Counter(cdl_labels.ravel())
    mode, n_mode = counts.most_common(1)[0]
    p_mode = n_mode / len(cdl_labels.ravel())
    if not np.isnan(mode) and p_mode >= threshold:
        return mode
    else:
        return np.nan


def compute_cosine_similarity(x, y):
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def batch_compute_cosine_similarity(X, Y):
    """
    Computes the cosine similarity between two arrays, where the similarity is
    computed for corresponding rows of X and Y. For example, if X and Y both
    have 100 rows, this function returns a (100,) vector of similarities.
    """
    similarities = np.zeros((X.shape[0],))
    for i in range(X.shape[0]):
        similarities[i] = compute_cosine_similarity(X[i], Y[i])
    return similarities


def sample_image(out_fn, lat, lon, height=400, width=400, zoom=19):
    """
    This function uses the Google Static Maps API to download and save
    one satellite image.
    :param out_fn: Output filename for saved image
    :param lat: Latitude of image center
    :param lon: Longitude of image center
    :param height: Height of image in pixels
    :param width: Width of image in pixels
    :param zoom: Zoom level of image
    :return: True if valid image saved, False if no image saved
    """
    # Google Static Maps API key
    keys = ["AIzaSyDzWmvcVKeyoDgyvnLgHpVnCLpypWLrkPY",
        "AIzaSyBW1imRBOtaa6uwvVRxjlaZ38U0rkjbYzM",
        "AIzaSyAejgapvGncLMRiMlUoqZ2h6yRF-lwNYMM",
        "AIzaSyD3f6sozQ3UqlP45Oj_8plnc5KILYC9amU",
        "AIzaSyAZlsqwvlFBArKqMCbPfE_BMVL_KU6dKSM",
        "AIzaSyCkRXcAdW-rwh7cpqOOrJBw2agkeftdcDc",
        "AIzaSyCzbSnF_OH2sElz1Mh2xuU0XLaF0oI3DxE"]
    # Pick key at random
    key = np.random.randint(0,6)
    api_key = keys[key]
    
    try:
        # Save extra tall satellite image
        height_buffer = 100
        url_pattern = 'https://maps.googleapis.com/maps/api/staticmap?center=%0.6f,%0.6f&zoom=%s&size=%sx%s&maptype=satellite&key=%s'
        url = url_pattern % (lat, lon, zoom, width, height + height_buffer, api_key)
        urllib.request.urlretrieve(url, out_fn)

        # Cut out text at the bottom of the image
        image = cv2.imread(out_fn)
        image = image[int(height_buffer/2):int(height+height_buffer/2),:,:]
        image
        cv2.imwrite(out_fn, image)

        # Check file size and delete invalid images < 10kb
        fs = os.stat(out_fn).st_size
        if fs < 10000:
            print('Invalid image')
            os.remove(out_fn)
            return None
        else:
            # Return RGB image instead of BGR
            return image[:,:,::-1]

    except:
        return None


