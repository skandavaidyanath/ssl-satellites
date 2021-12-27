import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    """
    Spacenet dataset
    Data Root Path: /atlas/u/kayush/winter2020/jigsaw/semseg/dataset/spacenet/list/
    """
    def __init__(self, split='train', data_root=None, data_list=None, pad=False, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.pad = pad
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label[label==255] = 1     ## Ayush added this to make sure the ignore_label is disabled
        if self.pad:
            ## Pad the images to 16 bands instead of 3 bands
            padding = np.zeros((image.shape[0], image.shape[1], 13))
            image = np.concatenate([padding, image], axis=-1)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
    
    
if __name__ == '__main__':
    dataset = SemData(split='train', data_root='/atlas/u/kayush/winter2020/jigsaw/semseg/dataset/spacenet/list/', 
                      data_list='/atlas/u/kayush/winter2020/jigsaw/semseg/dataset/spacenet/list/training.txt', 
                      transform=transforms.ToTensor())
    print(len(dataset))
    