from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import pickle
from torch.utils.data import DataLoader
import glob
from skimage import img_as_float

class XRayDatasetNew(Dataset):
    def __init__(self, datasource, transforms=[], loadlandmarks=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasource = datasource
        self.transforms = transforms
        self.loadlandmarksflag = loadlandmarks

    def __len__(self):
        return len(self.datasource["images"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        image = cv2.imread(self.datasource["images"][idx], cv2.IMREAD_GRAYSCALE)
        label = np.load(self.datasource["labels"][idx])
        sample = {'image': image, "label":label}
        
        if self.loadlandmarksflag:
            landmarks = np.loadtxt(self.datasource["landmarks"][idx]).astype(np.float32)
            sample["landmarks"] = landmarks
        
        for transform in self.transforms:
            sample = transform(sample)
        return sample

class XRayDataset(Dataset):
    def __init__(self, datasource, transforms=[], short=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasource = datasource
        self.transforms = transforms
        self.short = short

    def __len__(self):
        return len(self.datasource["images"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.datasource["images"][idx]
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        #image = image.astype(np.float32)
        landmarks = np.loadtxt(self.datasource["landmarks"][idx])
        landmarks = landmarks.astype(np.float32)
        if self.short is True:
            landmarks = landmarks[0::2]
        label = np.load(self.datasource["labels"][idx])
        sample = {'image': image, 'landmarks': landmarks, "label":label}

        for transform in self.transforms:
            sample = transform(sample)

        return sample
def get_xray_dataloader(cross_validation_set, transforms=[], batch_size=1, short=False):
    #cross_validation_set = pickle.load(fp)
    
    trainingset  = XRayDataset(cross_validation_set["trainingset"], transforms=transforms, short=short)
    validationset = XRayDataset(cross_validation_set["testingset"], transforms=transforms, short=short)
    testingset = XRayDataset(cross_validation_set["unseenset"], transforms=transforms, short=short)
    
    trainloader = DataLoader(trainingset, batch_size=batch_size, shuffle=True, num_workers=0)
    validloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testingset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return {"trainloader":trainloader, "validloader":validloader, "testloader":testloader}

def get_new_xray_dataloader(cross_validation_set, transforms=[], batch_size=1):
    #cross_validation_set = pickle.load(fp)
    
    trainingset  = XRayDatasetNew(cross_validation_set["trainingset"], transforms=transforms)
    validationset = XRayDatasetNew(cross_validation_set["validationset"], transforms=transforms)
    testingset = XRayDatasetNew(cross_validation_set["testingset"], transforms=transforms, loadlandmarks=True)
    
    trainloader = DataLoader(trainingset, batch_size=batch_size, shuffle=True, num_workers=0)
    validloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testingset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return {"trainloader":trainloader, "validloader":validloader, "testloader":testloader}