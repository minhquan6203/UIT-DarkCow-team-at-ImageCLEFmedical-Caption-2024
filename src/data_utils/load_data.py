from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional
import pandas as pd
import os
import numpy as np
import re
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class Captioning_Dataset(Dataset):
    def __init__(self, label_path, img_folder, with_labels=True):
        self.annotations = self.load_annotations(label_path,img_folder ,with_labels)


    def load_annotations(self,label_path, img_folder, with_labels):
        data=pd.read_csv(label_path)
        annotations = []
        if with_labels:
            for idx, row in data.iterrows():
                cap=row['Caption']
                id=row['ID']
                annotation = {
                            "id": id,
                            "caption": cap.lower(),
                            "image": os.path.join(img_folder,id),
                        }
                annotations.append(annotation)
        else:
            for idx, row in data.iterrows():
                id=row['ID']
                annotation = {
                            "id": id,
                            "image": os.path.join(img_folder,id),
                        }
                annotations.append(annotation)

        return annotations

    
    def __getitem__(self, index):
        item = self.annotations[index]
        return item
    def __len__(self) -> int:
        return len(self.annotations)
        
class Get_Loader:
    def __init__(self, config):
        self.num_worker = config['data']['num_worker']

        self.train_path = config['data']['train_dataset']
        self.dev_path=config["data"]["dev_dataset"]
        self.test_path=config['infer']['test_dataset']

        self.train_img_path = config['data']['train_images_folder']
        self.dev_img_path=config["data"]["dev_images_folder"]
        self.test_img_path=config['infer']['test_images_folder']

        self.train_batch=config['train']['per_device_train_batch_size']
        self.dev_batch=config['train']['per_device_dev_batch_size']
        self.test_batch=config['infer']['per_device_test_batch_size']

    def load_train_dev(self):
        print("Reading training data...")
        train_set = Captioning_Dataset(self.train_path,self.train_img_path)
        print("Reading validation data...")
        dev_set = Captioning_Dataset(self.dev_path,self.dev_img_path)
    
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=self.num_worker,shuffle=True)
        dev_loader = DataLoader(dev_set, batch_size=self.dev_batch, num_workers=self.num_worker,shuffle=True)
        return train_loader, dev_loader
    
    def load_test(self,with_labels):
        print("Reading testing data...")
        test_set = Captioning_Dataset(self.test_path,self.test_img_path,with_labels=with_labels)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, num_workers=self.num_worker, shuffle=False)
        return test_loader