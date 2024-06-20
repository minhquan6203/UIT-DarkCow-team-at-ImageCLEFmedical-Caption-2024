import argparse
import os
import yaml
import logging
from typing import Text, Dict, List
import pandas as pd
import torch
import transformers
import json
import shutil
from tqdm import tqdm
from builder.model_builder import build_model
import cv2
import matplotlib.pyplot as plt

class Image_Captioning_Demo:
    def __init__(self,config: Dict):
        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.save_path = os.path.join(config['train']['output_dir'],config['model']['type_model'])
        self.checkpoint_path=os.path.join(self.save_path, "best_model.pth")
        self.with_labels=config['infer']['with_labels']
        if config['train']['precision']=='float32':
            self.cast_dtype=torch.float32
        elif config['train']['precision']=='bfloat16':
            self.cast_dtype=torch.bfloat16
        else:
            self.cast_dtype=torch.float16
        self.base_model = build_model(config).to(self.device)
    
    def predict(self, image_path):
        base_name = os.path.basename(image_path).split()[0]
        checkpoint = torch.load(self.checkpoint_path)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.cast_dtype):
                pre_caps = self.base_model([base_name])
                return pre_caps
    
def main(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    task_infer = Image_Captioning_Demo(config)
    path = input('please enter image path: ')
    pre_caps = task_infer.predict(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    print(pre_caps)
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    main(args.config)



