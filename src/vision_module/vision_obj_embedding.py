import torch
from torch import nn
import os
from typing import List
from collections import Counter
from typing import List, Dict,Any
import numpy as np

class VisionObjEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_region_features = nn.Linear(config.d_obj,config.d_model)
        self.linear_region_boxes = nn.Linear(4, config.d_model)

        self.layer_norm_region = nn.LayerNorm(config.d_model)
        self.layer_norm_region_boxes = nn.LayerNorm(config.d_model)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.cuda_device=config.cuda_device
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')

    def forward(self,obj_info):
        region_features=torch.stack([region["region_features"] for region in obj_info]).to(self.device)
        region_boxes=torch.stack([region["region_boxes"] for region in obj_info]).to(self.device)

        region_features=self.linear_region_features(region_features)
        region_boxes=self.linear_region_boxes(region_boxes)

        region_obj_features = self.layer_norm_region(region_features) + self.layer_norm_region_boxes(region_boxes)
        obj_features = self.dropout(self.gelu(region_obj_features))
        return obj_features

class SemanticObjectEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.max_bbox = config.max_bbox
        self.cuda_device=config.cuda_device
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')

        self.linear_region_features = nn.Linear(config.d_obj,config.d_model)
        self.linear_region_boxes = nn.Linear(4, config.d_model)

        self.layer_norm_region = nn.LayerNorm(config.d_model)
        self.layer_norm_region_boxes = nn.LayerNorm(config.d_model)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def pad_tensor(self, tensor: torch.Tensor, max_len: int, value):
        if max_len == 0:
            tensor = torch.zeros((0, tensor.shape[-1]))
        else:
            pad_value_tensor = torch.zeros((max_len-tensor.shape[0], tensor.shape[-1])).fill_(value).to(self.device)
            tensor = torch.cat([tensor, pad_value_tensor], dim=0)
        return tensor

    def forward(self,
                obj_info,
                obj_embs) -> torch.Tensor:

        region_features=torch.stack([region["region_features"] for region in obj_info]).to(self.device)
        region_boxes=torch.stack([region["region_boxes"] for region in obj_info]).to(self.device)

        region_features=self.linear_region_features(region_features)
        region_boxes=self.linear_region_boxes(region_boxes)
        
        region_features=self.layer_norm_region(region_features)
        region_boxes=self.layer_norm_region_boxes(region_boxes)

        obj_features =  region_features + region_boxes + obj_embs
        obj_features = self.dropout(self.gelu(obj_features))

        return obj_features