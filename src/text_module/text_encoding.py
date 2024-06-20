from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
from PIL import Image

class Text_Encode(nn.Module):
    def __init__(self, config: Dict):
        super(Text_Encode,self).__init__()
        self.processor_text = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        if self.processor_text.pad_token is None:
            self.processor_text.pad_token="<pad>"
         
        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.padding = config["tokenizer"]["padding"]
        self.max_input_length = config["tokenizer"]["max_input_length"]
        self.truncation = config["tokenizer"]["truncation"]
    def forward(self, captions: List[str]):
        encoding_text=self.processor_text(captions,padding=self.padding,
                                        max_length=self.max_input_length,
                                        truncation=self.truncation, 
                                        return_tensors="pt").to(self.device)
        labels_input_ids = encoding_text["input_ids"].clone()
        decoder_attention_mask=encoding_text["attention_mask"].clone()
        labels_input_ids[decoder_attention_mask==0]=-100
        encoding_text.update({"labels":labels_input_ids})
        encoding_text.update({"decoder_attention_mask":decoder_attention_mask})
        return encoding_text