from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoTokenizer
from model.backbone.vision_casual_lm import Vision_Cassual_LM
from model.backbone.vision_seq2seq_lm import Vision_Seq2Seq_LM
from vision_module.vision_pixel_embedding import Vision_Embedding, Vision_Embedding_Extracted
from text_module.text_encoding import Text_Encode
from PIL import Image

class VisionLM_Captioning_Model(nn.Module):
    def __init__(self,config: Dict):
        super(VisionLM_Captioning_Model, self).__init__()
        self.processor_text = Text_Encode(config)
        if config["vision_embedding"]["already_extracted"]:
            self.processor_image = Vision_Embedding_Extracted(config)
        else:
            self.processor_image = Vision_Embedding(config)
        vision_name=config["vision_embedding"]["image_encoder"]
        lm_name=config["text_embedding"]["text_encoder"]
        use_lora=config["text_embedding"]["use_lora"]
        freeze_lm=config['text_embedding']['freeze']
        if config['train']['precision']=='float32':
            cast_dtype=torch.float32
        elif config['train']['precision']=='bfloat16':
            cast_dtype=torch.bfloat16
        else:
            cast_dtype=torch.float16
        if 't5' in lm_name.lower() or 't0' in lm_name.lower() or 'bart' in lm_name.lower():
            self.embedding = Vision_Seq2Seq_LM(vit_pretrained=vision_name,
                                    lm_pretrained=lm_name,
                                    freeze_lm=freeze_lm,
                                    cast_dtype=cast_dtype,
                                    use_lora=use_lora)
        else:
            self.embedding = Vision_Cassual_LM(vit_pretrained=vision_name,
                                    lm_pretrained=lm_name,
                                    freeze_lm=freeze_lm,
                                    cast_dtype=cast_dtype,
                                    use_lora=use_lora)
            
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token="<pad>"
        self.generator_args ={
            "max_length": config["generator_args"]["max_length"],
            "min_length": config["generator_args"]["min_length"],
            "num_beams": config["generator_args"]["num_beams"],
            "length_penalty": config["generator_args"]["length_penalty"],
            "no_repeat_ngram_size": config["generator_args"]["no_repeat_ngram_size"],
            "early_stopping": config["generator_args"]["early_stopping"],
        }
    def forward(self, images: List[str], captions: List[str] = None):
        inputs = self.processor_image(images)
        if captions is not None:
            encoding_text = self.processor_text(captions)
            del encoding_text["input_ids"]
            del encoding_text["attention_mask"]
            inputs.update(encoding_text)
            outputs = self.embedding(**inputs)
            return outputs.logits , outputs.loss
        else:
            pred_ids=self.embedding.generate(**inputs,**self.generator_args)
            pred_tokens=self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens