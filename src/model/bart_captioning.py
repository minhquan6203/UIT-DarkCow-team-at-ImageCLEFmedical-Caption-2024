from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoConfig
from model.backbone.custom_vision_encoder_decoder import VisionEncoderDecoderModel,VisionEncoderDecoderConfig
from text_module.text_encoding import Text_Encode
from vision_module.vision_pixel_encoding import Vision_Encode_Pixel
import os
from PIL import Image

class BART_Captioning_Model(nn.Module):
    def __init__(self,config: Dict):
        super(BART_Captioning_Model, self).__init__()
        self.processor_text = Text_Encode(config)
        self.processor_image = Vision_Encode_Pixel(config)
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

        lm_config=AutoConfig.from_pretrained(config["text_embedding"]["text_encoder"])
        lm_config.update({"add_cross_attention":True})
        vision_config=AutoConfig.from_pretrained(config["vision_embedding"]["image_encoder"])
        model_config=VisionEncoderDecoderConfig(encoder=vision_config.to_dict(),
                                                decoder=lm_config.to_dict())

        model_config.update({"vision_name": config["vision_embedding"]["image_encoder"],
                        "lm_name": config["text_embedding"]["text_encoder"],
                        "bos_token_id":lm_config.bos_token_id,
                        "decoder_start_token_id":lm_config.bos_token_id,
                        "pad_token_id": lm_config.pad_token_id,
                        "is_encoder_decoder": True,
                        "model_type": "vision-encoder-decoder",
                        "tie_word_embeddings": False,
                        "transformers_version": None})
        self.embedding = VisionEncoderDecoderModel(config=model_config)
    def forward(self, images: List[str], captions: List[str] = None):
        inputs = {'pixel_values':self.processor_image(images)}
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