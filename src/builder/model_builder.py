from model.blip2_captioning import BLIP2_Captioning_Model
from model.gpt2_captioning import GPT2_Captioning_Model
from model.bart_captioning import BART_Captioning_Model
from model.blip_captioning import BLIP_Captioning_Model
from model.vision_lm_captioning import VisionLM_Captioning_Model

def build_model(config):
    if config['model']['type_model']=='blip2':
        return BLIP2_Captioning_Model(config)
    if config['model']['type_model']=='gpt2':
        return GPT2_Captioning_Model(config)
    if config['model']['type_model']=='biobart':
        return BART_Captioning_Model(config)
    if config['model']['type_model']=='blip':
        return BLIP_Captioning_Model(config)
    if config['model']['type_model']=='vision_lm':
        return VisionLM_Captioning_Model(config)