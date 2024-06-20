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
from data_utils.load_data import Get_Loader
from eval_metric.evaluate import ScoreCalculator
class Image_Captioning_Predict:
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
        self.compute_score = ScoreCalculator()
        self.dataloader=Get_Loader(config)

    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)
        logging.info("loadding best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        # Obtain the prediction from the model
        logging.info("Obtaining predictions...")
        test_set = self.dataloader.load_test(self.with_labels)
        if self.with_labels:
            ids=[]
            gts=[]
            preds=[]
            self.base_model.eval()
            with torch.no_grad():
                for it,item in enumerate(tqdm(test_set)):
                    with torch.cuda.amp.autocast(dtype=self.cast_dtype):
                        pre_caps = self.base_model(item['image'])                    
                        ids.extend(item['id'])
                        gts.extend(item['caption'])
                        preds.extend(pre_caps)
            test_em=self.compute_score.em(gts,preds)
            test_f1=self.compute_score.f1_token(gts,preds)
            test_cider=self.compute_score.cider_score(gts,preds)
            data={'id':ids,
                "ground_truth":gts,
                "predicts": preds}
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.save_path,'result.csv'), index=False)
            print(f"test em: {test_em:.4f} test f1: {test_f1:.4f} test cider: {test_cider:.4f}")
        else:
            y_preds={}
            self.base_model.eval()
            with torch.no_grad():
                for it,item in enumerate(tqdm(test_set)):
                    with torch.cuda.amp.autocast(dtype=self.cast_dtype):
                        pre_caps = self.base_model(item['image'])
                        for i in range(len(pre_caps)):
                            if isinstance(item['id'][i],torch.Tensor):
                                ids=item['id'].tolist()
                            else:
                                ids=item['id']
                            y_preds[str(ids[i])] = pre_caps[i]
            with open(os.path.join(self.save_path,'results.json'), 'w', encoding='utf-8') as r:
                json.dump(y_preds, r, ensure_ascii=False, indent=4)

