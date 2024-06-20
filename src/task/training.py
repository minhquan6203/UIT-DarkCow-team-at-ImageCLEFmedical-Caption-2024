import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import os
import numpy as np
from tqdm import tqdm
import gc
from data_utils.load_data import Get_Loader
from model.blip2_captioning import *
from eval_metric.evaluate import ScoreCalculator
from utils.utils import countTrainableParameters, countParameters
from builder.model_builder import build_model

class Image_Captioning_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.save_path = os.path.join(config['train']['output_dir'],config['model']['type_model'])
        self.best_metric= config['train']['metric_for_best_model']
        self.learning_rate = config['train']['learning_rate']
        self.weight_decay=config['train']['weight_decay']
        self.dataloader = Get_Loader(config)
        if config['train']['precision']=='float32':
            self.cast_dtype=torch.float32
        elif config['train']['precision']=='bfloat16':
            self.cast_dtype=torch.bfloat16
        else:
            self.cast_dtype=torch.float16

        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.base_model=build_model(config).to(self.device)
        trainable_param = countTrainableParameters(self.base_model)
        model_param=countParameters(self.base_model)
        print('num trainable params: ', trainable_param)
        print(f'% param: {(trainable_param/model_param)*100:.4f}')
        self.compute_score = ScoreCalculator()
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
    
        train,valid = self.dataloader.load_train_dev()
        
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_loss = checkpoint['loss']
        else:
            best_loss = 100.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            train_loss = 0.
            valid_loss = 0.
            with tqdm(desc='Epoch %d - Training stage' % (epoch+1), unit='it', total=len(train)) as pbar:
                for it, item in enumerate(train):
                    with torch.cuda.amp.autocast(dtype=self.cast_dtype):
                        logits, loss = self.base_model(item['image'],item['caption'])
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    train_loss += loss.item()
                    pbar.set_postfix(loss=train_loss / (it + 1))
                    pbar.update()
                self.scheduler.step()
                train_loss /=len(train)
            
            with torch.no_grad():
                with tqdm(desc='Epoch %d - Valid stage' % (epoch+1), unit='it', total=len(valid)) as pbar:
                    for it, item in enumerate(valid):
                        with torch.cuda.amp.autocast(dtype=self.cast_dtype):
                            logits, loss = self.base_model(item['image'],item['caption'])
                            valid_loss+=loss.item()
                            pbar.set_postfix(loss=valid_loss / (it + 1))
                            pbar.update()
                    valid_loss /=len(valid)
            
            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid loss: {valid_loss:.4f}")
            # Mở tệp tin log.txt trong chế độ ghi ('w' để ghi đè, 'a' để thêm vào cuối file)
            with open(os.path.join(self.save_path,'log.txt'), 'a') as file:
                file.write(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}\n")
                file.write(f"train loss: {train_loss:.4f}\n")
                file.write(f"valid loss: {valid_loss:.4f}\n")


            if self.best_metric =='loss':
                loss=valid_loss
            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss}, os.path.join(self.save_path, 'last_model.pth'))
            
            # save the best model
            if epoch > 0 and loss >= best_loss:
              threshold += 1
            else:
              threshold = 0

            if loss < best_loss:
                best_loss = loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss':loss}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {loss:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break
