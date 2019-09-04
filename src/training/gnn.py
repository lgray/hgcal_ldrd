"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import time
import math

# Externals
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import tqdm

import numpy as np

from models import get_model, get_losses
# Locals
from .base import base


class GNNTrainer(base):
    """Trainer code for basic classification problems with binomial cross entropy."""

    def __init__(self, real_weight=1, fake_weight=1, category_weights=None, **kwargs):
        super(GNNTrainer, self).__init__(**kwargs)
        if category_weights is None:
            self._category_weights = torch.tensor([fake_weight, real_weight]).to(self.device)
        else:
            self._category_weights = torch.tensor(category_weights.astype(np.float32)).to(self.device)

    def build_model(self, name='EdgeNet',
                    loss_func='binary_cross_entropy',
                    optimizer='Adam', learning_rate=0.01, lr_scaling=None, lr_warmup_epochs=0,
                    **model_args):
        """Instantiate our model"""

        # Construct the model
        self.model = get_model(name=name, **model_args).to(self.device)

        # Construct the loss function
        get_losses()
        self.loss_func = getattr(nn.functional, loss_func)

        # Construct the optimizer
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(), lr=learning_rate)

        self.lr_scheduler = None
        if lr_scaling is not None:
            self.lr_scheduler = lr_scaling(self.optimizer)
           

    # @profile
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()        
        summary = dict()
        sum_loss = 0.
        start_time = time.time()
        # Loop over training batches
        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        t = tqdm.tqdm(enumerate(data_loader),total=int(math.ceil(total/batch_size)))
        cat_weights = self._category_weights
        
        acc_rate = 10.
        acc_norm = 1./acc_rate        
        acc_loss = 0.
        
        self.optimizer.zero_grad()
        for i,data in t:            
            data = data.to(self.device)
            batch_target = data.y
            if self.loss_func == F.binary_cross_entropy:
                #binary cross entropy expects a weight for each event in a batch
                #categorical cross entropy ex
                batch_weights_real = batch_target*self._category_weights[1]
                batch_weights_fake = (1 - batch_target)*self._category_weights[0]
                cat_weights = batch_weights_real + batch_weights_fake            
            batch_output = self.model(data)
            batch_loss = acc_norm * self.loss_func(batch_output,batch_target,weight=cat_weights) 
            #batch_loss = torch.tensor(0, dtype=torch.float).to(self.device)
            #sub_batches = batch_output.shape[0]//10000 + 1
            #for i in range(sub_batches):
            #    print(batch_output.shape[0],batch_output.shape[0]//10000 + 1)
            #    sub_batch_loss = self.loss_func(batch_output[i*10000:(i+1)*10000], 
            #                                    batch_target[i*10000:(i+1)*10000],
            #                                    weight=cat_weights)
            #    sub_batch_loss = sub_batch_loss / sub_batches
            #    sub_batch_loss.backward()
            #    batch_loss += sub_batch_loss
            #    print(batch_loss)            
            batch_loss.backward()
            batch_loss_item = batch_loss.item()
            acc_loss += batch_loss_item
            sum_loss += batch_loss_item
            if( (i+1) % acc_rate == 0 or (i+1) == total): 
                self.optimizer.step()
                self.optimizer.zero_grad()
                t.set_description("loss = %.5f" % acc_loss )
                t.refresh() # to show immediately the update
                acc_loss = 0.
                      
            #self.logger.debug('  batch %i, loss %f', i, batch_loss.item())

        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = acc_rate * sum_loss / (i + 1)
        self.logger.debug(' Processed %i batches', (i + 1))
        self.logger.info('  Training loss: %.5f', summary['train_loss'])
        self.logger.info('  Learning rate: %.5f', summary['lr'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.zero_grad()
        torch.cuda.empty_cache()
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        start_time = time.time()
        # Loop over batches
        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        t = tqdm.tqdm(enumerate(data_loader),total=int(math.ceil(total/batch_size)))
        num = torch.zeros_like(self._category_weights)
        denm = torch.zeros_like(self._category_weights)
        
        cat_wgt_shape = self._category_weights.shape[0]
      
        confusion_num = torch.zeros([cat_wgt_shape,cat_wgt_shape]).to(self.device)
        confusion_denm = torch.zeros([cat_wgt_shape,cat_wgt_shape]).to(self.device)
        
        for i, data in t:            
            # self.logger.debug(' batch %i', i)
            batch_input = data.to(self.device)
            batch_target = data.y
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            sum_loss += batch_loss.item()
            # Count number of correct predictions
            #print(batch_output)
            #print('torch.max',torch.argmax(batch_output,dim=-1))
            
            truth_cat_counts = torch.unique(batch_target, return_counts = True)
            pred = torch.argmax(batch_output,dim=-1)
            
            for j in range(cat_wgt_shape):
                cat_counts = torch.unique(pred[batch_target == j], return_counts=True)                
                confusion_num[:,j][cat_counts[0]] += cat_counts[1].float()
                confusion_denm[j,:][truth_cat_counts[0]] += truth_cat_counts[1].float()
                        
            matches = (pred == batch_target)
            trues_by_cat = torch.unique(pred[matches], return_counts=True)
            

            num[trues_by_cat[0]] += trues_by_cat[1].float()
            denm[truth_cat_counts[0]] += truth_cat_counts[1].float()
                        
            
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            #self.logger.debug(' batch %i loss %.3f correct %i total %i',
            #                  i, batch_loss.item(), matches.sum().item(),
            #                  matches.numel())
        torch.cuda.empty_cache()
        self.logger.debug('loss %.5f cat effs %s',sum_loss / (i + 1), np.array_str((num/denm).cpu().numpy()))
        self.logger.debug('loss %.5f cat confusions:\n %s',
                          sum_loss / (i + 1),
                          np.array_str((confusion_num/confusion_denm).cpu().numpy()))
        self.logger.debug('loss %.5f cat true counts %s',sum_loss / (i + 1), (denm).cpu().numpy())
        self.logger.debug('loss %.5f cat wgt counts %s',sum_loss / (i + 1), (self._category_weights*denm).cpu().numpy())
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(sum_loss / (i + 1))
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / sum_total
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('  Validation loss: %.5f acc: %.5f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary


def _test():
    t = GNNTrainer(output_dir='./')
    t.build_model()
