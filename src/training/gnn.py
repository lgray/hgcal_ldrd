"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import time
import math

# Externals
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import tqdm

from models import get_model
# Locals
from .base import base


class GNNTrainer(base):
    """Trainer code for basic classification problems with binomial cross entropy."""

    def __init__(self, real_weight=1, fake_weight=1, **kwargs):
        super(GNNTrainer, self).__init__(**kwargs)
        self.real_weight = real_weight
        self.fake_weight = fake_weight

    def build_model(self, name='EdgeNet',
                    loss_func='binary_cross_entropy',
                    optimizer='Adam', learning_rate=0.01, lr_scaling=None, lr_warmup_epochs=0,
                    **model_args):
        """Instantiate our model"""

        # Construct the model
        self.model = get_model(name=name, **model_args).to(self.device)

        # Construct the loss function
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
        self.lr_scheduler.step()
        # Loop over training batches
        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        t = tqdm.tqdm(enumerate(data_loader),total=int(math.ceil(total/batch_size)))
        for i,data in t:            
            data = data.to(self.device)
            batch_target = data.y
            batch_weights_real = batch_target*self.real_weight
            batch_weights_fake = (1 - batch_target)*self.fake_weight
            batch_weights = batch_weights_real + batch_weights_fake
            self.optimizer.zero_grad()
            batch_output = self.model(data)
            batch_loss = self.loss_func(batch_output, batch_target, weight=batch_weights)
            batch_loss.backward()
            batch_loss_item = batch_loss.item()            
            self.optimizer.step()
            
            sum_loss += batch_loss.item()
            t.set_description("loss = %.5f" % batch_loss.item() )
            t.refresh() # to show immediately the update
            #self.logger.debug('  batch %i, loss %f', i, batch_loss.item())

        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = sum_loss / (i + 1)
        self.logger.debug(' Processed %i batches', (i + 1))
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        # self.logger.info('  Learning rate: %.5f', summary['lr'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
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
        for i, data in t:            
            # self.logger.debug(' batch %i', i)
            batch_input = data.to(self.device)
            batch_target = data.y
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            sum_loss += batch_loss.item()
            # Count number of correct predictions
            matches = ((batch_output > 0.5) == (batch_target > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            #self.logger.debug(' batch %i loss %.3f correct %i total %i',
            #                  i, batch_loss.item(), matches.sum().item(),
            #                  matches.numel())                           
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / sum_total
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary


def _test():
    t = GNNTrainer(output_dir='./')
    t.build_model()
