

import time
import torch
from progress.bar import Bar

from src.lib.utils.utils import AverageMeter
from src.lib.models.my_network.loss import Loss
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch)
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class Trainer(object):
  def __init__(
    self, model, train_config,optimizer=None,):
    self.train_config = train_config
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses()
    self.model_with_loss = ModleWithLoss(model, self.loss)
    self.optimizer.add_param_group({'params': self.loss.parameters()})
    self.writer = SummaryWriter(train_config['logdir'])
    shutil.rmtree(train_config['logdir'])
    os.makedirs(train_config['logdir'])
  def set_device(self, device):
    self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      model_with_loss.eval()
      torch.cuda.empty_cache()

    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if self.train_config['num_iters'] < 0 else self.train_config['num_iters']
    bar = Bar('{}/{}'.format('mot', ''), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      batch['img'] = batch['img'].to(device = self.train_config['device'],dtype=torch.float32)
      batch['tlbrs'] = batch['tlbrs'].to(device=self.train_config['device'], dtype=torch.float32)
      batch['ids'] = batch['ids'].to(device=self.train_config['device'], dtype=torch.long)

      n_iter = iter_id + epoch * num_iters
      if phase == 'train':
        output, loss, loss_stats = model_with_loss(batch)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('Loss/train', loss, n_iter)
        self.writer.add_scalar('Accuracy/train', loss_stats['acc'], n_iter)
      else:
        with torch.no_grad():
          output, loss, loss_stats = model_with_loss(batch)
          loss = loss.mean()
          self.writer.add_scalar('Loss/test', loss, n_iter)
          self.writer.add_scalar('Accuracy/test', loss_stats['acc'], n_iter)


      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(),self.train_config['batch_size'])
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if self.train_config['print_iter'] > 0:
        if iter_id % self.train_config['print_iter'] == 0:
          print('{}/{}| {}'.format('mot', '', Bar.suffix))
      else:
        bar.next()

      del output, loss, loss_stats, batch

    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results

  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self):
    loss_states = ['loss']
    loss = Loss()
    return loss_states, loss
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)