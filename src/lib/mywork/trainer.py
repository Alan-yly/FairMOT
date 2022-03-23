import time
import torch
from progress.bar import Bar
from ..utils.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
    self.loss_state = {}
  def forward(self, batch):
    out1 = self.model(batch[0][0],batch[1][0],batch[0][1],batch[1][1])
    out2 = self.model(batch[0][0],batch[2][0],batch[0][1],batch[2][1])
    loss = self.loss(out1,out2)
    return loss

class Trainer(object):
  def __init__(
    self, model,loss,train_config,optimizer=None,):
    self.train_config = train_config
    self.optimizer = optimizer
    self.model_with_loss = ModleWithLoss(model, loss)
    self.loss = self.model_with_loss.loss
    self.loss_dict = self.model_with_loss.loss.loss_dict
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
    self.loss.reset_ap()

    results = {}
    avg_loss_stats = {l: AverageMeter() for l in self.loss_dict}
    num_iters = len(data_loader) if self.train_config['num_iters'] < 0 else self.train_config['num_iters']
    bar = Bar('{}/{}'.format('mot', ''), max=num_iters)
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      n_iter = iter_id + epoch * num_iters
      if phase == 'train':
        loss = model_with_loss(batch)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('Loss/train', loss, n_iter)
        self.writer.add_scalar('p/train', self.loss_dict['p'].mean(), n_iter)
        self.writer.add_scalar('n/train', self.loss_dict['n'].mean(), n_iter)
      else:
        with torch.no_grad():
          loss = model_with_loss(batch)
          loss = loss.mean()
          self.writer.add_scalar('Loss/test', loss, n_iter)
          self.writer.add_scalar('p/test', self.loss_dict['p'].mean(), n_iter)
          self.writer.add_scalar('n/test', self.loss_dict['n'].mean(), n_iter)

      bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          self.loss_dict[l].mean().item(),self.train_config['batch_size'])
        bar.suffix = bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      bar.next()
      del  loss, batch
    print(self.loss.compute_ap())
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results

  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError


  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)