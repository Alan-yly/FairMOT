from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from datasets.mydataset.utils import LoadImagesAndLabels
from trains.mytrainer.base_trainer import Trainer
from models.my_network.model import Net
from models.model import load_model
from models.model import save_model
import torch
import yaml
import os
def main():
    config = list(yaml.safe_load_all(open('config.yaml')))[0]
    print(config)
    compound_coef = config['model']['compound_coef']
    train_config = config['train']
    val_config = config['val']
    model_config = config['model']
    imgsize = model_config['imgsize'][compound_coef]
    trainset = LoadImagesAndLabels(train_config['label_pth'],(imgsize,imgsize))
    valset = LoadImagesAndLabels(val_config['label_pth'], (imgsize,imgsize),False)
    model = Net(compound_coef,train_config['class_num'])
    model.to(train_config['device'])
    if train_config['init_weights'] != '':
        model.load_weights(os.path.join(train_config['init_weights'],'efficientnet-d{}.pth'.format(compound_coef)))


    optimizer = torch.optim.Adam(model.parameters(), train_config['base_lr'])
    start_epoch = train_config['start_epoch']

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=val_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    print('Starting training...')
    trainer = Trainer(model,train_config,optimizer)
    if train_config['device'] == 'cuda':
        trainer.set_device('cuda:0')

    if train_config['load_model'] != '':
        model, optimizer, start_epoch = load_model(
            model, train_config['load_model'], trainer.optimizer, True, train_config['base_lr'], train_config['lr_step'])

    for epoch in range(start_epoch,train_config['num_epochs']+1):
        mark = epoch if train_config['save_all'] else 'last'
        # log_dict_train, _ = trainer.train(epoch, train_loader)
        print('!')
        if val_config['val_intervals'] > 0 and epoch % val_config['val_intervals'] == 0:
            trainer.val(epoch, val_loader)
            save_model(os.path.join(train_config['save_dir'], 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(train_config['save_dir'], 'model_last.pth'),
                       epoch, model, optimizer)
        if epoch in train_config['lr_step']:
            save_model(os.path.join(train_config['save_dir'], 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = train_config['base_lr'] * (0.1 ** (train_config['lr_step'].index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(train_config['save_dir'], 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)


if __name__ == '__main__':
    main()
