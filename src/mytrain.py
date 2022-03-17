import lib.mywork.dataset as dataset
import lib.mywork.mynetwork as network
import lib.mywork.dataloader as dataloader
from lib.mywork.trainer import Trainer
from lib.models.model import load_model
from lib.models.model import save_model


import torch
import yaml
import os
def main():
    config = list(yaml.safe_load_all(open('config.yaml')))[0]
    print(config)
    train_config = config['train']
    val_config = config['val']
    model_config = config['model']
    trainset = dataset.Dataset(train_config)
    model = network.Mynetwork(model_config)
    loss = network.Myloss(model_config)
    optimizer = torch.optim.Adam(model.parameters(), train_config['base_lr'])
    start_epoch = train_config['start_epoch']

    # Get dataloader

    train_loader = dataloader.Dataloader(
        trainset,
        batch_size=train_config['batch_size'],
        shuffle=True,
    )
    val_loader = dataloader.Dataloader(
        trainset,
        batch_size=val_config['batch_size'],
        shuffle=True,
    )
    print('Starting training...')
    trainer = Trainer(model,loss,train_config,optimizer)
    if train_config['device'] == 'cuda':
        trainer.set_device('cuda:0')

    if train_config['load_model'] != '':
        model, optimizer, start_epoch = load_model(
            model, train_config['load_model'], trainer.optimizer, True, train_config['base_lr'], train_config['lr_step'])

    for epoch in range(start_epoch,train_config['num_epochs']+1):
        mark = epoch if train_config['save_all'] else 'last'
        trainer.train(epoch, train_loader)
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
