import torch
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, outputs, batch):
        labels = batch['ids']
        outputs = outputs[0]
        loss = self.loss(outputs,labels)
        acc = torch.mean((torch.argmax(outputs,-1) == labels)*1.)
        loss_stats = {'loss': loss,'acc':acc}
        return loss, loss_stats