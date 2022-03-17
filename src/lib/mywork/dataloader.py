import math
from torch import stack as stk
class Dataloader():
    def __init__(self,dataset,batch_size,shuffle):
        self.len = math.ceil(dataset.__len__()/batch_size)
        # 为next函数计数器初始化
        self.count = 0
        self.batch_size = batch_size
        self.inds = [i for i in range(dataset.__len__())]
        self.dataset = dataset
    def __iter__(self):
        self.count = 0
        import random
        random.shuffle(self.inds)
        return self
    def __next__(self):
        # 获取下一个数
        if self.count < self.len:
            result = self.inds[self.count*self.batch_size:min((self.count+1)*self.batch_size,self.dataset.__len__())]
            tar_f = []
            tar_m = []
            p_f = []
            p_m = []
            n_f = []
            n_m = []
            for ind in result:
                x,y,z = self.dataset.__getitem__(ind)
                tar_f.append(x[0])
                tar_m.append(x[1])
                p_f.append(y[0])
                p_m.append(y[1])
                n_f.append(z[0])
                n_m.append(z[1])
            self.count += 1
            return (stk(tar_f),stk(tar_m)),(stk(p_f),stk(p_m)),(stk(n_f),stk(n_m))
        else:
            raise StopIteration
    def __len__(self):
        return self.len