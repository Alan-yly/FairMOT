import json
import os.path


class det_feat_recorder():
    def __init__(self,seq,out_path,mode):
        self.mode = mode
        if mode == 'record':
            self.out_path = out_path
            self.seq = seq
            self.records = []
            self.record_mats = []
        elif mode == 'get':
            fp = open(os.path.join(out_path,seq+'.json'), 'r')
            self.records,self.record_mats = json.load(fp)
            fp.close()
            self.ind = -1
            self.mat_ind = -1
        else:
            exit()

    def __del__(self):
        # if self.mode == 'record':
        #     self.fp = open(os.path.join(self.out_path, self.seq + '.json'), 'w')
        #     json.dump((self.records,self.record_mats),self.fp)
        #     self.fp.close()
        #     print('finish record!')
        pass
    def record(self,a,b):
        if self.mode == 'record':
            self.records.append((a,b))
    def record_mat(self,mat):
        if self.mode == 'record':
            self.record_mats.append(mat)
    def get(self):
        if self.mode == 'get':
            self.ind += 1
            return self.records[self.ind]
    def get_mat(self):
        if self.mode == 'get':
            self.mat_ind += 1
            return self.record_mats[self.mat_ind]
