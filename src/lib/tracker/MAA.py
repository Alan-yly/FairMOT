import  numpy as np
class uset():
    def __init__(self,n):
        self.sets = [i for i in range(n)]
    def union(self,x,y):
        self.sets[self.find(y)] = self.find(x)
        return
    def find(self,x):
        if self.sets[x] == x:
            return x
        self.sets[x] = self.find(self.sets[x])
        return self.sets[x]


class MAA():
    def __init__(self,mat,delta,mindist):
        num_track = mat.shape[0]
        num_dets = mat.shape[1]
        self.mat = mat
        self.record = [False for _ in range(num_dets + num_track)]
        self.groups = uset(num_dets+num_track)
        self.delta = delta
        self.mindist = mindist
    @staticmethod
    def search(mode,mat,groups,ind,delta,mindist,record):
        if groups.find(ind) != ind or record[ind] :
            return
        record[ind] = True
        if mode == 'Track':
            print(ind,mat.shape)
            tmp = mat[ind]
            if np.sum(tmp[tmp < mindist]-min(tmp) < delta) <= 1:
                return
            for i,x in enumerate(tmp):
                if x - min(tmp) < delta and x < mindist:
                    MAA.search('Det',mat,groups,len(mat)+i,delta,mindist,record)
                    groups.union(ind,len(mat)+i)
            return
        elif mode == 'Det':
            tmp = mat[:,ind-len(mat)]
            if np.sum(tmp[tmp < mindist] - min(tmp) < delta) <= 1:
                return
            for i,x in enumerate(tmp):
                if x - min(tmp) < delta and x < mindist:
                    MAA.search('Track',mat,groups,i,delta,mindist,record)
                    groups.union(ind,i)
        else:
            print("error!")
            exit()
            return
    def serach_all(self):
        for i,_ in enumerate(self.record):
            if i < self.mat.shape[0]:
                MAA.search('Track',self.mat,self.groups,i,self.delta,self.mindist,self.record)
            else:
                MAA.search('Det',self.mat,self.groups,i,self.delta,self.mindist,self.record)
        map = {}
        for i,x in enumerate(self.groups.sets):
            if x not in map.keys():
                map[x] = set()
            map[x].add(i)
        mats = []
        tracks = []
        dets = []
        for k in map.keys():
            v = map[k]
            if len(v) == 1:
                if k < self.mat.shape[0]:
                    tracks.append(k)
                else:
                    dets.append(k-self.mat.shape[0])
            else:
                tmp_track = []
                tmp_det = []
                for i in v:
                    if i < self.mat.shape[0]:
                        tmp_track.append(i)
                    else:
                        tmp_det.append(i-self.mat.shape[0])
                mats.append((tmp_track,tmp_det))
        mats.append((tracks,dets))
        return mats


mat = np.array(
    [
        [1,1,1,0.38,1,1,1,1],
        [1,1,1,1,1,1,1,0.87],
        [1,0.34,1,1,1,0.31,1,1],
        [1,1,1,0.31,1,1,1,1],
        [0.92,1,1,1,0.06,1,1,1],
        [1,1,0.11,1,1,1,0.19,1],
        [1,0.29,1,1,1,0.37,1,1]
    ]
)
maa = MAA(mat,0.1,0.5,)
print(maa.serach_all())