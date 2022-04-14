import numpy as np
from tracker import matching
M = 20
N = 10
def generate_graph(tracks):
    """
            note:this function may be used in train phase or inference phase,
                to generate_overlap_graph.
            parameter:
                tracks must be a list and it`s only contain bbox in one image.
            return overlap_graph [K,C,self.M,self.N]
    """
    def tlbr2tblrhw(tlbr):
        tlbr = list(tlbr)
        temp = tlbr[1]
        tlbr[1] = tlbr[2]
        tlbr[2] = temp
        tlbr += [tlbr[1] - tlbr[0],tlbr[3]-tlbr[2]]
        return  tlbr
    sub_graphs = np.zeros((len(tracks),M, N))
    temp = []
    for i in range(len(tracks)):
        temp.append((tracks[i].tlbr[3],i))
    temp.sort(key=lambda x: x[0])
    tracks.sort(key = lambda track:track.tlbr[3])
    shelter_id = [[] for _ in range(len(tracks))]
    iou_dist_mat = 1 - matching.iou_distance(tracks, tracks)
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            wi = tracks[i].tlwh[2]
            wj = tracks[j].tlwh[2]
            hi = tracks[i].tlwh[3]
            hj = tracks[j].tlwh[3]
            ratio = 1/0.5
            if iou_dist_mat[i, j] > 0 and 1/ratio < wi/wj < ratio and 1/ratio < hi/hj < ratio:
                be_shelted = np.concatenate([tracks[i].tlbr[1::-1], tracks[i].tlbr[-1:-3:-1]], 0)
                shelter = np.concatenate([tracks[j].tlbr[1::-1], tracks[j].tlbr[-1:-3:-1]], 0)
                overlap_tblr = (max(be_shelted[0], shelter[0]), min(be_shelted[2], shelter[2]),
                                max(be_shelted[1], shelter[1]), min(be_shelted[3], shelter[3]),)

                be_shelted = tlbr2tblrhw(be_shelted)
                be_shelted_tblr = ()
                for k in range(4):
                    be_shelted_tblr += (int(round((overlap_tblr[k] - be_shelted[k // 2 * 2]) / be_shelted[k // 2 + 4]
                                                  * (M,N)[k // 2])),)
                sub_graphs[i, be_shelted_tblr[0]:be_shelted_tblr[1], be_shelted_tblr[2]:be_shelted_tblr[3]] = temp[j][1] + 1
                shelter_id[i].append(temp[j][1] + 1)
    for i in range(len(temp)):
        while temp[i][1] != i:
            y = temp[i][1]

            t = tracks[y]
            tracks[y] = tracks[i]
            tracks[i] = t

            # sub_graphs[y] = sub_graphs[y] + sub_graphs[i]
            # sub_graphs[i] = sub_graphs[y] - sub_graphs[i]
            # sub_graphs[y] = sub_graphs[y] - sub_graphs[i]

            t = sub_graphs[y]
            sub_graphs[y] = sub_graphs[i]
            sub_graphs[i] = t

            t = shelter_id[y]
            shelter_id[y] = shelter_id[i]
            shelter_id[i] = t

            t = temp[y]
            temp[y] = temp[i]
            temp[i] = t

    return sub_graphs,shelter_id


def generate_shelter_relation(tracks):
    """
            note:this function may be used in train phase or inference phase,
                to generate_overlap_graph.
            parameter:
                tracks must be a list and it`s only contain bbox in one image.
            return overlap_graph [K,C,self.M,self.N]
    """

    temp = []
    for i in range(len(tracks)):
        temp.append((tracks[i].tlbr[3],i))
    temp.sort(key=lambda x: x[0])
    tracks.sort(key = lambda track:track.tlbr[3])
    shelter_id = [[] for _ in range(len(tracks))]
    iou_dist_mat = 1 - matching.iou_distance(tracks, tracks)
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if iou_dist_mat[i, j] > 0.5:
                shelter_id[i].append(temp[j][1])
    for i in range(len(temp)):
        while temp[i][1] != i:
            y = temp[i][1]

            t = tracks[y]
            tracks[y] = tracks[i]
            tracks[i] = t


            t = shelter_id[y]
            shelter_id[y] = shelter_id[i]
            shelter_id[i] = t

            t = temp[y]
            temp[y] = temp[i]
            temp[i] = t

    return shelter_id