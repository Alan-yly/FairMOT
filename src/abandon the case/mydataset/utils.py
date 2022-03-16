import numpy as np
import  cv2
import random
import os
import torch
import json
from torchvision.transforms import transforms as T

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,

def preprocess(image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

def random_affine(img, targets=None,ids =None,degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    import math
    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets.copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, ] = xy[i]
            ids = ids[i]
            ind = np.logical_and(targets[:, 0] < width,targets[:, 2] > 0)
            ind = np.logical_and(ind,targets[:, 1] < height)
            ind = np.logical_and(ind,targets[:, 3] > 0)
            targets = targets[ind]
            ids = ids[ind]

        return imw, targets,ids, M
    else:
        return imw


class LoadImagesAndLabels(torch.utils.data.Dataset):  # for training
    def __init__(self, labels_pth, img_size, augment=True, transforms=T.Compose([T.ToTensor()])):
        super(LoadImagesAndLabels, self).__init__()
        f = open(labels_pth,'r')
        labels = json.load(f)
        f.close()
        self.labels = [(key,labels[key]) for key in labels.keys()]

        self.nF = len(self.labels)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms
        self.max_objs = 100
    def __getitem__(self, index):

        img_path = self.labels[index][0]
        labels = self.labels[index][1]
        ids = []
        tlbrs = []
        for label in labels:
            ids.append([label[0]])
            tlbrs.append(label[1])
        return self.get_data(img_path,np.array(tlbrs).astype(np.float),np.array(ids).astype(np.float))

    def get_data(self, img_path,labels,ids):
        img, framed_imgs, framed_metas = preprocess([img_path],self.width)
        img = img[0]

        meta = framed_metas[0]
        h, w, _ = img.shape
        img = framed_imgs[0]
        ratio = meta[-5] / meta[-3]
        padw = meta[-2]
        padh = meta[-1]
        # Load labels
        labels *= ratio
        labels[:, 0] += padw
        labels[:, 1] += padh
        labels[:, 2] += padw
        labels[:, 3] += padh


        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        # Augment image and labels
        if self.augment:
            img, labels,ids, M = random_affine(img, labels,ids, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))
            nL = len(labels)
            if nL > 0:
                if (random.random() > 0.5):
                    img = np.fliplr(img)
                    labels[:,0] = img.shape[1] - labels[:,0]
                    labels[:,2] = img.shape[1] - labels[:,2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)
        tlbrs = np.zeros((self.max_objs,4))
        tlbrs[:len(labels)] = labels
        id = np.ones((self.max_objs,1))* (-1)
        id[:len(ids)] = ids
        id = id.astype(np.int64)
        return {'img':img,'tlbrs':tlbrs,'ids':id}

    def __len__(self):
        return self.nF  # number of batches




if __name__ == '__main__':
    trainset = LoadImagesAndLabels('G:\project\MOT\DataSet\\all_lables.json',(512,512))
    dataset = torch.utils.data.DataLoader(
        trainset,
        batch_size=10,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

