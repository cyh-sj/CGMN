import numpy as np

def getIou(boxA, boxB):
    #boxA = [int(x) for x in boxA]
    #boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[0] - boxA[2]) * (boxA[1] - boxA[3])
    boxBArea = (boxB[0] - boxB[2]) * (boxB[1] - boxB[3])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    if iou >= 0.4:
        iou = 1.0
    else:
        iou = 0

    return iou

url = '/home/cyh/cross-model/DARN/data/data/'

#dataset = ['f30k_precomp/', 'coco_precomp/']
dataset = ['coco_precomp/']
#set2 = ['train','dev','testall']
set2 = ['testall']
name = '_ims_bbx.npy'
savename = '_IOU.npy'

for i in dataset:
    for j in set2:
        bboxfull = np.load(url+i+j+name)
        size = len(bboxfull)
        print(size)
        Iou = np.zeros([size,36,36])

        for k in range(size):
            print(k)
            bboxes = bboxfull[k]

            for a in range(36):
                for b in range(36):
                    Iou[k,a,b] = getIou(bboxes[a], bboxes[b])
        np.save(url+i+j+savename, Iou)

