from scipy.spatial.distance import directed_hausdorff
import numpy as np
from chamfer_distance import ChamferDistance
import torch

with open("scan_001.xyz","r") as fid:
    pointcloud_01 = fid.read()

pointcloud_01 = np.array([[float(p) for p in pc.split()] for pc in pointcloud_01.strip("\n").split("\n")])

with open("scan_002.xyz","r") as fid:
    pointcloud_02 = fid.read()
    
pointcloud_02 = np.array([[float(p) for p in pc.split()] for pc in pointcloud_02.strip("\n").split("\n")])
   
# d, _, _ = directed_hausdorff(pointcloud_01, pointcloud_01)
# print("Hausdorff distance between PC1 and itself", d)
# d, _, _ = directed_hausdorff(pointcloud_01, pointcloud_02)
# print("Hausdorff distance between PC1 and PC2", d)

def mean_surface_distance(x, y):    
    msd = (np.sum([directed_hausdorff(np.expand_dims(x[idx, :], axis=0), y)[0] for idx in range(np.shape(x)[0])])
    + np.sum([directed_hausdorff(np.expand_dims(y[idx, :], axis=0), x)[0] for idx in range(np.shape(y)[0])])) / (np.shape(x)[0] + np.shape(y)[0])
    return msd

def residual_mean_surface_distance(x, y):    
    rmsd = (np.sum([directed_hausdorff(np.expand_dims(x[idx, :], axis=0), y)[0]**2 for idx in range(np.shape(x)[0])])
    + np.sum([directed_hausdorff(np.expand_dims(y[idx, :], axis=0), x)[0]**2 for idx in range(np.shape(y)[0])])) / (np.shape(x)[0] + np.shape(y)[0])
    return rmsd


# d = mean_surface_distance(pointcloud_01, pointcloud_01)
# print("Mean surface distance between PC1 and itself", d)
# d = mean_surface_distance(pointcloud_01, pointcloud_02)
# print("Mean surface distance between PC1 and PC2", d)


# d = residual_mean_surface_distance(pointcloud_01, pointcloud_01)
# print("Residual mean surface distance between PC1 and itself", d)
# d = residual_mean_surface_distance(pointcloud_01, pointcloud_02)
# print("Residual mean surface distance between PC1 and PC2", d)

# chamfer_dist = ChamferDistance()
# print(torch.unsqueeze(torch.tensor(pointcloud_01, dtype=torch.float32), dim=0))
# d = torch.sum(chamfer_dist(torch.unsqueeze(torch.tensor(pointcloud_01, dtype=torch.float32), dim=0),
#                  torch.unsqueeze(torch.tensor(pointcloud_01, dtype=torch.float32), dim=0)))
# print("Chamfer distance between PC1 and itself", d)
# d = torch.sum(chamfer_dist(torch.unsqueeze(torch.tensor(pointcloud_01, dtype=torch.float32), dim=0),
#                  torch.unsqueeze(torch.tensor(pointcloud_02, dtype=torch.float32), dim=0)))
# print("Chamfer distance between PC1 and PC2", d)

def f_score(gt, pred, radius):
    tp = 0
    fp = 0
    fn = 0
    for p in pred:
        detected = False
        for g in gt:
            if np.linalg.norm(p - g) < radius:
                tp += 1
                detected = True
                break
        if not detected:
            fp += 1
    
    for g in gt:
        detected = False
        for p in pred:
            if np.linalg.norm(p - g) < radius:
                detected = True
                break
        if not detected:
            fn += 1
    return tp/(tp + 0.5*(fp + fn))

f_score(pointcloud_01, pointcloud_01, 0.1)
f_score(pointcloud_01, pointcloud_02, 0.1)
f_score(pointcloud_01, pointcloud_02, 0.5)
f_score(pointcloud_01, pointcloud_02, 1.0)
    
            
                
    
    