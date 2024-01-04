import numpy as np

path = "/media/zetlin/Data2/waymo_dataset/v1_1/training/data/scid_1a23a32935238e99__aid_546__atype_1.npz"
data = dict(np.load(path, allow_pickle=True))
print(data)