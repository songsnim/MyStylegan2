# gpu = 2
# iter = 800000
# start_iter = 0
# batch = 32
# latent = 32
# img_size = 64
# center_crop = 50
# lr = 0.002
# train_path = "datasets/CelebA/train"
# test_path = "datasets/CelebA/test/"

# py = False
# ipynb = True
# augment_p = 0
# ada_target = 0.6
# ada_length = 500*1000
# augment = True
# d_reg_every = 16
# g_reg_every = 16
# r1 = 10
# path_regularize = 2
# path_batch_shrink = 2

# ckpt = None
# wandb = False
# description = 'first try for refactoring'


"""
stylespaces dimensions
128 x 9
64 x 3 
32 x 2
"""

gpu_num = 2  # gpu(A100) index
iter = 800000  # total iteration
start_iter = 0
batch = 32
latent = 64
img_size = 64
center_crop = 50
lr = 0.002
train_path = "LMDB/CelebA/train"
test_path = "LMDB/CelebA/test"

mixing = 0.9
disc_latent_ratio = 0.3
augment_p = 0
ada_target = 0.6
ada_length = 500*1000
augment = True
kl_pred_loss = False
d_reg_every = 16
g_reg_every = 16
r1 = 10
path_regularize = 2
path_batch_shrink = 2

py = False
ipynb = True
description = 'dev input_noise debug'
ckpt = None
wandb = True
