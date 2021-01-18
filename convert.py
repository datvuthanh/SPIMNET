import numpy as np 

with open('/home/memi/Desktop/SPIMNET/new_costspace/cost_16x16_220951_003373_2x.npy', 'rb') as f:
  cost_volume = np.load(f)

left_cost_volume = np.transpose(cost_volume, (2, 0, 1))

left_cost_volume.shape
ndisp = left_cost_volume.shape[0]
height = left_cost_volume.shape[1]
width = left_cost_volume.shape[2]

right_cost_volume = np.zeros([ndisp, height, width], dtype=np.float32)

for d in range(ndisp):
    right_cost_volume[d, :, :width-d] = left_cost_volume[d, :, d:]

left_cost_volume = np.transpose(left_cost_volume, (1, 2, 0))
right_cost_volume = np.transpose(right_cost_volume, (1, 2, 0))

print(right_cost_volume.shape)
with open('/home/memi/Desktop/SPIMNET/costspace_for_torch/220951_003373/left_003373.npy', 'wb') as f:
    np.save(f, left_cost_volume)

with open('/home/memi/Desktop/SPIMNET/costspace_for_torch/220951_003373/right_003373.npy', 'wb') as f:
    np.save(f, right_cost_volume)