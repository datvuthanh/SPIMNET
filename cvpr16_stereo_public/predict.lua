npy4th = require 'npy4th'
require 'xlua'
require 'optim'
require 'cunn'
require 'image'
require 'gnuplot'
require 'smooth'
require 'colormap'

local c = require 'trepl.colorize'

local left_path = '/home/memi/Desktop/SPIMNET/test_dataset/20170222_0951/RGBResize/220951_043015_RGBResize.png'
local right_path = '/home/memi/Desktop/SPIMNET/test_dataset/20170222_0951/NIRResize/220951_043015_NIRResize.png'
local result_path = '/home/memi/Desktop/SPIMNET/results_torch/220951_043015'

left_cost_volume = npy4th.loadnpy('/home/memi/Desktop/SPIMNET/costspace_for_torch/220951_043015/left_043015.npy')
right_cost_volume = npy4th.loadnpy('/home/memi/Desktop/SPIMNET/costspace_for_torch/220951_043015/right_043015.npy')

torch.manualSeed(123)
cutorch.setDevice(1)


local l_fn = string.format(left_path)
local r_fn = string.format(right_path)

local l_img = image.load(l_fn,3)
l_img = image.scale(l_img, 1164, 858)
image.save(string.format('%s/original/left.png',result_path), l_img)
l_img = l_img:byte():cuda()

-- local l_img = image.load(l_fn, 3, 'byte'):cuda()
local r_img = image.load(r_fn,3)
r_img = image.scale(r_img,1164,858)
image.save(string.format('%s/original/right.png',result_path), r_img)
r_img = r_img:byte():cuda()

local img_h = l_img:size(2)
local img_w = l_img:size(3)

print('image size: ' .. img_h .. ' x ' .. img_w)

local total_loc = 52
local disp_range = 52
unary_vol = torch.CudaTensor(img_h, img_w, total_loc):zero()
right_unary_vol = torch.CudaTensor(img_h, img_w, total_loc):zero()

unary_vol = left_cost_volume
right_unary_vol = right_cost_volume

---Test image
-- local max_disparity = 52
-- local scale_factor = 255 / (max_disparity - 1) 

-- lu = unary_vol:view(1,img_h,img_w,disp_range):permute(1,4,2,3):clone()
-- _,pred = lu:max(2) 
-- pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
-- image.save(string.format('test.png'), pred:byte())
------

paths.mkdir('unary_img')
_,pred = unary_vol:max(3)
pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]

image.save(string.format('%s/left_without_CA.png',result_path), pred:byte())

lu = unary_vol:view(1,img_h,img_w,disp_range):permute(1,4,2,3):clone()
ru = right_unary_vol:view(1,img_h,img_w,disp_range):permute(1,4,2,3):clone()


--- cost agg
print('cost agg..')
local tic = torch.tic()
lu,ru = smooth.nyu.cross_agg(l_img:view(1,3,img_h,img_w), r_img:view(1,3,img_h,img_w), lu, ru, 2)
print('cost agg tmr.. ' .. torch.toc(tic))

_,pred = lu:max(2)
pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
image.save(string.format('%s/left_cost_img.png',result_path), pred:byte())

_,pred = ru:max(2)
pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
image.save(string.format('%s/right_cost_img.png',result_path), pred:byte())
print('writing cost image done..')

-- SGM

print('SGM..')
local tic = torch.tic()

lu:mul(-1)
ru:mul(-1)

lu = smooth.nyu.sgm(l_img, r_img, lu, -1)
ru = smooth.nyu.sgm(l_img, r_img, ru, 1)
print('SGM tmr.. ' .. torch.toc(tic))

_,pred = lu:min(2)
pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
image.save(string.format('%s/left_sgm_img.png',result_path), pred:byte())

_,pred = ru:min(2)
pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
image.save(string.format('%s/right_sgm_img.png',result_path), pred:byte())
-- lu: 1 x disp x h x w
print('writing SGM image done..')

-- cost agg 2

print('cost agg 2..')
local tic = torch.tic()
lu,ru = smooth.nyu.cross_agg(l_img:view(1,3,img_h,img_w), r_img:view(1,3,img_h,img_w), lu, ru, 16)
print('cost agg tmr.. ' .. torch.toc(tic))

_,pred = lu:min(2)
pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
image.save(string.format('%s/left_cost_img_2.png',result_path), pred:byte())

_,pred = ru:min(2)
pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
image.save(string.format('%s/right_cost_img_2.png',result_path), pred:byte())


-- more nyu postprocess

disp = {}
_, pred = lu:min(2)
disp[1] = (pred - 1):cuda()
_, pred = ru:min(2)
disp[2] = (pred - 1):cuda()

print('POST processing..')
local tic = torch.tic()
final_pred, outlier = smooth.nyu.post(disp, lu)
print('post tmr.. ' .. torch.toc(tic))

image.save(string.format('%s/final_disparitymap.png',result_path), final_pred:view(img_h, img_w):byte())

-- image.save(string.format('outlier.png'), (outlier*51):view(img_h, img_w):byte())

print('writing post image done..')