from carla_nr import carla_nr,carla_nr_vis
import os
from cfg import cfgs
import torch
import cv2
import numpy as np
import torch.nn.functional as F
texture_size=2
img_path_vis="/home/Newdisk2/hanye/myproject/proben/Datasets/Tank_select/RGB/"
img_path_infrared="/home/Newdisk2/hanye/myproject/proben/Datasets/Tank_select/thermal_8_bit/"
model_obj="/home/Newdisk2/hanye/myproject/proben_attack/infrared_camou/tank04.obj"
model_face="/home/Newdisk2/hanye/myproject/proben_attack/infrared_camou/tank04_face.txt"
save_image_path_vis="./renderer/vis/"
save_image_path_infrared="./renderer/infrared/"
images_parameter="/home/Newdisk2/hanye/myproject/proben_attack/infrared_camou/sd_470.txt"
label_path="/home/Newdisk2/hanye/myproject/proben/Datasets/Tank_select/labels_all_infrared/"
#读取欧拉角信息
name_list=os.listdir(img_path_vis)
info=[]
# with open(images_parameter, "r") as f:
#     for i in f:
#         data = i.split('\n')[0].split(' ')[0].split('\t')
#         data[0] = data[0].replace('BMP', 'bmp')
#         if data[0] in name_list:
#             info.append(data)
rimg_infrared=carla_nr(model_obj,model_face,texture_size=texture_size)
rimg_vis=carla_nr_vis(model_obj,model_face,texture_size=texture_size)
adv_infrared=torch.tensor(np.load("infrared_lossout.npy")).cuda().mean(dim=5,keepdim=True)
adv_vis=torch.tensor(np.load("vis_lossout.npy")).cuda()
i_vis=["03_50_0_0.9_-1.2.bmp",0,	0,	0, 0,	-90,90,	-4.63,	-0.31,	5.49,		-50,	0,	0]
i_infrared=["03_50_0_0.9_-1.2.bmp",0,	0,	0, 0,	-90,90,	-4.63,	-0.31,	5.49,		-50,	0,	0]

with open(label_path + str(i_infrared[0][:-4] + ".txt"), "r") as f:
    for fi in f:
        if int(fi.split(" ")[0]) == 0:
            w = int(float(fi.split(" ")[1]) * 640)
            h = int(float(fi.split(" ")[2]) * 512)

img_infrared = cfgs.read_image(img_path_infrared + i_infrared[0]).cuda()
veh_trans = [[float(i_infrared[1]), float(i_infrared[2]), float(i_infrared[3])], [float(i_infrared[4]), float(i_infrared[5]), float(i_infrared[6])]]
cam_trans = [[float(i_infrared[7]), float(i_infrared[8]), float(i_infrared[9])], [float(i_infrared[10]), float(i_infrared[11]), float(i_infrared[12])]]
box = [w, h, 0, 0]

adv_img_infrared, original_img_infrared, flag_infra = rimg_infrared(img_infrared.clone(), cam_trans, veh_trans,
                                                                    cfgs.tanh_infrared(adv_infrared), box, 640)
with open(label_path + str(i_vis[0][:-4] + ".txt"), "r") as f:
    for fi in f:
        if int(fi.split(" ")[0]) == 0:
            w = int(float(fi.split(" ")[1]) * 1280)
            h = int(float(fi.split(" ")[2]) * 1024)

img_vis = cfgs.read_image(img_path_vis + i_vis[0]).cuda()
img_vis = F.interpolate(img_vis, scale_factor=2, mode='nearest')
veh_trans = [[float(i_vis[1]), float(i_vis[2]), float(i_vis[3])], [float(i_vis[4]), float(i_vis[5]), float(i_vis[6])]]
cam_trans = [[float(i_vis[7]), float(i_vis[8]), float(i_vis[9])], [float(i_vis[10]), float(i_vis[11]), float(i_vis[12])]]
box = [w, h, 0, 0]
adv_img_vis_ori_size, original_img_vis, flag_vis = rimg_vis(img_vis.clone(), cam_trans, veh_trans,
                                                            cfgs.tanh(adv_vis), box, 1280)

os.makedirs(save_image_path_vis, exist_ok=True)
cv_img = adv_img_vis_ori_size.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
cv2.imwrite("renderer_result_vis.jpg", cv_img * 255)

os.makedirs(save_image_path_infrared, exist_ok=True)
cv_img = adv_img_infrared.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
cv2.imwrite("renderer_result_infrared.jpg", cv_img * 255)
