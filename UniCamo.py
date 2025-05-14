from carla_nr import carla_nr,carla_nr_vis
import sys
sys.path.append('/data/Newdisk/hanye/code/proben_attack_car/xk_fca/xk_fca')
import torch
import numpy as np
from gsam import GSAM, LinearScheduler
import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import detectron2.data.transforms as T
import torch.nn.functional as F
from multiprocessing.pool import Pool
from cfg_dig import cfgs,cfg1,cfg2,cfg3,cfg4,cfg5,cfg6,cfg7,cfg8
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
def cal_rpn_loss(rpn_score,rpn_boxes,gt_bbox):
    rpn_idx = get_match_idx(rpn_boxes, gt_bbox)
    rpn_score_out = rpn_score[rpn_idx]
    loss_rpn = F.binary_cross_entropy_with_logits(rpn_score_out,
                                                   torch.zeros_like(rpn_score_out).to(rpn_score_out.device),
                                                   reduction="sum")
    return loss_rpn
def cal_score_loss(score,idx):
    output = score[idx.detach().cpu().numpy().tolist()]

    # output_t= output[torch.argmax(output, dim=1) == 0]
    if len(output) > 0:
        target = torch.zeros(output.shape[0]).to(torch.int64).to(output.device)

        loss_score=-nn.CrossEntropyLoss()(output, target)
        # print(loss_score)
    else:
        loss_score = None
    return loss_score
def cal_IOU(bbox1,bbox2):
    x1,y1=max(bbox1[0],bbox2[0]),max(bbox1[1],bbox2[1])
    x2,y2=min(bbox1[2],bbox2[2]),min(bbox1[3],bbox2[3])
    inter=(x2-x1)*(y2-y1)
    iou=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])+(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])-inter
    return iou/inter
def get_match_idx(rpn_boxes,gt_box):
    idx=[]
    for i,box in enumerate(rpn_boxes):
        if cal_IOU(box,gt_box).item()>0.3:
            idx.append(i)
    return idx
def CAL_LOSS(image_name,veh_trans,cam_trans, cfg, adv_infrared, adv_vis,rimg_infrared, rimg_vis,models):
    adv_img_vis, img_vis, adv_img_infrared, img_infrared,original_img_vis,gt_box = generate_adv_img(image_name,veh_trans ,cam_trans, cfg, adv_infrared, adv_vis,
                                                                            rimg_infrared, rimg_vis)
    loss=cal_loss(adv_img_vis, adv_img_infrared,models,cfg,original_img_vis,gt_bbox)
    return loss
def cal_loss(adv_img_vis,adv_img_infrared,models,cfg,original_img_vis,gt_bbox):
    for k in range(3):
        if k == 0:
            input_adv=adv_img_infrared.squeeze(0).mul(255)
            _, score, idx,rpn_score,rpn_boxes = models[k](input_adv)
            loss_rpn1=cal_rpn_loss(rpn_score, rpn_boxes, gt_bbox,k)
            if len(idx[0]) > 0:
                loss_score1=cal_score_loss(score,idx)
        elif k == 1:
            input_adv = torch.cat((adv_img_vis.squeeze(0), adv_img_infrared[0][0].unsqueeze(0)), axis=0).mul(255)
            _, score, idx,rpn_score,rpn_boxes = models[k](input_adv)
            loss_rpn2=cal_rpn_loss(rpn_score, rpn_boxes, gt_bbox,k)
            if len(idx[0]) > 0:
                loss_score2=cal_score_loss(score,idx)
        elif k == 2:
            input_adv = torch.cat((adv_img_vis.squeeze(0), adv_img_infrared.squeeze(0)), axis=0).mul(255)
            _, score, idx,rpn_score,rpn_boxes = models[k](input_adv)
            loss_rpn3=cal_rpn_loss(rpn_score, rpn_boxes, gt_bbox,k)
            if len(idx[0]) > 0:
                loss_score3=cal_score_loss(score,idx)
    loss_smooth_vis = cfg.smooth(original_img_vis) * cfg.loss_smooth_weight
    loss_smooth = loss_smooth_vis
    loss_rpn=0.2*loss_rpn1+0.4*loss_rpn2+0.4*loss_rpn3
    loss_score=0.2*loss_score1+0.4*loss_score2+0.4*loss_score3
    return loss_smooth,loss_rpn,loss_score
def generate_adv_img(image_name,veh_trans ,cam_trans,cfg,adv_infrared,adv_vis,rimg_infrared,rimg_vis,cpu_index,cnt,ouler_angle):
    with open(cfg.labes_path_infrared + str(image_name[:-4] + ".txt"), "r") as f:
        for fi in f:
            if int(fi.split(" ")[0]) == 0:
                w = int(float(fi.split(" ")[1]) * 640)
                h = int(float(fi.split(" ")[2]) * 512)
    box=[int(float(fi.split(" ")[1]) * 640),int(float(fi.split(" ")[2]) * 512),int(float(fi.split(" ")[3]) * 640),int(float(fi.split(" ")[4]) * 512)]
    gt_box=torch.tensor([box[0]-box[2]/2,box[1]-box[3]/2,box[0]+box[2]/2,box[1]+box[3]/2]).to(torch.float32).to(adv_infrared.device)
    img_infrared = cfg.read_image(cfg.image_path_infrared + image_name).cuda()
    veh_trans_infrared=veh_trans
    cam_trans_infrared=cam_trans

    box_infrared = [w, h, 0, 0]
    adv_img_infrared, original_img_infrared, flag_infra = rimg_infrared(img_infrared.clone(), cam_trans_infrared,
                                                                        veh_trans_infrared,
                                                                        cfg.tanh_infrared(adv_infrared), box_infrared,
                                                                        640,cpu_index,cnt,ouler_angle)

    # adv_img_infrared1, original_img_infrared1,flag_infra1 = rimg_infrared(img_infrared.clone(), cam_trans, veh_trans,
    #                                                         cfg.tanh(adv_infrared), box,
    #                                                         640)

    with open(cfg.labes_path_vis + str(image_name[:-4] + ".txt"), "r") as f:
        for fi in f:
            if int(fi.split(" ")[0]) == 0:
                w = int(float(fi.split(" ")[1]) * 1280)
                h = int(float(fi.split(" ")[2]) * 1024)

    img_vis = cfg.read_image(cfg.image_path_vis + image_name).cuda()
    img_vis = F.interpolate(img_vis, scale_factor=2, mode='nearest')
    veh_trans_vis = veh_trans
    cam_trans_vis = cam_trans
    box_vis = [w, h, 0, 0]
    adv_img_vis_ori_size, original_img_vis, flag_vis = rimg_vis(img_vis.clone(), cam_trans_vis, veh_trans_vis,
                                                                cfg.tanh(adv_vis), box_vis, 1280,cpu_index,cnt,cfg.tanh_infrared(adv_infrared),ouler_angle)
    adv_img_vis = F.interpolate(adv_img_vis_ori_size, scale_factor=0.5, mode='nearest')
    img_vis = F.interpolate(img_vis, scale_factor=0.5, mode='nearest')  # mode试一下别的
    return adv_img_vis,img_vis,adv_img_infrared,img_infrared,original_img_vis,gt_box

def load_model(method):
    print("method:",method)
    cfg = get_cfg()
    cfg.merge_from_file("/data/Newdisk/hanye/code/proben/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LOGITS = True
    cfg.MODEL.ROI_BOX_HEAD.DROP_OUT = True
    cfg.MODEL.BACKBONE.FREEZE_AT = 3
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.ENABLE_GAUSSIANNLLOSS = True
    if method == 'thermal_only':
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.WEIGHTS='/data/Newdisk/hanye/code/proben/demo/Tank/realworld_dataset_weight/out_model_thermal_only_best.pth'
    elif method == 'early_fusion':
        cfg.INPUT.FORMAT = 'BGRT'
        cfg.INPUT.NUM_IN_CHANNELS = 4
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
        cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
        cfg.MODEL.WEIGHTS='/data/Newdisk/hanye/code/proben/demo/Tank/realworld_dataset_weight/out_model_early_fusion_best.pth'
    elif method == 'middle_fusion':
        cfg.INPUT.FORMAT = 'BGRTTT'
        cfg.INPUT.NUM_IN_CHANNELS = 6
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
        cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        cfg.MODEL.WEIGHTS='/data/Newdisk/hanye/code/proben/demo/Tank/realworld_dataset_weight/out_model_middle_fusion_best.pth'
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    predictor=DefaultPredictor(cfg)
    predictor.model.eval()
    for m in predictor.model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    predictor.model.eval()
    return predictor,transform_gen
def four_color(adv_t,group_idx):
    max = adv_t.max()
    min = adv_t.min()
    len = max - min
    interval = []
    for i in range(3):
        step = len / 4
        interval.append(min + (i + 1) * step)
    for idx in group_idx:
        element = adv_t[:, idx][0][0][0][0][0]
        if element < interval[0]:
            temp = torch.zeros_like(adv_t[:, idx]).cuda()
            adv_t[:, idx] = torch.where(adv_t[:, idx] < interval[0], temp, adv_t[:, idx])
        elif element >= interval[0] and element < interval[1]:
            temp = torch.ones_like(adv_t[:, idx]).cuda() * 0.25
            adv_t[:, idx] = torch.where(adv_t[:, idx] < interval[1], temp, adv_t[:, idx])
        elif element >= interval[1] and element < interval[2]:
            temp = torch.ones_like(adv_t[:, idx]).cuda() * 0.50
            adv_t[:, idx] = torch.where(adv_t[:, idx] < interval[2], temp, adv_t[:, idx])
        elif element >= interval[2]:
            temp = torch.ones_like(adv_t[:, idx]).cuda() * 0.75
            adv_t[:, idx] = torch.where(adv_t[:, idx] >= interval[2], temp, adv_t[:, idx])
    return adv_t

def attack(cfg,cpu_sum,cpu_index,cuda,epoch_i):

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(int(cuda))
    # device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    info=[]
    with open(cfg.images_parameter,"r") as f:
        for i in f:
            data=i.split('\n')[0].split(' ')
            data[0]=data[0].replace('BMP','bmp')
            label_name=data[0].replace("bmp","txt")
            if os.path.exists(os.path.join("/data/Newdisk/hanye/code/proben_attack_person/Datasets/LLVIP/labels/",label_name)):
                info.append(data)
    print("The number of images is {}".format(len(info)))


    #初始化neural render
    rimg_infrared=carla_nr(cfg.model_obj,cfg.model_face,texture_size=cfg.texture_size)
    rimg_vis=carla_nr_vis(cfg.model_obj,cfg.model_face,texture_size=cfg.texture_size)





    if epoch_i==1:
        adv_infrared = cfgs.atanh(rimg_infrared.get_start_textures().cuda())
        adv_vis = cfgs.atanh(torch.rand(rimg_vis.get_start_textures().shape).cuda()).mul(100)
        adv_infrared=cfgs.zhuanhuan_infrared(adv_infrared,rimg_infrared.list1,rimg_infrared.list2,rimg_infrared.unit)
        adv_vis = cfgs.merge_face(adv_vis, rimg_vis.get_face_list(), cfgs.merge_face_num)


    else:
        adv_infrared=torch.tensor(np.load("infrared_lossout_dig.npy")).cuda().mean(dim=5,keepdim=True)
        adv_vis=torch.tensor(np.load("vis_lossout_dig.npy")).mul(100).cuda()

    adv_infrared.requires_grad=True
    adv_vis.requires_grad = True


    optimizer = torch.optim.SGD([adv_infrared,adv_vis], lr=cfg.optimizer_learn)
    # lr_scheduler = LinearScheduler(T_max=40 * len(info), max_value=0.1,
    #                                min_value=0.1 * 0.01, optimizer=optimizer)
    # rho_scheduler = LinearScheduler(T_max=40 * len(info), max_value=1,
    #                                 min_value=0.1)
    # gsam_optimizer = GSAM(params=[adv_infrared,adv_vis],base_optimizer=optimizer,gsam_alpha=0.4,rho_scheduler=rho_scheduler)


    #加载模型
    # yolov5s_model_infrared = DetectMultiBackend(cfg.yolov5s_parameter_infrared, device=device)
    # yolov5s_model_vis = DetectMultiBackend(cfg.yolov5s_parameter_infrared, device=device)
    method_names=['thermal_only','early_fusion','middle_fusion']
    models=[]
    for name in method_names:
        predictor,transform_gen=load_model(name)
        models.append(predictor)
    loss_total=0
    for epoch in range(1):

        # cfg.save_textures(adv_t,cfg.save_textures_path)
        cnt=0
        for index,i in enumerate(info):
            # print(i)
            if index<len(info)*(cpu_index) //cpu_sum or index>len(info)*(cpu_index+1) //cpu_sum:
                continue
            # adv_img_vis,img_vis,adv_img_infrared,img_infrared,original_img_vis=generate_adv_img(i, cfg, adv_infrared, adv_vis,rimg_infrared,rimg_vis)

            try:
                cnt += 1

                image_name=i[0]
                veh_trans = [[float(i[1]), float(i[2]), float(i[3])], [float(i[4]), float(i[5]), float(i[6])]]
                cam_trans = [[float(i[7]), float(i[8]), float(i[9])],
                                      [float(i[10]), float(i[11]), float(i[12])]]
                str="0"
                for l in range(1,13):
                    str += ","+i[l]
                ouler_angle=str
                print("starting generating...")
                adv_img_vis, img_vis, adv_img_infrared, img_infrared, original_img_vis,gt_bbox = generate_adv_img(image_name,
                                                                                                          veh_trans,
                                                                                                          cam_trans,
                                                                                                          cfg,
                                                                                                          adv_infrared,
                                                                                                          adv_vis,
                                                                                                          rimg_infrared,
                                                                                                          rimg_vis,cpu_index,cnt,ouler_angle)
                print("finishing generate adversaral....")


                if cfg.save_image_flag:
                    os.makedirs(cfg.save_image_path_vis, exist_ok=True)
                    cv_img = adv_img_vis.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    cv2.imwrite(cfg.save_image_path_vis + image_name, cv_img * 255)

                    os.makedirs(cfg.save_image_path_infrared, exist_ok=True)
                    cv_img = adv_img_infrared.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    cv2.imwrite(cfg.save_image_path_infrared + image_name, cv_img * 255)

                    os.makedirs(cfg.save_image_path_ori_infrared, exist_ok=True)
                    cv_img = img_infrared.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    cv2.imwrite(cfg.save_image_path_ori_infrared + image_name, cv_img * 255)

                    os.makedirs(cfg.save_image_path_ori_vis, exist_ok=True)
                    cv_img = img_vis.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    cv2.imwrite(cfg.save_image_path_ori_vis + image_name, cv_img * 255)

                print("starting calculating...")
                for k in range(3):
                    if k == 0:
                        input_adv = adv_img_infrared.squeeze(0).mul(255)
                        _, feature,score, idx,  = models[k](input_adv)
                        # loss_rpn1 = cal_rpn_loss(rpn_score, rpn_boxes, gt_bbox)
                        loss_score1 = cal_score_loss(score, idx)
                        print("finishing 1")
                    elif k == 1:
                        input_adv = torch.cat((adv_img_vis.squeeze(0), adv_img_infrared[0][0].unsqueeze(0)),axis=0).mul(255)
                        _, feature,score, idx,  = models[k](input_adv)
                        # loss_rpn2 = cal_rpn_loss(rpn_score, rpn_boxes, gt_bbox)
                        loss_score2 = cal_score_loss(score, idx)
                        print("finishing 2")
                    elif k == 2:
                        input_adv = torch.cat((adv_img_vis.squeeze(0), adv_img_infrared.squeeze(0)), axis=0).mul(255)
                        _, feature,score, idx  = models[k](input_adv)
                        # loss_rpn3 = cal_rpn_loss(rpn_score, rpn_boxes, gt_bbox)
                        loss_score3 = cal_score_loss(score,idx)
                        print("finishing 3")
                loss_smooth_vis = cfg.smooth(original_img_vis) * cfg.loss_smooth_weight
                loss_smooth = loss_smooth_vis
                # loss_rpn = 0.2 * loss_rpn1 + 0.4 * loss_rpn2 + 0.4 * loss_rpn3
                if loss_score1 is not None and loss_score2 is not None and loss_score3 is not None:
                    loss_score=None
                else:
                    first=True
                    if loss_score1 is not None:
                        loss_score=0.2*loss_score1
                        first=False
                    if loss_score2 is not None:
                        if first==True:
                            loss_score=0.4*loss_score2
                            first=False
                        else:
                            loss_score += 0.4 * loss_score2
                    if loss_score3 is not None:
                        if first==True:
                            loss_score=0.4*loss_score3
                            first = False
                        else:
                            loss_score += 0.4 * loss_score3

                if loss_score is not None:
                    loss_score=loss_score.mul(200)
                    loss=loss_smooth+loss_score
                else:
                    loss = loss_smooth
                print("loss:",loss)
                loss.requires_grad_(True)

                loss.backward()
                # print("finished backward ...")
                adv_infrared.grad.data = cfg.zhuanhuan_infrared(adv_infrared.grad.data, rimg_infrared.list1,
                                                                rimg_infrared.list2, rimg_infrared.unit)

                adv_vis.grad.data = cfg.merge_face(adv_vis.grad.data, rimg_vis.get_face_list(), cfg.merge_face_num)
                optimizer.step()
                print("1111")
                print("epoch:{},images:{},loss:{},loss_smooth:{},loss_score:{}".format(epoch_i, i[0], loss,loss_smooth,loss_score))
            except:
                with open("fail_name_list.txt","a") as f:
                    f.write(i[0])

            # loss_total += loss
            # print("epach:{},images:{},loss_pred:{},loss_smooth:{}".format(epoch, i[0], loss_pred,loss_smooth))





            #


    np.save('infrared_lossout_dig_{}.npy'.format(cpu_index),adv_infrared.detach().cpu().numpy(),
            allow_pickle=True)
    np.save('vis_lossout_dig_{}.npy'.format(cpu_index), adv_vis.detach().cpu().numpy(), allow_pickle=True)
    # with open("/home/Newdisk2/hanye/myproject/proben_attack/infrared_camou/name_list_"+str(cpu_index)+".txt","w") as f:
    #     for name in name_list:
    #         f.write(name+"\n")
    #     f.close()
    # if epoch_i==1:
    #     with open(
    #             "/home/Newdisk2/hanye/myproject/proben_attack/infrared_camou/features_lossout_" + str(cpu_index) + ".txt",
    #             "w") as f:
    #         f.write(str(loss_total / cnt) + "\n")
    #         f.close()
    # else:
    #     with open("/home/Newdisk2/hanye/myproject/proben_attack/infrared_camou/features_lossout_"+str(cpu_index)+".txt","a+") as f:
    #         f.write(str(loss_total/cnt)+"\n")
    #         f.close()



if __name__=="__main__":

    #Pool = Pool(cpu_num)
    mp.set_start_method('spawn')
    gpu_num = 8
    cfg_list=[cfg1,cfg2,cfg3,cfg4,cfg5,cfg6,cfg7,cfg8]


    for i in range(40):
        if i==0:
            continue

        processes = []
        pool_list=[]
        for gpu_i in range(gpu_num):
            pool_list.append((cfg_list[gpu_i], gpu_num, gpu_i, str(gpu_i),i))
        for i in range(gpu_num):
            p=mp.Process(target=attack,args=(pool_list[i]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # pool = Pool(cpu_num)
        # pool.map(attack,pool_list)
        # pool.close()
        # pool.join()
        adv_texture_i=None
        adv_texture_v = None
        flag_tex=0
        for gpu_i in range(gpu_num):
            if flag_tex==0:
                adv_texture_i=(np.load('infrared_lossout_dig_{}.npy'.format(gpu_i)))
                adv_texture_v = (np.load('vis_lossout_dig_{}.npy'.format(gpu_i)))
                flag_tex=1
            else:
                adv_texture_i+=(np.load('infrared_lossout_dig_{}.npy'.format(gpu_i)))
                adv_texture_v += (np.load('vis_lossout_dig_{}.npy'.format(gpu_i)))

        np.save("infrared_lossout_dig.npy",adv_texture_i/gpu_num)
        np.save("vis_lossout_dig.npy", adv_texture_v / gpu_num)





