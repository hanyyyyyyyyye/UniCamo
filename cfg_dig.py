import numpy as np
import torch
import detectron2.data.transforms as T
import cv2
import torch.nn as nn
import torch


class cfg():
    labes_path_infrared = "/data/Newdisk/hanye/code/proben_attack_person/Datasets/LLVIP/labels/"  # 所有图片的真实框
    labes_path_vis = "/data/Newdisk/hanye/code/proben_attack_person/Datasets/LLVIP/labels/"
    # images_parameter="/home/Newdisk2/hanye/myproject/proben/Datasets/Tank_select/sd_filter_select.txt" #图片的渲染参数
    images_parameter = "/data/Newdisk/hanye/code/proben_attack_car/infrared_camou/new_sd.txt"

    model_obj = "audi_et_te.obj"
    model_face = "exterior_face.txt"
    texture_size = 2
    save_textures_path = "infrared_zjut_1.npy"  # 保存纹理的位置
    load_textures_flag = False  # 是否读取纹理
    load_textures_path = "gf_11_16_color_smooth.npy"  # 读取纹理的位置
    save_image_flag = True  # 是否保存图片
    save_image_path_infrared = "/data/Newdisk/hanye/code/proben_attack_car/result_image/digital/car/infrared_all/"  # 保存图片的文件路径
    save_image_path_infrared_p = "/data/Newdisk/hanye/code/proben_attack_person/result_image/infrared_all_p/"  # 保存图片的文件路径
    save_image_path_vis = "/data/Newdisk/hanye/code/proben_attack_car/result_image/digital/car/vis_all/"
    save_image_path_ori_infrared = "/data/Newdisk/hanye/code/proben_attack_person/result_image/ori_infrared_lossout2_1/"
    save_image_path_ori_vis = "/data/Newdisk/hanye/code/proben_attack_person/result_image/ori_vis_lossout2_1/"  # 保存图片的文件路径
    save_ori_image = '/home/Newdisk/yanyunjie/code_practics/patch/yolov5-master/ori_image/'
    yolov5s_parameter_infrared = "/data/Newdisk/hanye/code/proben_attack/infrared_camou/infrared.pt"  # yolov5的权重路径
    yolov5s_parameter_vis = '/data/Newdisk/hanye/code/proben_attack/yolov5-master/yolov5s_xk.pt'
    epoch = 10
    merge_face_num =  1
    image_path_infrared = "/data/Newdisk/hanye/code/proben_attack_person/Datasets/LLVIP/thermal_8_bit/"
    image_path_vis = "/data/Newdisk/hanye/code/proben_attack_person/Datasets/LLVIP/RGB/"
    image_path_infrared_bg = "/data/Newdisk/hanye/code/proben/Datasets/thermal_bg/"
    image_path_rgb_bg = "/data/Newdisk/hanye/code/proben/Datasets/rgb_bg/"
    pre_conf = 0.3
    feature1_weight = 1000
    feature2_weight = 0.1
    loss_color_weight = 0.1
    loss_smooth_weight = 0.00001
    optimizer_learn = 0.1

    def atanh(self, data):
        return torch.atanh_((data * 2) - 1)

    def zhuanhuan_infrared(self, adv, list1, list2, unit):

        a1 = adv[:, list1, :, :, :, :]
        a2 = adv[:, list2, :, :, :, :]
        a1_shape = a1.shape
        a2_shape = a2.shape
        a1 = a1.view(1, len(list1) // unit, -1, 1).mean(dim=2, keepdim=True).repeat_interleave(unit * adv.shape[2] ** 3,
                                                                                               dim=2)
        a2 = a2.view(1, len(list2), -1, 1).mean(dim=2, keepdim=True).repeat_interleave(adv.shape[2] ** 3, dim=2)
        adv[:, list1, :, :, :, :] = a1.view(a1_shape)
        adv[:, list2, :, :, :, :] = a2.view(a2_shape)
        # a=adv.view(1,adv.shape[1],-1,1)
        # a=a.mean(dim=2, keepdim=True).repeat_interleave(adv.shape[2]**3, dim=2)
        return adv

    def merge_face(self, img, face_list, merge_size):

        middle_img = img[:, face_list, :, :, :].clone()
        middle_img_shape = middle_img.shape
        if middle_img.shape[1] % merge_size != 0:
            print('*********************************************************')
            print(middle_img.shape[1])
            print("无法整数合并")
            exit(0)
        middle_img = middle_img.view((img.shape[0], middle_img.shape[1] // merge_size, -1, 3))
        r = middle_img[:, :, :, [0]]
        g = middle_img[:, :, :, [1]]
        b = middle_img[:, :, :, [2]]
        r = r.mean(dim=2, keepdim=True).repeat_interleave(middle_img.shape[2], dim=2)
        g = g.mean(dim=2, keepdim=True).repeat_interleave(middle_img.shape[2], dim=2)
        b = b.mean(dim=2, keepdim=True).repeat_interleave(middle_img.shape[2], dim=2)
        middle_img[:, :, :, [0]] = r
        middle_img[:, :, :, [1]] = g
        middle_img[:, :, :, [2]] = b
        img[:, face_list, :, :, :] = middle_img.view(middle_img_shape)
        return img

    def nps(self, adv):
        adv1 = adv[0].view(-1, 3)
        color = torch.tensor(
            [[129, 127, 38], [24, 62, 12], [24, 63, 63],
             [117, 249, 77], [117, 250, 97],
             [55, 125, 34], [55, 126, 71], ]) / 255
        color = color.cuda()
        colorlist = []
        for i in range(len(color)):
            colorlist.append(color[i].unsqueeze(0).repeat_interleave(adv1.shape[0], dim=0))

        T = torch.abs(adv1 - colorlist[0]).sum(dim=1)
        for i in range(1, len(colorlist)):
            T = torch.min(T, torch.abs(adv1 - colorlist[i]).sum(dim=1))
        return T.sum()

    def smooth(self, img):
        mask1 = (img[:, :, 1:, :-1] != 0) * 1
        mask2 = (img[:, :, :-1, :-1] != 0) * 1
        maska = (mask1 == mask2) * 1
        mask3 = (img[:, :, :-1, 1:] != 0) * 1
        mask4 = (img[:, :, :-1, :-1] != 0) * 1
        maskb = (mask3 == mask4) * 1
        s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2) * maska
        s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2) * maskb

        return torch.sum(s1 + s2)

    def yolo_pre(self, pre):
        flag = ((pre[:, 5] > self.pre_conf) * 1).repeat(self.texture_size ** 3, 1).T
        return (flag * pre)[:, 4].sum()
        # return (flag * pre)[:, 4].mean()

    def cal_rpn_loss(self, rpn_score, rpn_boxes, gt_bbox):
        rpn_idx = get_match_idx(rpn_boxes, gt_bbox)
        rpn_score_out = rpn_score[rpn_idx]
        loss_rpn = F.binary_cross_entropy_with_logits(rpn_score_out,
                                                      torch.zeros_like(rpn_score_out).to(rpn_score_out.device),
                                                      reduction="sum")
        return loss_rpn

    def cal_score_loss(self, score, idx):
        output = score[0][idx[0].detach().cpu().numpy().tolist()]
        output = output[torch.argmax(output, dim=1) == 0]
        if len(output) > 0:
            label = torch.zeros((len(output)), dtype=torch.int64).to(input_adv.device)
            loss_score = F.cross_entropy(output, label, reduction="mean")
        else:
            loss_score = 0.0 * F.cross_entropy(
                output,
                torch.zeros(0, dtype=torch.long, device=self.pred_class_logits.device),
                reduction="sum",
            )
        return loss_score

    def read_image(self, image_path):
        from PIL import Image
        from torchvision import datasets, transforms, models
        MIN_SIZE_TEST = 512
        # Maximum size of the side of the image during testing
        MAX_SIZE_TEST = 1280
        transform_gen = T.ResizeShortestEdge(
            [MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST
        )

        img = cv2.imread(image_path)

        img = transform_gen.get_transform(img).apply_image(img)

        img1 = img.copy()
        image = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).contiguous() / 255

        return image

    def save_textures(self, textures, Tpath):

        stextures = textures.detach().cpu().numpy()
        np.save(Tpath, stextures)

    def tanh_infrared(self, data):
        data = (torch.tanh(data) + 1) / 2
        one = torch.ones_like(data).to(data.device)
        thredhold = torch.unique(data)[round(len(torch.unique(data)) / 3)]
        data = torch.where(data > thredhold, one * 0.9, data)
        data = torch.where(data <= thredhold, one * 0.2, data)
        return data

    def tanh(self, data):
        return (torch.tanh(data) + 1) / 2

    def cal_loss(self, features_adv, features_ori):
        res = None
        for k in features_adv:
            # num = 1
            # for d in features_adv[k].shape:
            #     num *= d
            if res == None:
                res = torch.norm(torch.abs(features_adv[k] - features_ori[k]), 2).mul(0.2).clone()

            else:
                res += torch.norm(torch.abs(features_adv[k] - features_ori[k]), 2).mul(0.2).clone()

        return 1 / res

    def cal_loss1(self, features_adv, features_ori):
        res = None

        for k in features_adv:
            # num = 1
            # for d in features_adv[k].shape:
            #     num *= d
            if res == None:
                res = torch.norm(torch.abs(features_adv[k] - features_ori[k]), 2).mul(0.2).clone()

            else:
                res += torch.norm(torch.abs(features_adv[k] - features_ori[k]), 2).mul(0.2).clone()

        return res

    def cal_loss_rpn_roi(self, pred_objectness_logits, pred_anchor_deltas, cls_list_out, box_list_out, score_list_out):
        ctr = nn.MSELoss()
        for i in range(len(pred_objectness_logits)):
            one = torch.ones_like(pred_objectness_logits[i]).to(pred_objectness_logits[i].device)
            if i == 0:
                loss_po = ctr(one * pred_objectness_logits[i].min(), pred_objectness_logits[i])
            else:
                loss_po += ctr(one * pred_objectness_logits[i].min(), pred_objectness_logits[i])
        # for i in range(len(pred_anchor_deltas)):
        #     zero=torch.zeros_like(pred_anchor_deltas[i]).to(pred_anchor_deltas[i].device)
        #     if i==0:
        #         loss_pa=ctr(zero,pred_anchor_deltas[i])
        #     else:
        #         loss_pa += ctr(zero, pred_objectness_logits[i])
        loss_rpn = loss_po

        loss_out = None
        loss_score_out = None
        loss_box_out = None
        if len(cls_list_out) > 0:
            first = True
            for i in range(len(cls_list_out)):
                if cls_list_out[i].item() == 0:
                    if first == True:
                        loss_score_out = score_list_out[i]
                        loss_box_out = ctr(box_list_out[i],
                                           torch.zeros_like(box_list_out[i]).to(box_list_out[i].device))
                        first = False
                    else:
                        loss_score_out += score_list_out[i]
                        loss_box_out += ctr(box_list_out[i],
                                            torch.zeros_like(box_list_out[i]).to(box_list_out[i].device))

        if loss_score_out is not None:
            loss_out = loss_score_out + loss_box_out

        return loss_out, loss_rpn


cfgs = cfg()
cfg1 = cfg()
cfg2 = cfg()
cfg3 = cfg()
cfg4 = cfg()
cfg5 = cfg()
cfg6 = cfg()
cfg7 = cfg()
cfg8 = cfg()

