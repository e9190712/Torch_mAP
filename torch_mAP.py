import os
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pycocotools.coco import COCO
import numpy as np
from mmdet.apis import init_detector, inference_detector_fixed
from pprint import pprint
from tqdm import tqdm
import cv2

def coco_label_serious_proc(coco, size_, img_dict, catIDs):
    class_mask_dict = {'person': [], 'bicycle': [], 'car': [],
                  'motorcycle': [], 'bus': [], 'train': [], 'truck': []}
    annIds = coco.getAnnIds(imgIds=img_dict['id'], catIds=catIDs, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for i in range(len(anns)):
        ### 81cls
        # if anns[i]['category_id'] == 6 :
        #     anns[i]['category_id'] = 5
        # if anns[i]['category_id'] == 7 :
        #     anns[i]['category_id'] = 6
        # if anns[i]['category_id'] == 8 :
        #     anns[i]['category_id'] = 7
        ### 81cls
        obj_pts = []
        temp_img = np.zeros(size_[:2], np.uint8)
        # print(anns[i])
        for seg in anns[i]['segmentation']:
            if isinstance(seg, str):
                continue
            pt = []
            for idx in range(0, len(seg), 2):
                pt.append([seg[idx], seg[idx+1]])
            points = np.array(pt, np.int32)
            obj_pts.append(points)
        cv2.fillPoly(temp_img, pts=obj_pts, color=1)
        # print(list(class_mask_dict.keys())[anns[i]['category_id'] - 1])
        # plt.imshow(temp_img)
        # plt.show()
        temp_img = np.expand_dims(temp_img, axis=0)
        class_mask_dict[list(class_mask_dict.keys())[anns[i]['category_id'] - 1]].append(temp_img) # coco
        # class_mask_dict[list(class_mask_dict.keys())[anns[i]['category_id']]].append(temp_img) # google
    return class_mask_dict
def model_label_serious_proc(result, class_names, size_=(0,0), IoU_th=0.2):
    class_mask_dict = {'person': [], 'bicycle': [], 'car': [],
                  'motorcycle': [], 'bus': [], 'train': [], 'truck': []}
    class_score_dict = {'person': [], 'bicycle': [], 'car': [],
                  'motorcycle': [], 'bus': [], 'train': [], 'truck': []}
    for cur_result in result:
        # if pred obj is None
        if cur_result is None:
            continue
        # if pred obj is None

        seg_pred = cur_result[0].cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1].cpu().numpy().astype(np.int64)
        cate_score = cur_result[2].cpu().numpy().astype(np.float64)
        for idx in range(len(cate_score[cate_score > IoU_th])): # cate_score -> confidence
            temp_img = np.zeros(size_[:2], np.uint8)
            cur_mask = seg_pred[idx, ...]
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            cur_mask_bool = cur_mask.astype(np.bool_)
            cur_cate = cate_label[idx]
            label_text = class_names[cur_cate]
            temp_img[cur_mask_bool] = 1
            temp_img = np.expand_dims(temp_img, axis=0)
            if label_text in class_mask_dict.keys():
                class_mask_dict[label_text].append(temp_img)
                class_score_dict[label_text].append(cate_score[cate_score > IoU_th][idx])
    return class_mask_dict, class_score_dict
# annFile = '/home/jovyan/work/google_val/annotations.json'
# annFile = '/home/jovyan/work/google_val/annotations_81cls.json'
annFile = '/home/jovyan/work/stitch_cocodataset/stitch_val2017.json'
img_path = '/home/jovyan/work/stitch_cocodataset/stitch_val'
# annFile = '/home/jovyan/work/coco/annotations/instances_val2017_filtered.json'
# img_path = '/home/jovyan/work/coco/val2017'
config_file = 'configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.py'
checkpoint_file = '/home/jovyan/work/solov2_weight_221119/solov2_8cls_epoch50.pth'
# checkpoint_file = '/home/jovyan/work/solov2_light_release_r50_fpn_8gpu_3x_BN_google1219/epoch_39.pth'
# checkpoint_file = '/home/jovyan/work/SOLOv2_LIGHT_448_R50_3x.pth'
# checkpoint_file = '/home/jovyan/work/solov2_light_release_r50_fpn_8gpu_3x_BN_cocostitch_768x448/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
coco = COCO(annFile)
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']
catIDs = []
imgIds = coco.getImgIds(catIds=catIDs)
metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
for rand_ID in tqdm(range(0, len(imgIds))):
    img_dict = coco.loadImgs(imgIds[rand_ID])[0]
    img = cv2.imread(os.path.join(img_path, img_dict['file_name']))
    origin_size = img.shape
    with torch.cuda.amp.autocast():
        out_ = inference_detector_fixed(model, dict(img=img)) # model result
    target_masks_dict = coco_label_serious_proc(coco, origin_size, img_dict, catIDs)
    res_masks_dict, res_score_dict = model_label_serious_proc(out_, model.CLASSES, origin_size, IoU_th=0.2)
    temp_target_list = []
    temp_preds_list = []
    target_mask_list = []
    preds_mask_list  = []
    res_score_list = []
    cls_idx_pred_list = []
    cls_idx_target_list = []
    for cls_idx, class_name in enumerate(class_list):
        for gt_idx, target_mask in enumerate(target_masks_dict[class_name]):
            target_mask_list.append(target_mask)
            cls_idx_target_list.append(cls_idx)
    for cls_idx, class_name in enumerate(class_list):
        for res_mask, res_score in zip(res_masks_dict[class_name], res_score_dict[class_name]):
            preds_mask_list.append(res_mask)
            res_score_list.append(res_score)
            cls_idx_pred_list.append(cls_idx)
    if preds_mask_list:
        temp_target_list.append(dict(
            masks=torch.tensor(np.concatenate(target_mask_list)),
            labels=torch.tensor(cls_idx_target_list)
        ))
        temp_preds_list.append(dict(
            masks=torch.tensor(np.concatenate(preds_mask_list)),
            scores=torch.tensor(res_score_list),
            labels=torch.tensor(cls_idx_pred_list)
        ))
        metric.update(temp_preds_list, temp_target_list)
pprint(metric.compute())