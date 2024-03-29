import os
import cv2
import time
import numpy as np
import onnxruntime
import torch
import scipy.ndimage
from tqdm import tqdm
from scipy import ndimage
from utils.dataloaders import LoadImages # yolov5-seg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pycocotools.coco import COCO
from pprint import pprint

# ort_session = onnxruntime.InferenceSession("../work/mike/runs/train-seg-relu-tidl/exp/weights/best_tidl.onnx", 
                                           # providers=['CUDAExecutionProvider'])
ort_session = onnxruntime.InferenceSession("weight/train-seg-relu-tidl-coco-0630-fp32.onnx", 
                                           providers=['CUDAExecutionProvider'])
output_names = [x.name for x in ort_session.get_outputs()]


def coco_label_serious_proc(coco, size_, img_dict, catIDs, iou_method, test_img):
    class_info_dict = {'person': [], 'bicycle': [], 'car': [],
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
        # print(anns[i]['bbox'])
        if iou_method == 'segm':
            for seg in anns[i]['segmentation']:
                if isinstance(seg, str):
                    continue
                pt = []
                for idx in range(0, len(seg), 2):
                    pt.append([seg[idx], seg[idx+1]])
                points = np.array(pt, np.int32)
                obj_pts.append(points)
            cv2.fillPoly(temp_img, pts=obj_pts, color=1)
            # print(list(class_info_dict.keys())[anns[i]['category_id'] - 1])
            # plt.imshow(temp_img)
            # plt.show()
            temp_img = np.expand_dims(temp_img, axis=0)
            class_info_dict[list(class_info_dict.keys())[anns[i]['category_id'] - 1]].append(temp_img) # coco
            # class_info_dict[list(class_info_dict.keys())[anns[i]['category_id']]].append(temp_img) # google
        elif iou_method == 'bbox':
            # fig, ax = plt.subplots(1, 1)
            # rect = patches.Rectangle((int(anns[i]['bbox'][0]), int(anns[i]['bbox'][1])), int(anns[i]['bbox'][2]), int(anns[i]['bbox'][3]), edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            # ax.imshow(test_img)
            # plt.show()
            class_info_dict[list(class_info_dict.keys())[anns[i]['category_id'] - 1]].append(anns[i]['bbox']) # coco
    return class_info_dict
        
        

def model_label_serious_proc(mask, output, class_name, model_size, coco_size, iou_method, test_img):
    class_info_dict = {'person': [], 'bicycle': [], 'car': [],
                  'motorcycle': [], 'bus': [], 'train': [], 'truck': []}
    class_score_dict = {'person': [], 'bicycle': [], 'car': [],
                  'motorcycle': [], 'bus': [], 'train': [], 'truck': []}

    for bbidx in range(len(output)):
        clsid = output[bbidx, 5]
        if iou_method == "segm":
            class_info_dict[class_name[int(clsid)]].append(mask[bbidx])
        elif iou_method == "bbox":
            x_scale = coco_size[1] / model_size[1]
            y_scale = coco_size[0] / model_size[0]
            x1 = output[bbidx, 0] * x_scale
            y1 = output[bbidx, 1] * y_scale
            x2 = output[bbidx, 2] * x_scale
            y2 = output[bbidx, 3] * y_scale
            
            # fig, ax = plt.subplots(1, 1)
            # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            # ax.imshow(test_img)
            # plt.show()
            
            class_info_dict[class_name[int(clsid)]].append([x1, y1, x2 - x1, y2 - y1])
        class_score_dict[class_name[int(clsid)]].append(output[bbidx, 4])
    
    # print(clsid)
    # print("%.2f" % score)

    return class_info_dict, class_score_dict

def overlay_masks_on_image(image, masks, alpha=0.5):
    """
    Overlay masks on the original image.

    image: Input image, shape (H, W, 3)
    masks: Binary masks, shape (N, H, W), N is the number of masks
    alpha: Transparency factor for the masks

    return: Overlaid image
    """
    assert image.shape[:2] == masks.shape[1:], "Image and mask dimensions must match."

    # If the image has integer values in the range 0-255, normalize it to float values in the range 0-1
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Create a color map for masks
    num_masks = masks.shape[0]
    colors = plt.cm.hsv(np.linspace(0, 1, num_masks)).tolist()

    # Overlay masks on the image
    overlaid_image = image.copy()
    for idx in range(num_masks):
        color = np.array(colors[idx][:3])
        mask = masks[idx]

        # Apply the color to the mask
        colored_mask = np.repeat(mask[..., None], 3, axis=-1) * color

        # Overlay the colored mask on the image with alpha blending
        overlaid_image = overlaid_image * (1 - alpha * mask[..., None]) + colored_mask * alpha * mask[..., None]

    return overlaid_image
def make_grid(anchors, stride, nx=20, ny=20, na=3):
    shape = 1, na, ny, nx, 2  # grid shape
    y, x = np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32) # must be np.float32, otherwise the precision will be very low
    yv, xv = np.meshgrid(y, x, indexing='ij')
    grid = np.stack((xv, yv), 2)
    grid = np.expand_dims(np.repeat(np.expand_dims(grid, 0), 3, 0), 0) - 0.5
    anchor_grid = np.repeat(np.repeat((anchors * stride).reshape((1, na, 1, 1, 2)), ny, 2), nx, 3)
    return grid, anchor_grid
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def nms_numpy(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)
def crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)
    # x1, y1, x2, y2 = np.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
def process_mask(protos, masks_in, bboxes, shape, coco_shape, upsample=False):
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    # masks = (masks_in @ sigmoid(protos.reshape(c, -1))).reshape(-1, mh, mw)  # CHW
    masks = np.matmul(masks_in, protos.reshape(c, -1))
    # print(masks)
    masks = sigmoid(masks)
    masks = masks.reshape(-1, mh, mw)
    
    # downsampled_bboxes = bboxes.clone()
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        zoom_factor_y = ih / masks.shape[1]
        zoom_factor_x = iw / masks.shape[2]
        masks = scipy.ndimage.zoom(masks, zoom=(1, zoom_factor_y, zoom_factor_x), order=1)
    # print(masks[-1])
    masks_list = []
    for mask in masks:
        mask = np.expand_dims(cv2.resize(mask, dsize=(coco_shape[1], coco_shape[0])), 0)
        mask = mask > 0.5
        masks_list.append(mask)
    return masks_list

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

num_mask = 32
num_cls = 7
no = num_cls + num_mask + 5
na = 3
conf_thres = 0.25
iou_thres = 0.45
mi = 5 + num_cls  # mask start index
agnostic_nms=False
merge = False
max_det = 300
max_wh = 7680  # (pixels) maximum box width and height
max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
anchors = np.array([[[1.25000, 1.62500],
        [2.00000, 3.75000],
        [4.12500, 2.87500]],
           [[1.87500, 3.81250],
        [3.87500, 2.81250],
        [3.68750, 7.43750]],
           [[ 3.62500,  2.81250],
        [ 4.87500,  6.18750],
        [11.65625, 10.18750]]])
stride = np.array([8,16,32])

annFile = '/mnt/mike/coco/annotations/instances_val2017_filtered.json'
img_path = '/mnt/mike/coco/val2017'
coco = COCO(annFile)
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']
catIDs = []
imgIds = coco.getImgIds(catIds=catIDs)
iou_method = "bbox" # bbox, segm

if iou_method == 'bbox':
    metric = MeanAveragePrecision(iou_type=iou_method, box_format='xywh', class_metrics=True)
elif iou_method == 'segm':
    metric = MeanAveragePrecision(iou_type=iou_method, class_metrics=True)

for rand_ID in tqdm(range(0, len(imgIds))):
    img_dict = coco.loadImgs(imgIds[rand_ID])[0]
    img_coco = cv2.imread(os.path.join(img_path, img_dict['file_name']))
    target_masks_dict = coco_label_serious_proc(coco, img_coco.shape, img_dict, catIDs, iou_method, img_coco)
    
    im_orig = cv2.resize(img_coco, (640,640))
    im = im_orig.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # im = im.permute(0, 3, 1, 2)
    im = im.cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: im}
    ort_outs = ort_session.run(output_names, ort_inputs)
    z = []
    for i in range(3):
        bs, _, ny, nx = ort_outs[i].shape
        ort_outs[i] = ort_outs[i].reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
        xy = ort_outs[i][...,:2]
        wh = ort_outs[i][...,2:4]
        conf = ort_outs[i][...,4:6]
        mask = ort_outs[i][...,6:]
        grid, anchor_grid = make_grid(anchors[i], stride[i], nx, ny)
        anchor_grid = anchor_grid.astype(np.int32) # must be change to int32 if write to bin file
        xy = (sigmoid(xy) * 2 + grid) * stride[i] # xy
        wh = (sigmoid(wh) * 2) ** 2 * anchor_grid # wh
        conf = sigmoid(conf)
        res = np.concatenate((xy,wh,conf,mask), 4)
        res = res.reshape(bs, na * nx * ny, no)
        z.append(res)
    pred = np.concatenate(z, 1)
    bs = pred.shape[0]  # batch size
    nc = pred.shape[2] - num_mask - 5  # number of classes
    xc = pred[..., 4] > conf_thres  # candidates

    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    t = time.time()
    output = [np.zeros((0, 6 + num_mask))] * bs
    for xi, x in enumerate(pred):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks
        j = np.argmax(x[:, 5:mi], axis=1)
        j = np.expand_dims(j, axis=1)  # Add a new dimension to match the keepdims behavior
        conf = np.amax(x[:, 5:mi], axis=1, keepdims=True)
        x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[np.argsort(-x[:, 4])[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic_nms else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_numpy(boxes, scores, iou_thres)

        i = i[:max_det]  # limit detections
        
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
        if len(output[xi]):
            masks = []
            if iou_method == 'segm':
                masks = process_mask(ort_outs[3][xi], output[xi][:, 6:], output[xi][:, :4], im.shape[2:], img_coco.shape[:2], upsample=True)  # HWC

            ''' test display '''
            # for mask in masks:
            #     overlaid_image = overlay_masks_on_image(img_coco, mask)
            #     plt.imshow(overlaid_image)
            #     plt.show()
            ''' end '''
            
            res_masks_dict, res_score_dict = model_label_serious_proc(masks, output[xi], class_list, im.shape[2:], img_coco.shape[:2], iou_method, img_coco)

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
                if iou_method == 'segm':
                    temp_target_list.append(dict(
                        masks=torch.tensor(np.concatenate(target_mask_list)),
                        labels=torch.tensor(cls_idx_target_list)
                    ))
                    temp_preds_list.append(dict(
                        masks=torch.tensor(np.concatenate(preds_mask_list)),
                        scores=torch.tensor(res_score_list),
                        labels=torch.tensor(cls_idx_pred_list)
                    ))
                elif iou_method == 'bbox':
                    temp_target_list.append(dict(
                        boxes=torch.tensor(target_mask_list),
                        labels=torch.tensor(cls_idx_target_list)
                    ))
                    temp_preds_list.append(dict(
                        boxes=torch.tensor(preds_mask_list),
                        scores=torch.tensor(res_score_list),
                        labels=torch.tensor(cls_idx_pred_list)
                    ))
                metric.update(temp_preds_list, temp_target_list)
pprint(metric.compute())
