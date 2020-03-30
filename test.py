import argparse
from collections import deque
import colorsys
import os
import pathlib
import pdb

import cv2
import numpy as np
import torch
from darknet import Darknet

# TODO: consistent variable naming (plural/singular)


def img_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)


def unique_colors(num_colors):
    """
    Yield N unique BGR colors using HSV color space as an intermediate.
    """

    for H in np.linspace(0, 1, num_colors, endpoint=False):
        rgb = colorsys.hsv_to_rgb(H, 1.0, 1.0)
        bgr = (int(255 * rgb[2]), int(255 * rgb[1]), int(255 * rgb[0]))
        yield bgr


def draw_boxes(img, bboxes, prob=None, cls_idx=None, class_names=None):
    if cls_idx is not None:
        unique_cls_idxs = set(cls_idx)
        colors = dict()
        num_colors = len(unique_cls_idxs)
        for idx, color in zip(unique_cls_idxs, unique_colors(num_colors)):
            colors[idx] = color

    for i, bbox in enumerate(bboxes):
        bbox_str = []
        if cls_idx is not None:
            color = colors[cls_idx[i]]
            if class_names is not None:
                bbox_str.append(class_names[cls_idx[i]])
            else:
                bbox_str.append(str(cls_idx[i]))
        else:
            color = (0, 255, 0)

        if prob is not None:
            bbox_str.append("({:.2f})".format(prob[i]))

        bbox_str = " ".join(bbox_str)

        tl_x, tl_y, br_x, br_y = bbox
        cv2.rectangle(
            img, (tl_x, tl_y), (br_x, br_y), color=color, thickness=2
        )

        if bbox_str:
            cv2.rectangle(
                img, (tl_x + 1, tl_y + 1),
                (tl_x + int(8 * len(bbox_str)), tl_y + 18),
                color=(20, 20, 20), thickness=cv2.FILLED
            )
            cv2.putText(
                img, bbox_str, (tl_x + 1, tl_y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), thickness=1
            )


def _non_max_suppression(bboxes, prob, iou_thresh=0.3):
    """
    Perform non-maximum suppression on an array of bboxes and
    return the indices of detections to retain.

    Derived from:
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Args:
        bboxes (np.array): An Mx5 array of bboxes (consisting of M
            detections/bboxes), where bboxes[:, :4] represent the four
            bbox coordinates.

        prob (np.array): An array of M elements corresponding to the
            max class probability of each detection/bbox.
    """

    area = ((bboxes[:,2] - bboxes[:,0]) + 1) * ((bboxes[:,3] - bboxes[:,1]) + 1)

    # Sort detections by probability (largest to smallest).
    idxs = deque(np.argsort(prob)[::-1])
    idxs_to_keep = list()

    while idxs:
        # Grab current index (index corresponding to the detection with the
        # greatest probability currently in the list of indices).
        curr_idx = idxs.popleft()
        idxs_to_keep.append(curr_idx)

        # Find the coordinates of the regions of overlap between the current
        # detection and all other detections.
        overlaps_tl_x = np.maximum(bboxes[curr_idx, 0], bboxes[idxs, 0])
        overlaps_tl_y = np.maximum(bboxes[curr_idx, 1], bboxes[idxs, 1])
        overlaps_br_x = np.minimum(bboxes[curr_idx, 2], bboxes[idxs, 2])
        overlaps_br_y = np.minimum(bboxes[curr_idx, 3], bboxes[idxs, 3])

        # Compute width and height of overlapping regions.
        overlap_w = np.maximum(0, (overlaps_br_x - overlaps_tl_x) + 1)
        overlap_h = np.maximum(0, (overlaps_br_y - overlaps_tl_y) + 1)

        # Compute amount of overlap (intersection).
        inter = overlap_w * overlap_h
        union = area[curr_idx] + area[idxs] - inter
        iou = inter / union

        idxs_to_remove = [idxs[i] for i in np.where(iou > iou_thresh)[0]]
        for idx in idxs_to_remove:
            idxs.remove(idx)

    return idxs_to_keep


def non_max_suppression(bboxes, prob, cls_idx=None, iou_thresh=0.3):
    """
    TODO
    """

    if cls_idx is not None:
        # Perform per-class non-maximum suppression.
        idxs_to_keep = []
        classes = set(cls_idx)
        for class_ in classes:
            curr_class_idxs = np.where(cls_idx == class_)[0]
            class_bboxes = bboxes[curr_class_idxs]
            class_prob = prob[curr_class_idxs]
            class_idxs_to_keep = _non_max_suppression(class_bboxes, class_prob)
            idxs_to_keep.extend(curr_class_idxs[class_idxs_to_keep].tolist())
    else:
        idxs_to_keep = _non_max_suppression(bboxes, prob, iou_thresh)
    return idxs_to_keep


def cxywh_to_tlbr(bboxes):
    """
    Args:
        bboxes (np.array): An MxN array of detections where bboxes[:, :4]
            correspond to coordinates (center x, center y, width, height).

    Returns:
        An MxN array of detections where bboxes[:, :4] correspond to
        coordinates (top left x, top left y, bottom right x, bottom right y).
    """

    tlbr = np.copy(bboxes)
    tlbr[:, :2] = bboxes[:, :2] - (bboxes[:, 2:4] // 2)
    tlbr[:, 2:4] = bboxes[:, :2] + (bboxes[:, 2:4] // 2)
    return tlbr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path_args = parser.add_argument_group()
    parser.add_argument(
        "--config-path", "-c", type=pathlib.Path,
        default=pathlib.Path("models/yolov3.cfg"),
        help="Path to Darknet model config file"
    )
    parser.add_argument(
        "--weights-path", "-w", type=pathlib.Path,
        default=pathlib.Path("models/yolov3.weights"),
        help="Path to Darknet model weights file"
    )
    parser.add_argument(
        "--image-path", "-i", type=pathlib.Path, required=True,
        help="Path to image file"
    )
    args = vars(parser.parse_args())

    for path_arg in ("config_path", "weights_path", "image_path"):
        args[path_arg] = str(args[path_arg].expanduser().absolute())

    net = Darknet(args["config_path"]).load_weights(args["weights_path"])
    net.eval()
    net_info = net.net_info

    img = cv2.imread(args["image_path"])
    resized = cv2.resize(img, (net_info["height"], net_info["width"]))

    inp = img_to_tensor(resized)
    out = net.forward(inp)

    bboxes = out["bbox_xywhs"].detach().numpy()
    cls_idx = out["max_class_idx"].numpy()

    prob = bboxes[:, 4]
    bboxes = bboxes[:, :4]

    prob_thresh = 0.05
    mask = prob >= prob_thresh

    bboxes = bboxes[mask, :]
    cls_idx = cls_idx[mask]
    prob = prob[mask]

    rows, cols = img.shape[:2]
    bboxes[:, [0, 2]] *= cols
    bboxes[:, [1, 3]] *= rows
    bboxes = bboxes.astype(np.int)

    bboxes = cxywh_to_tlbr(bboxes)
    #idxs_to_keep = _non_max_suppression(bboxes, prob)
    idxs_to_keep = non_max_suppression(bboxes, prob, cls_idx=cls_idx)
    bboxes = bboxes[idxs_to_keep, :]
    cls_idx = cls_idx[idxs_to_keep]

    class_names = None
    class_names_fpath = "models/coco.names"
    if os.path.isfile(class_names_fpath):
        with open(class_names_fpath, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

    draw_boxes(img, bboxes, cls_idx=cls_idx, class_names=class_names)
    cv2.imshow("img", img)
    cv2.waitKey(0)
