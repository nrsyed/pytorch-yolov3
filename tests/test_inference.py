import os

import cv2
import numpy as np
from pycocotools import coco, cocoeval

from darknet import Darknet
import inference

from devtools import coco_util


def test_bbox_coord_conversion():
    bbox_xywh = np.array([
        [5, 8, 10, 13, 10000],
        [100, 200, 30, 17, 19000]
    ], dtype=np.int)

    bbox_tlbr = np.array([
        [0, 2, 10, 14, 10000],
        [85, 192, 115, 208, 19000]
    ])

    assert (inference.cxywh_to_tlbr(bbox_xywh) == bbox_tlbr).all()


def test_inference():
    """
    With a probability threshold ("score" threshold) of 0.2, the standard
    yolov3 model yields a "bbox" mAP @[IoU=0.50:0.95] of 0.33983872 on the 9
    sample images from the COCO validation set as computed by pycocotools.
    This test checks that the mAP is similar.
    """
    model = "yolov3"
    model_dir = "models"
    config_path = os.path.join(model_dir, model + ".cfg")
    weights_path = os.path.join(model_dir, model + ".weights")

    net = Darknet(config_path, device="cpu")
    net.load_weights(weights_path)
    net.eval()

    image_dir = os.path.join("sample_dataset", "images")
    fnames = os.listdir(image_dir)

    images = []
    for fname in fnames:
        fpath = os.path.join(image_dir, fname)
        images.append(cv2.imread(fpath))

    # Accumulate images instead of batching; helps run on systems (including
    # Travis CI) with lower amounts of RAM.
    results = []
    for image in images:
        results.extend(
            inference.inference(net, image, device="cpu", prob_thresh=0.2)
        )

    with open("models/coco.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    pred_dataset = inference.to_coco(fnames, results, class_names)
    truth_dataset = coco_util.load_coco_dataset("sample_dataset/sample.json")

    # Match predicted COCO dataset image ids and cat ids with original
    # ground truth dataset.
    coco_util.match_ids(pred_dataset, truth_dataset)

    # Ground truth COCO API dataset.
    gt_coco = coco.COCO()
    gt_coco.dataset = truth_dataset
    gt_coco.createIndex()

    # Detections COCO API dataset.
    dt_coco = coco.COCO()
    dt_coco.dataset = pred_dataset
    dt_coco.createIndex()

    eval_ = cocoeval.COCOeval(gt_coco, dt_coco)
    eval_.params.iouType = "bbox"
    eval_.evaluate()
    eval_.accumulate()
    eval_.summarize()

    assert np.isclose(eval_.stats[0], 0.33983872, atol=0.0015)
