import pdb
import cv2
import numpy as np
import torch
from darknet import Darknet


def img_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)


def draw_boxes(img, bboxes, cls_idx=None, classes=None):
    rows, cols = img.shape[:2]

    bboxes[:, [0, 2]] *= cols
    bboxes[:, [1, 3]] *= rows
    bboxes = bboxes.astype(np.int)

    for i, bbox in enumerate(bboxes):
        center_x, center_y, w, h = bbox
        tl_x = center_x - int(w / 2)
        tl_y = center_y - int(h / 2)
        lr_x = tl_x + w
        lr_y = tl_y + h
        cv2.rectangle(
            img, (tl_x, tl_y), (lr_x, lr_y), color=(0, 255, 0), thickness=1
        )

        if cls_idx is not None:
            cv2.putText(
                img, str(cls_idx[i]), (tl_x, tl_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), thickness=2
            )

if __name__ == "__main__":
    paths = {
        "yolov3-tiny": {
            "config": "models/yolov3-tiny.cfg",
            "weights": "models/yolov3-tiny.weights"
        },
        "yolov3": {
            "config": "models/yolov3.cfg",
            "weights": "models/yolov3.weights"
        },
        "yolov3-spp": {
            "config": "models/yolov3-spp.cfg",
            "weights": "models/yolov3-spp.weights"
        }
    }

    #model = "yolov3"
    #model = "yolov3-tiny"
    #model = "yolov3-spp"
    net = Darknet(paths[model]["config"])
    net.load_weights(paths[model]["weights"])
    net.eval()
    net_info = net.net_info

    img_path = "images/dog-cycle-car.png"
    img = cv2.imread(img_path)
    resized = cv2.resize(img, (net_info["height"], net_info["width"]))

    inp = img_to_tensor(resized)
    out = net.forward(inp)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)

    bboxes = out["bbox_xywhs"].detach().numpy()
    cls_idx = out["max_class_idx"].numpy()

    conf_thresh = 0.05
    mask = bboxes[:, 4] >= conf_thresh
    bboxes = bboxes[mask, :]
    cls_idx = cls_idx[mask]

    draw_boxes(img, bboxes[:, :4], cls_idx=cls_idx, classes=None)
    cv2.imshow("img", img)
    cv2.waitKey(0)
