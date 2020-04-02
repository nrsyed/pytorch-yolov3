import argparse
from collections import deque
import colorsys
import os
import pathlib
import pdb
import time
import threading

import cv2
import numpy as np
import torch
from darknet import Darknet

# TODO: consistent variable naming (plural/singular)


import functools
def profile(func):
    def decorated(*args, **kwargs):
        start_t = time.time()
        retval = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - start_t:.4f}")
        return retval
    return decorated


class VideoGetter():
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.cap.read()

    def stop(self):
        self.stopped = True


class VideoShower():
    def __init__(self, frame=None, win_name=""):
        self.frame = frame
        self.win_name = win_name
        self.stopped = False

    def start(self):
        threading.Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow(self.win_name, self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        cv2.destroyAllWindows()
        self.stopped = True


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
    if class_names is not None:
        colors = dict()
        num_colors = len(class_names)
        colors = list(unique_colors(num_colors))

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


def do_inference(
    net, image, device="cuda", prob_thresh=0.12, nms_iou_thresh=0.3, resize=True
):
    orig_rows, orig_cols = image.shape[:2]
    net_info = net.net_info
    if resize and image.shape[:2] != [net_info["height"], net_info["width"]]:
        image = cv2.resize(image, (net_info["height"], net_info["width"]))

    image = np.transpose(np.flip(image, 2), (2, 0, 1)).astype(np.float32) / 255.
    inp = torch.tensor(image, device=device).unsqueeze(0)

    out = net.forward(inp)

    bboxes = out["bbox_xywhs"].detach().cpu().numpy()
    cls_idx = out["max_class_idx"].cpu().numpy()

    prob = bboxes[:, 4]
    bboxes = bboxes[:, :4]

    mask = prob >= prob_thresh

    bboxes = bboxes[mask, :]
    cls_idx = cls_idx[mask]
    prob = prob[mask]

    bboxes[:, [0, 2]] *= orig_cols
    bboxes[:, [1, 3]] *= orig_rows
    bboxes = bboxes.astype(np.int)

    bboxes = cxywh_to_tlbr(bboxes)
    idxs_to_keep = non_max_suppression(
        bboxes, prob, cls_idx=cls_idx, iou_thresh=nms_iou_thresh
    )
    bboxes = bboxes[idxs_to_keep, :]
    cls_idx = cls_idx[idxs_to_keep]
    return bboxes, cls_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class-list", "-l", type=pathlib.Path, default=None,
        help="Path to text file of class names"
    )
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
    parser.add_argument("--device", "-d", default="cuda")

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image-path", "-i", type=pathlib.Path, help="Path to image file"
    )
    source.add_argument(
        "--video-path", "-v", type=pathlib.Path, help="Path to video file"
    )
    source.add_argument(
        "--camera", "-C", type=int, help="Camera or video capture device ID"
    )
    parser.add_argument(
        "-t", "--threading", choices=("none", "get", "show", "both"),
        default="none"
    )
    args = vars(parser.parse_args())

    path_args = (
        "class_list", "config_path", "weights_path", "image_path", "video_path"
    )
    for path_arg in path_args:
        if args[path_arg] is not None:
            args[path_arg] = str(args[path_arg].expanduser().absolute())

    device = args["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    net = Darknet(args["config_path"], device=device)
    net.load_weights(args["weights_path"])
    net.eval()

    if device.startswith("cuda"):
        net.cuda()

    class_names = None
    if args["class_list"] is not None and os.path.isfile(args["class_list"]):
        with open(args["class_list"], "r") as f:
            class_names = [line.strip() for line in f.readlines()]

    if args["image_path"]:
        image = cv2.imread(args["image_path"])
        bboxes, cls_idx = do_inference(net, image, device=device)
        draw_boxes(image, bboxes, cls_idx=cls_idx, class_names=class_names)
        cv2.imshow("img", image)
        cv2.waitKey(0)
    elif args["camera"] is not None:
        num_samples = 50
        fps = deque(maxlen=num_samples)

        if args["threading"] == "get":
            video_getter = VideoGetter(args["camera"]).start()

            while True:
                start_t = time.time()
                if video_getter.stopped:
                    break
                frame = video_getter.frame
                bboxes, cls_idx = do_inference(net, frame, device=device)
                draw_boxes(
                    frame, bboxes, cls_idx=cls_idx, class_names=class_names
                )

                fps.append(int(1 / (time.time() - start_t)))
                cv2.putText(
                    frame,  f"{int(sum(fps) / num_samples)} fps",
                    (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255)
                )
                cv2.imshow("YOLOv3", frame)
                if cv2.waitKey(1) == ord("q"):
                    video_getter.stop()
                    break
        elif args["threading"] == "show":
            cap = cv2.VideoCapture(args["camera"])
            grabbed, frame = cap.read()
            video_shower = VideoShower(frame, win_name="YOLOv3").start()
            while True:
                start_t = time.time()
                grabbed, frame = cap.read()
                if not grabbed or video_shower.stopped:
                    break
                bboxes, cls_idx = do_inference(net, frame, device=device)
                draw_boxes(
                    frame, bboxes, cls_idx=cls_idx, class_names=class_names
                )

                fps.append(int(1 / (time.time() - start_t)))
                cv2.putText(
                    frame,  f"{int(sum(fps) / num_samples)} fps",
                    (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255)
                )
                video_shower.frame = frame
            cap.release()
        elif args["threading"] == "both":
            video_getter = VideoGetter(args["camera"]).start()
            video_shower = VideoShower(video_getter.frame, win_name="YOLOv3")
            video_shower.start()
            while True:
                if video_getter.stopped or video_shower.stopped:
                    video_shower.stop()
                    video_getter.stop()
                    break
                start_t = time.time()
                frame = video_getter.frame
                bboxes, cls_idx = do_inference(net, frame, device=device)
                draw_boxes(
                    frame, bboxes, cls_idx=cls_idx, class_names=class_names
                )

                fps.append(int(1 / (time.time() - start_t)))
                cv2.putText(
                    frame,  f"{int(sum(fps) / num_samples)} fps",
                    (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255)
                )
                video_shower.frame = frame
        else:
            cap = cv2.VideoCapture(args["camera"])
            grabbed, frame = cap.read()

            while True:
                start_t = time.time()
                grabbed, frame = cap.read()
                if not grabbed:
                    break
                bboxes, cls_idx = do_inference(net, frame, device=device)
                draw_boxes(
                    frame, bboxes, cls_idx=cls_idx, class_names=class_names
                )

                fps.append(int(1 / (time.time() - start_t)))
                cv2.putText(
                    frame,  f"{int(sum(fps) / num_samples)} fps",
                    (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255)
                )
                cv2.imshow("YOLOv3", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            cap.release()

        cv2.destroyAllWindows()
