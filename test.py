import argparse
from collections import deque
import colorsys
import datetime
import os
import pathlib
import time
import threading

import cv2
import numpy as np
import torch
from darknet import Darknet

# TODO: consistent variable naming (plural/singular)


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
            # We can actually see an ~8% increase in FPS by only calling
            # cv2.imshow when a new frame is set with an if statement.
            if self.frame is not None:
                cv2.imshow(self.win_name, self.frame)
                self.frame = None

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


def detect_cam(
    net, device="cuda", cam_id=0, class_names=None, show_fps=False,
    frames=None
):
    video_getter = VideoGetter(cam_id).start()
    video_shower = VideoShower(video_getter.frame, "YOLOv3").start()

    if show_fps:
        num_fps_frames = 30
        previous_fps = deque(maxlen=num_fps_frames)

    num_frames = 0
    while True:
        loop_start_time = time.time()

        if video_getter.stopped or video_shower.stopped:
            video_getter.stop()
            video_shower.stop()
            break

        frame = video_getter.frame
        bboxes, cls_idx = do_inference(net, frame, device=device)
        draw_boxes(
            frame, bboxes, cls_idx=cls_idx, class_names=class_names
        )

        if show_fps:
            cv2.putText(
                frame,  f"{int(sum(previous_fps) / num_fps_frames)} fps",
                (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                (255, 255, 255)
            )

        video_shower.frame = frame
        if frames is not None:
            frames.append(frame)

        previous_fps.append(int(1 / (time.time() - loop_start_time)))


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
    parser.add_argument(
        "--device", "-d", default="cuda", help="Device (cpu or cuda) to use"
    )
    parser.add_argument(
        "--output-path", "-o", type=pathlib.Path,
        help="Path for writing output image/video file"
    )
    parser.add_argument(
        "--show-fps", action="store_true",
        help="Display frames processed per second (for --cam or --video input)"
    )

    source_ = parser.add_argument_group(
        title="source", description="Input source (required)"
    )
    source = source_.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cam", "-C", type=int, metavar="CAM_ID",
        help="Camera or video capture device ID"
    )
    source.add_argument(
        "--image", "-i", type=pathlib.Path, metavar="IMAGE_PATH",
        help="Path to image file"
    )
    source.add_argument(
        "--video", "-v", type=pathlib.Path, metavar="VIDEO_PATH",
        help="Path to video file"
    )
    args = vars(parser.parse_args())

    path_args = (
        "class_list", "config_path", "weights_path", "image",
        "video", "output_path"
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

    if args["image"]:
        image = cv2.imread(args["image"])
        bboxes, cls_idx = do_inference(net, image, device=device)
        draw_boxes(image, bboxes, cls_idx=cls_idx, class_names=class_names)

        if args["output_path"] is not None:
            cv2.imwrite(args["output_path"], image)

        cv2.imshow("img", image)
        cv2.waitKey(0)
    elif args["video"]:
        frames = None
        if args["output_path"] is not None:
            frames = []

        cap = cv2.VideoCapture(args["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            while True:
                grabbed, frame = cap.read()
                if not grabbed:
                    break

                bboxes, cls_idx = do_inference(net, frame, device=device)
                draw_boxes(
                    frame, bboxes, cls_idx=cls_idx, class_names=class_names
                )

                cv2.imshow("YOLOv3", frame)

                if args["output_path"] is not None:
                    frames.append(frame)

                if cv2.waitKey(1) == ord("q"):
                    break
        except Exception as e:
            raise e
        finally:
            cap.release()

            if args["output_path"] is not None and frames:
                h, w = frames[0].shape[:2]

                output_path = args["output_path"]
                if not output_path.endswith(".mp4"):
                    output_path += ".mp4"

                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    int(fps), (w, h)
                )

                for frame in frames:
                    writer.write(frame)
                writer.release()
    elif args["cam"] is not None:
        frames = None
        if args["output_path"] is not None:
            frames = []
            start_time = time.time()

        try:
            detect_cam(
                net, device=device, class_names=class_names,
                cam_id=args["cam"], show_fps=args["show_fps"], frames=frames
            )
        except Exception as e:
            raise e
        finally:
            if args["output_path"] is not None and frames:
                elapsed = time.time() - start_time
                overall_fps = 1 / (elapsed / len(frames))

                h, w = frames[0].shape[:2]

                output_path = args["output_path"]
                if not output_path.endswith(".mp4"):
                    output_path += ".mp4"

                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    int(overall_fps), (w, h)
                )

                for frame in frames:
                    writer.write(frame)
                writer.release()

    cv2.destroyAllWindows()
