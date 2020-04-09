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
    def __init__(self, frame=None, win_name="Video"):
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
        cv2.destroyWindow(self.win_name)
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


def detect_in_cam(
    net, device="cuda", cam_id=0, class_names=None, show_fps=False, frames=None
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


def detect_in_video(
    net, filepath, device="cuda", class_names=None, frames=None,
    show_video=True
):
    cap = cv2.VideoCapture(filepath)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        bboxes, cls_idx = do_inference(net, frame, device=device)
        draw_boxes(
            frame, bboxes, cls_idx=cls_idx, class_names=class_names
        )

        if args["output"] is not None:
            frames.append(frame)

        if show_video:
            cv2.imshow("YOLOv3", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def write_mp4(frames, fps, filepath):
    if not filepath.endswith(".mp4"):
        filepath += ".mp4"

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        filepath, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h)
    )

    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    source_ = parser.add_argument_group(title="input source [required]")
    source_args = source_.add_mutually_exclusive_group(required=True)
    source_args.add_argument(
        "--cam", "-C", type=int, metavar="CAM_ID", nargs="?", const=0,
        help="Camera or video capture device ID. [Default: 0]"
    )
    source_args.add_argument(
        "--image", "-I", type=pathlib.Path, metavar="PATH",
        help="Path to image file."
    )
    source_args.add_argument(
        "--video", "-V", type=pathlib.Path, metavar="PATH",
        help="Path to video file."
    )

    model_args = parser.add_argument_group(title="model parameters")
    model_args.add_argument(
        "--config", "-c", type=pathlib.Path, required=True, metavar="PATH",
        help="[Required] Path to Darknet model config file."
    )
    model_args.add_argument(
        "--weights", "-w", type=pathlib.Path, required=True, metavar="PATH",
        help="[Required] Path to Darknet model weights file."
    )
    model_args.add_argument(
        "--class-names", "-n", type=pathlib.Path, metavar="PATH",
        help="Path to text file of class names. If omitted, class index is \
            displayed instead of name."
    )
    model_args.add_argument(
        "--device", "-d", type=str, default="cuda",
        help="Device for inference ('cpu', 'cuda'). [Default: 'cuda']"
    )

    other_args = parser.add_argument_group(title="Output/display options")
    other_args.add_argument(
        "--output", "-o", type=pathlib.Path, metavar="PATH",
        help="Path for writing output image/video file. --cam and --video \
            input source options only support .mp4 output filetype. \
            If --video input source selected, output FPS matches input FPS."
    )
    other_args.add_argument(
        "--show-fps", action="store_true",
        help="Display frames processed per second (for --cam input)."
    )

    args = vars(parser.parse_args())

    path_args = ("class_names", "config", "weights", "image", "video", "output")
    for path_arg in path_args:
        if args[path_arg] is not None:
            args[path_arg] = str(args[path_arg].expanduser().absolute())

    device = args["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    net = Darknet(args["config"], device=device)
    net.load_weights(args["weights"])
    net.eval()

    if device.startswith("cuda"):
        net.cuda(device=device)

    class_names = None
    if args["class_names"] is not None and os.path.isfile(args["class_names"]):
        with open(args["class_names"], "r") as f:
            class_names = [line.strip() for line in f.readlines()]

    if args["image"]:
        source = "image"
    elif args["video"]:
        source = "video"
    else:
        source = "cam"

    if source == "image":
        image = cv2.imread(args["image"])
        bboxes, cls_idx = do_inference(net, image, device=device)
        draw_boxes(image, bboxes, cls_idx=cls_idx, class_names=class_names)
        if args["output"]:
            cv2.imwrite(args["output"], image)
        cv2.imshow("img", image)
        cv2.waitKey(0)
    else:
        frames = None
        if args["output"]:
            frames = []

        if source == "cam":
            start_time = time.time()

            # Wrap in try/except block so that writing output video is written
            # even if an error occurs while streaming webcam input.
            try:
                detect_in_cam(
                    net, device=device, class_names=class_names,
                    cam_id=args["cam"], show_fps=args["show_fps"], frames=frames
                )
            except Exception as e:
                raise e
            finally:
                if args["output"] and frames:
                    # Get average FPS and write output at that framerate.
                    fps = 1 / ((time.time() - start_time) / len(frames))
                    write_mp4(frames, fps, args["output"])
        elif source == "video":
            detect_in_video(
                net, filepath=args["video"], device=device,
                class_names=class_names, frames=frames
            )

            if args["output"] and frames:
                # Get input video FPS and write output video at same FPS.
                cap = cv2.VideoCapture(args["video"])
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                write_mp4(frames, fps, args["output"])

    cv2.destroyAllWindows()
