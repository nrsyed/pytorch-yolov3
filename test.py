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

# TODO: allow str/file/path pointer for webcam inference.


class VideoGetter():
    def __init__(self, src=0):
        """
        Class to read frames from a VideoCapture in a dedicated thread.

        Args:
            src (int|str): Video source. Int if webcam id, str if path to file.
        """
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """
        Method called in a thread to continually read frames from `self.cap`.
        This way, a frame is always ready to be read. Frames are not queued;
        if a frame is not read before `get()` reads a new frame, previous
        frame is overwritten and cannot be obtained again.
        """
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.cap.read()

    def stop(self):
        self.stopped = True


class VideoShower():
    def __init__(self, frame=None, win_name="Video"):
        """
        Class to show frames in a dedicated thread.

        Args:
            frame (np.ndarray): (Initial) frame to display.
            win_name (str): Name of `cv2.imshow()` window.
        """
        self.frame = frame
        self.win_name = win_name
        self.stopped = False

    def start(self):
        threading.Thread(target=self.show, args=()).start()
        return self

    def show(self):
        """
        Method called within thread to show new frames.
        """
        while not self.stopped:
            # We can actually see an ~8% increase in FPS by only calling
            # cv2.imshow when a new frame is set with an if statement. Thus,
            # set `self.frame` to None after each call to `cv2.imshow()`.
            if self.frame is not None:
                cv2.imshow(self.win_name, self.frame)
                self.frame = None

            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        cv2.destroyWindow(self.win_name)
        self.stopped = True


def unique_colors(num_colors):
    """
    Yield `num_colors` unique BGR colors. Uses HSV space as intermediate.

    Args:
        num_colors (int): Number of colors to yield.

    Yields:
        3-tuple of 8-bit BGR values.
    """

    for H in np.linspace(0, 1, num_colors, endpoint=False):
        rgb = colorsys.hsv_to_rgb(H, 1.0, 1.0)
        bgr = (int(255 * rgb[2]), int(255 * rgb[1]), int(255 * rgb[0]))
        yield bgr


def draw_boxes(
    img, bbox_tlbr, class_prob=None, class_idx=None, class_names=None
):
    """
    Draw bboxes (and class names or indices for each bbox) on an image.
    Bboxes are drawn in-place on the original image; this function does not
    return a new image.

    If `class_prob` is provided, the prediction probability for each bbox
    will be displayed along with the bbox. If `class_idx` is provided, the
    class index of each bbox will be displayed along with the bbox. If both
    `class_idx` and `class_names` are provided, `class_idx` will be used to
    determine the class name for each bbox and the class name of each bbox
    will be displayed along with the bbox.

    If `class_names` is provided, a unique color is used for each class.

    Args:
        img (np.ndarray): Image on which to draw bboxes.
        bbox_tlbr (np.ndarray): Mx4 array of M detections.
        class_prob (np.ndarray): Array of M elements corresponding to predicted
            class probabilities for each bbox.
        class_idx (np.ndarray): Array of M elements corresponding to the
            class index with the greatest probability for each bbox.
        class_names (list): List of all class names in order.
    """
    colors = None
    if class_names is not None:
        colors = dict()
        num_colors = len(class_names)
        colors = list(unique_colors(num_colors))

    for i, (tl_x, tl_y, br_x, br_y) in enumerate(bbox_tlbr):
        bbox_text = []
        if colors is not None:
            color = colors[class_idx[i]]
        else:
            color = (0, 255, 0)

        if class_names is not None:
            bbox_text.append(class_names[class_idx[i]])
        elif class_idx is not None:
            bbox_text.append(str(class_idx[i]))

        if class_prob is not None:
            bbox_text.append("({:.2f})".format(class_prob[i]))

        bbox_text = " ".join(bbox_text)

        cv2.rectangle(
            img, (tl_x, tl_y), (br_x, br_y), color=color, thickness=2
        )

        if bbox_text:
            cv2.rectangle(
                img, (tl_x + 1, tl_y + 1),
                (tl_x + int(8 * len(bbox_text)), tl_y + 18),
                color=(20, 20, 20), thickness=cv2.FILLED
            )
            cv2.putText(
                img, bbox_text, (tl_x + 1, tl_y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), thickness=1
            )


def _non_max_suppression(bbox_tlbr, prob, iou_thresh=0.3):
    """
    Perform non-maximum suppression on an array of bboxes and return the
    indices of detections to retain.

    Derived from:
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Args:
        bbox_tlbr (np.ndarray): An Mx4 array of bboxes (consisting of M
            detections/bboxes), where bbox_tlbr[:, :4] represent the four
            bbox coordinates.
        prob (np.ndarray): An array of M elements corresponding to the
            max class probability of each detection/bbox.

    Returns:
        List of bbox indices to keep (ie, discard everything except
        `bbox_tlbr[idxs_to_keep]`).
    """

    # Compute area of each bbox.
    area = (
        ((bbox_tlbr[:,2] - bbox_tlbr[:,0]) + 1)
        * ((bbox_tlbr[:,3] - bbox_tlbr[:,1]) + 1)
    )

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
        overlaps_tl_x = np.maximum(bbox_tlbr[curr_idx, 0], bbox_tlbr[idxs, 0])
        overlaps_tl_y = np.maximum(bbox_tlbr[curr_idx, 1], bbox_tlbr[idxs, 1])
        overlaps_br_x = np.minimum(bbox_tlbr[curr_idx, 2], bbox_tlbr[idxs, 2])
        overlaps_br_y = np.minimum(bbox_tlbr[curr_idx, 3], bbox_tlbr[idxs, 3])

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


def non_max_suppression(bbox_tlbr, class_prob, class_idx=None, iou_thresh=0.3):
    """
    Perform non-maximum suppression (NMS) of bounding boxes. If `class_idx` is
    provided, per-class NMS is performed by performing NMS on each class and
    combining the results. Else, bboxes are suppressed without regard for
    class.

    Args:
        bbox_tlbr (np.ndarray): Mx4 array of M bounding boxes, where dim 1
            indices are: top left x, top left y, bottom right x, bottom
            right y.
        class_prob (np.ndarray): Array of M elements corresponding to predicted
            class probabilities for each bbox.
        class_idx (np.ndarray): Array of M elements corresponding to the
            class index with the greatest probability for each bbox. If
            provided, per-class NMS is performed; else, all bboxes are
            treated as a single class.
        iou_thresh (float): Intersection over union (IOU) threshold for
            bbox to be considered a duplicate. 0 <= `iou_thresh` < 1.

    Returns:
        List of bbox indices to keep (ie, discard everything except
        `bbox_tlbr[idxs_to_keep]`).
    """

    if class_idx is not None:
        # Perform per-class non-maximum suppression.
        idxs_to_keep = []

        # Set of unique class indices.
        unique_class_idxs = set(class_idx)

        for class_ in unique_class_idxs:
            # Bboxes corresponding to the current class index.
            curr_class_mask = np.where(class_idx == class_)[0]
            curr_class_bbox = bbox_tlbr[curr_class_mask]
            curr_class_prob = class_prob[curr_class_mask]

            curr_class_idxs_to_keep = _non_max_suppression(
                curr_class_bbox, curr_class_prob, iou_thresh
            )
            idxs_to_keep.extend(curr_class_mask[curr_class_idxs_to_keep].tolist())
    else:
        idxs_to_keep = _non_max_suppression(bbox_tlbr, class_prob, iou_thresh)
    return idxs_to_keep


def cxywh_to_tlbr(bbox_xywh):
    """
    Args:
        bbox_xywh (np.array): An MxN array of detections where bbox_xywh[:, :4]
            correspond to coordinates (center x, center y, width, height).

    Returns:
        An MxN array of detections where bbox_tlbr[:, :4] correspond to
        coordinates (top left x, top left y, bottom right x, bottom right y).
    """

    bbox_tlbr = np.copy(bbox_xywh)
    bbox_tlbr[:, :2] = bbox_xywh[:, :2] - (bbox_xywh[:, 2:4] // 2)
    bbox_tlbr[:, 2:4] = bbox_xywh[:, :2] + (bbox_xywh[:, 2:4] // 2)
    return bbox_tlbr


def do_inference(
    net, image, device="cuda", prob_thresh=0.12, nms_iou_thresh=0.3, resize=True
):
    """
    Run inference on an image and return the corresponding bbox coordinates,
    bbox class probabilities, and bbox class indices.

    Args:
        net (torch.nn.Module): Instance of network class.
        image (np.ndarray): Image array.
        device (str): Device for inference (eg, "cpu", "cuda").
        prob_thresh (float): Probability threshold for detections to keep.
            0 <= prob_thresh < 1.
        nms_iou_thresh (float): Intersection over union (IOU) threshold for
            non-maximum suppression (NMS). Per-class NMS is performed.
        resize (bool): If True, resize `image` to dimensions given by the
            `net_info` attribute/block of `net` (from the Darknet .cfg file).

    Returns:
        bbox_tlbr (np.ndarray): Mx4 array of bbox top left/bottom right coords
        class_prob (np.ndarray): Array of M predicted class probabilities.
        class_idx (np.ndarray): Array of M predicted class indices.
    """
    orig_rows, orig_cols = image.shape[:2]
    net_info = net.net_info
    if resize and image.shape[:2] != [net_info["height"], net_info["width"]]:
        image = cv2.resize(image, (net_info["height"], net_info["width"]))

    image = np.transpose(np.flip(image, 2), (2, 0, 1)).astype(np.float32) / 255.
    inp = torch.tensor(image, device=device).unsqueeze(0)

    out = net.forward(inp)

    bbox_xywhs = out["bbox_xywhs"].detach().cpu().numpy()
    class_idx = out["max_class_idx"].cpu().numpy()

    bbox_xywh = bbox_xywhs[:, :4]
    class_prob = bbox_xywhs[:, 4]

    thresh_mask = class_prob >= prob_thresh

    bbox_xywh = bbox_xywh[thresh_mask, :]
    class_idx = class_idx[thresh_mask]
    class_prob = class_prob[thresh_mask]

    bbox_xywh[:, [0, 2]] *= orig_cols
    bbox_xywh[:, [1, 3]] *= orig_rows
    bbox_xywh = bbox_xywh.astype(np.int)

    bbox_tlbr = cxywh_to_tlbr(bbox_xywh)
    idxs_to_keep = non_max_suppression(
        bbox_tlbr, class_prob, class_idx=class_idx, iou_thresh=nms_iou_thresh
    )
    bbox_tlbr = bbox_tlbr[idxs_to_keep, :]
    class_idx = class_idx[idxs_to_keep]

    return bbox_tlbr, class_prob, class_idx


def detect_in_cam(
    net, cam_id=0, device="cuda", class_names=None, show_fps=False, frames=None
):
    """
    Run and display real-time inference on a webcam stream.

    Performs inference on a webcam stream, draw bounding boxes on the frame,
    and display the resulting video in real time.

    Args:
        net (torch.nn.Module): Instance of network class.
        cam_id (int): Camera device id.
        device (str): Device for inference (eg, "cpu", "cuda").
        class_names (list): List of all model class names in order.
        show_fps (bool): Whether to display current frames processed per second.
        frames (list): Optional list to populate with frames being displayed;
            can be used to write or further process frames after this function
            completes. Because mutables (like lists) are passed by reference
            and are modified in-place, this function has no return value.
    """
    video_getter = VideoGetter(cam_id).start()
    video_shower = VideoShower(video_getter.frame, "YOLOv3").start()

    if show_fps:
        # Number of frames to average for computing FPS.
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
        bbox_tlbr, _, class_idx = do_inference(net, frame, device=device)
        draw_boxes(
            frame, bbox_tlbr, class_idx=class_idx, class_names=class_names
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
    """
    Run and optionally display inference on a video file.

    Performs inference on a video, draw bounding boxes on the frame,
    and optionally display the resulting video.

    Args:
        net (torch.nn.Module): Instance of network class.
        filepath (str): Path to video file.
        device (str): Device for inference (eg, "cpu", "cuda").
        cam_id (int): Camera device id.
        class_names (list): List of all model class names in order.
        frames (list): Optional list to populate with frames being displayed;
            can be used to write or further process frames after this function
            completes. Because mutables (like lists) are passed by reference
            and are modified in-place, this function has no return value.
        show_video (bool): Whether to display processed video during processing.
    """
    cap = cv2.VideoCapture(filepath)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        bbox_tlbr, _, class_idx = do_inference(net, frame, device=device)
        draw_boxes(
            frame, bbox_tlbr, class_idx=class_idx, class_names=class_names
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
    """
    Write provided frames to an .mp4 video.

    Args:
        frames (list): List of frames (np.ndarray).
        fps (int): Framerate (frames per second) of the output video.
        filepath (str): Path to output video file.
    """
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
        bbox_tlbr, _, class_idx = do_inference(net, image, device=device)
        draw_boxes(image, bboxes, class_idx=class_idx, class_names=class_names)
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
