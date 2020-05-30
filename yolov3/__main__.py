import argparse
import os
import pathlib
import time

import cv2
import torch

import yolov3
from .inference import (
    cam_thread_both, cam_thread_get, cam_thread_show, cam_single_thread
)


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode", choices=["both", "get", "show", "single"], help="Threading mode"
    )
    parser.add_argument(
        "-C", "--cam-id", type=int, default=0, help="Video capture webcam id"
    )
    parser.add_argument(
        "-c", "--config", type=pathlib.Path, required=True, metavar="<path>",
        help="[Required] Path to Darknet model config file."
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", metavar="<device>",
        help="Device for inference ('cpu', 'cuda'). [Default 'cuda']"
    )
    parser.add_argument(
        "-i", "--iou-thresh", type=float, default=0.3, metavar="<iou>",
        help="Non-maximum suppression IOU threshold. [Default 0.3]"
    )
    parser.add_argument(
        "-n", "--class-names", type=pathlib.Path, metavar="<path>",
        help="Path to text file of class names. If omitted, class index is \
            displayed instead of name."
    )
    parser.add_argument(
        "-p", "--prob-thresh", type=float, default=0.05, metavar="<prob>",
        help="Detection probability threshold. [Default 0.05]"
    )
    parser.add_argument(
        "-w", "--weights", type=pathlib.Path, required=True, metavar="<path>",
        help="[Required] Path to Darknet model weights file."
    )

    parser.add_argument(
        "-o", "--output", type=pathlib.Path, metavar="<path>",
        help="Path for writing output .mp4 file."
    )

    args = vars(parser.parse_args())

    # Expand pathlib Paths and convert to string.
    path_args = ("class_names", "config", "weights", "output")
    for path_arg in path_args:
        if args[path_arg] is not None:
            args[path_arg] = str(args[path_arg].expanduser().absolute())

    device = args["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    net = yolov3.Darknet(args["config"], device=device)
    net.load_weights(args["weights"])
    net.eval()

    if device.startswith("cuda"):
        net.cuda(device=device)

    class_names = None
    if args["class_names"] is not None and os.path.isfile(args["class_names"]):
        with open(args["class_names"], "r") as f:
            class_names = [line.strip() for line in f.readlines()]


    frames = []
    inference_kwargs = {
        "cam_id": args["cam_id"],
        "device": device,
        "prob_thresh": args["prob_thresh"],
        "nms_iou_thresh": args["iou_thresh"],
        "class_names": class_names,
        "frames": frames
    }

    start_time = time.time()
    try:
        if args["mode"] == "both":
            cam_thread_both(net, **inference_kwargs)
        elif args["mode"] == "get":
            cam_thread_get(net, **inference_kwargs)
        elif args["mode"] == "show":
            cam_thread_show(net, **inference_kwargs)
        elif args["mode"] == "single":
            cam_single_thread(net, **inference_kwargs)
    except Exception as e:
        raise e
    finally:
        fps = 1 / ((time.time() - start_time) / len(frames))
        print(fps)
        if args["output"] and frames:
            # Get average FPS and write output at that framerate.
            write_mp4(frames, fps, args["output"])
