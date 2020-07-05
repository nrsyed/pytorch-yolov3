import argparse
import os
import pathlib
import time
import warnings

import cv2
import torch

import yolov3


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

    source_ = parser.add_argument_group(title="input source [required]")
    source_args = source_.add_mutually_exclusive_group(required=True)
    source_args.add_argument(
        "-C", "--cam", metavar="cam_id", nargs="?", const=0,
        help="Camera or video capture device ID or path. [Default 0]"
    )
    source_args.add_argument(
        "-I", "--image", type=pathlib.Path, metavar="<path>",
        help="Path to image file or directory of images."
    )
    source_args.add_argument(
        "-V", "--video", type=pathlib.Path, metavar="<path>",
        help="Path to video file."
    )

    model_args = parser.add_argument_group(title="model parameters")
    model_args.add_argument(
        "-c", "--config", type=pathlib.Path, required=True, metavar="<path>",
        help="[Required] Path to Darknet model config file."
    )
    model_args.add_argument(
        "-d", "--device", type=str, default="cuda", metavar="<device>",
        help="Device for inference ('cpu', 'cuda'). [Default 'cuda']"
    )
    model_args.add_argument(
        "-i", "--iou-thresh", type=float, default=0.3, metavar="<iou>",
        help="Non-maximum suppression IOU threshold. [Default 0.3]"
    )
    model_args.add_argument(
        "-n", "--class-names", type=pathlib.Path, metavar="<path>",
        help="Path to text file of class names. If omitted, class index is \
            displayed instead of name."
    )
    model_args.add_argument(
        "-p", "--prob-thresh", type=float, default=0.05, metavar="<prob>",
        help="Detection probability threshold. [Default 0.05]"
    )
    model_args.add_argument(
        "-w", "--weights", type=pathlib.Path, required=True, metavar="<path>",
        help="[Required] Path to Darknet model weights file."
    )

    other_args = parser.add_argument_group(title="Output/display options")
    other_args.add_argument(
        "-o", "--output", type=pathlib.Path, metavar="<path>",
        help="Path for writing output video file. Only .mp4 filetype \
            currently supported. If --video input source selected, output \
            FPS matches input FPS."
    )
    other_args.add_argument(
        "--show-fps", action="store_true",
        help="Display frames processed per second (for --cam input)."
    )
    other_args.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    args = vars(parser.parse_args())

    # Expand pathlib Paths and convert to string.
    path_args = (
        "class_names", "config", "weights", "image", "video", "output"
    )
    for path_arg in path_args:
        if args[path_arg] is not None:
            args[path_arg] = str(args[path_arg].expanduser().absolute())

    device = args["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA not available; falling back to CPU. Pass `-d cpu` or ensure "
            "compatible versions of CUDA and pytorch are installed.",
            RuntimeWarning, stacklevel=2
        )
        device = "cpu"

    net = yolov3.Darknet(args["config"], device=device)
    net.load_weights(args["weights"])
    net.eval()

    if device.startswith("cuda"):
        net.cuda(device=device)

    if args["verbose"]:
        if device == "cpu":
            device_name = "CPU"
        else:
            device_name = torch.cuda.get_device_name(net.device)
        print(f"Running model on {device_name}")

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
        # If --cam argument is str representation of an int, interpret it as
        # an int device ID. Else interpret as a path to a video capture stream.
        if isinstance(args["cam"], str) and args["cam"].isdigit():
            args["cam"] = int(args["cam"])

    if source == "image":
        if os.path.isdir(args["image"]):
            image_dir = args["image"]
            fnames = os.listdir(image_dir)
        else:
            image_dir, fname = os.path.split(args["image"])
            fnames = [fname]

        images = []
        for fname in fnames:
            images.append(cv2.imread(os.path.join(image_dir, fname)))

        # TODO: batch images
        results = []
        for image in images:
            results.extend(
                yolov3.inference(
                    net, image, device=device, prob_thresh=args["prob_thresh"],
                    nms_iou_thresh=args["iou_thresh"]
                )
            )

        for image, (bbox_xywh, _, class_idx) in zip(images, results):
            yolov3.draw_boxes(
                image, bbox_xywh, class_idx=class_idx, class_names=class_names
            )
            cv2.imshow("YOLOv3", image)
            cv2.waitKey(0)
    else:
        frames = None
        if args["output"]:
            frames = []

        if source == "cam":
            start_time = time.time()

            # Wrap in try/except block so that output video is written
            # even if an error occurs while streaming webcam input.
            try:
                yolov3.detect_in_cam(
                    net, cam_id=args["cam"], device=device,
                    prob_thresh=args["prob_thresh"],
                    nms_iou_thresh=args["iou_thresh"],
                    class_names=class_names, show_fps=args["show_fps"],
                    frames=frames
                )
            except Exception as e:
                raise e
            finally:
                if args["output"] and frames:
                    # Get average FPS and write output at that framerate.
                    fps = 1 / ((time.time() - start_time) / len(frames))
                    write_mp4(frames, fps, args["output"])
        elif source == "video":
            yolov3.detect_in_video(
                net, filepath=args["video"], device=device,
                prob_thresh=args["prob_thresh"],
                nms_iou_thresh=args["iou_thresh"], class_names=class_names,
                frames=frames
            )
            if args["output"] and frames:
                # Get input video FPS and write output video at same FPS.
                cap = cv2.VideoCapture(args["video"])
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                write_mp4(frames, fps, args["output"])

    cv2.destroyAllWindows()
