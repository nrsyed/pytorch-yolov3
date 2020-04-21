import argparse
import os
import pathlib
import time

import cv2
import torch

from darknet import Darknet
import inference


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
        "-C", "--cam", metavar="cam_id", nargs="?", const=0,
        help="Camera or video capture device ID or path. [Default 0]"
    )
    source_args.add_argument(
        "-I", "--image", type=pathlib.Path, metavar="<path>",
        help="Path to image file."
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
        help="Path for writing output image/video file. --cam and --video \
            input source options only support .mp4 output filetype. \
            If --video input source selected, output FPS matches input FPS."
    )
    other_args.add_argument(
        "--show-fps", action="store_true",
        help="Display frames processed per second (for --cam input)."
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
        # If --cam argument is str representation of an int, interpret it as
        # an int device ID. Else interpret as a path to a video capture stream.
        if isinstance(args["cam"], str) and args["cam"].isdigit():
            args["cam"] = int(args["cam"])

    if source == "image":
        image = cv2.imread(args["image"])
        bbox_xywh, _, class_idx = inference.inference(
            net, image, device=device, prob_thresh=args["prob_thresh"],
            nms_iou_thresh=args["iou_thresh"]
        )[0]
        inference.draw_boxes(
            image, bbox_xywh, class_idx=class_idx, class_names=class_names
        )
        if args["output"]:
            cv2.imwrite(args["output"], image)
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
                inference.detect_in_cam(
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
            inference.detect_in_video(
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
