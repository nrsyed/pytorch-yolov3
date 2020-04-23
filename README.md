## Installation

Python &ge; 3.6 required. Install the requirements and download weight files to
the `models` directory by running:

```
pip install -r requirements.txt
./get_weights.sh
```

The model config files in the `models` directory were obtained from
https://github.com/pjreddie/darknet/tree/master/cfg. The list of COCO class
names in the `models` directory was obtained from
https://github.com/pjreddie/darknet/tree/master/data and modified to match
the COCO annotation category names. Weights for yolov3, yolov3-tiny, and
yolov3-spp can be found on Joseph Redmon's website at
https://pjreddie.com/darknet/yolo/.

## Usage

Run `python main.py --help` for a complete list of options. Refer to the
[Examples](#examples) section for representative examples.

### Required arguments

One of the following input source arguments is required:

+ `-C`/`--cam` `[cam_id]`: Webcam device ID or path to video stream
	(e.g., RTSP URL); defaults to `0` if none provided.
+ `-I`/`--image` `<path>`: Path to an image file or directory of images.
+ `-V`/`--video` `<path>`: Path to a video file.

The following arguments are always required:

+ `-c`/`--config` `<path>`: Path to Darknet model .cfg file.
+ `-w`/`--weights` `<path>`: Path to Darknet model .weights file.

### Optional arguments

+ `-d`/`--device` `<device>`: Device on which to load the model (e.g., `cpu`,
	`cuda`, `cuda:1`)
+ `-i`/`--iou-thresh` `<iou>`: Non-maximum suppression IOU threshold.
+ `-n`/`--class-names` `<path>`: Path to text file of class names containing
	one class name per line. If omitted, the class index is displayed on
	the resulting image instead of the class name.
+ `-p`/`--prob-thresh` `<prob>`: Detection probability threshold; predictions
	with a computed probability below this threshold are ignored.
+ `-o`/`--output` `<path>`: Path for output video file (use with `--cam` or
  `--video` input source options). Only .mp4 filetype supported. For `--cam`,
  the output video framerate (FPS) is equal to the average framerate over the
  duration the recording. For `--video`, the output video framerate is equal to
	input file framerate.
+ `--show-fps`: Display frames processed per second (`--cam` input only).


## Examples
&#35; TODO
