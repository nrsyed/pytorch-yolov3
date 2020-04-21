# pytorch-yolov3
A pytorch implementation of YOLOv3

## Installation
None necessary. Simply make sure the requirements are installed (preferably in
a virtual environment) with:

```
pip install -r requirements.txt
```

The model config files in the `models` directory were obtained from
https://github.com/pjreddie/darknet/tree/master/cfg. The list of COCO class
names in the `models` directory was obtained from
https://github.com/pjreddie/darknet/tree/master/data. Weights for yolov3,
yolov3-tiny, and yolov3-spp can be found on Joseph Redmon's website at
https://pjreddie.com/darknet/yolo/. For convenience, simply run
`./get_weights.sh` to download weight files to the `models` directory.

Usage, examples, and utilities can be found in `test.py`; see the next section
for more information.

## Usage

Run `python main.py --help` for a complete list of options. Refer to the
[Examples](#examples) section below for representative examples.

### Required arguments

At least one of the following input source arguments is required:

+ `-C`/`--cam` `[cam_id]`: Webcam device ID or path to video stream
	(e.g., RTSP URL); defaults to `0` if none provided.

+ `-I`/`--image` `<path>`: Path to an image file.

+ `-V`/`--video` `<path>`: Path to a video file.


The following arguments are always required:

+ `-c`/`--config` `<path>`: Path to Darknet model .cfg file.

+ `-w`/`--weights` `<path>`: Path to Darknet model .weights file.

## Examples
&#35; TODO
