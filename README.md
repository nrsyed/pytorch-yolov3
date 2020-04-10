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

Run `python test.py --help` for usage information.

## Examples
&#35; TODO
