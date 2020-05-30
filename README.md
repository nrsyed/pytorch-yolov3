# Multithreading demonstration and benchmarks

## Results

System 1:
* CPU: i5-6500
* RAM: 48GB @ 2133 MHz
* GPU0: RTX 2080 Ti
* GPU1: GTX 950

Thread both (get and show in separate threads; process in main thread):
* RTX 2080 Ti
  * yolov3: 26.5
  * yolov3-tiny: 147.3
* GTX 950
  * yolov3: 6.8
  * yolov3-tiny: 61.2
* CPU
	* yolov3: 1.5
	* yolov3-tiny: 13.1

Thread get (get frames in separate thread; process and show in main thread):
* RTX 2080 Ti
  * yolov3: 25.1
  * yolov3-tiny: 107.6
* GTX 950
  * yolov3: 7.4
  * yolov3-tiny: 55.5
* CPU
	* yolov3: 1.4
	* yolov3-tiny: 12.2

Thread show (show frames in separate thread; get and process in main thread):
* RTX 2080 Ti
  * yolov3: 14.4
  * yolov3-tiny: 14.7
* GTX 950
  * yolov3: 7.5
  * yolov3-tiny: 14.6
* CPU
	* yolov3: 1.5
	* yolov3-tiny: 13.2

Single thread (get, process, and show frames in single thread):
* RTX 2080 Ti
  * yolov3: 14.3
  * yolov3-tiny: 14.4
* GTX 950
  * yolov3: 7.3
  * yolov3-tiny: 14.4
* CPU
	* yolov3: 1.4
	* yolov3-tiny: 13.0

## Requirements

+ Python &ge; 3.6

## Installation

```
git clone https://github.com/nrsyed/pytorch-yolov3.git
cd pytorch-yolov3
git checkout demo/multithread
pip install -e .
./get_weights.sh
```

## Usage

TODO

## Acknowledgments
The model config files in the `models` directory were obtained from
https://github.com/pjreddie/darknet/tree/master/cfg. The list of COCO class
names in the `models` directory was obtained from
https://github.com/pjreddie/darknet/tree/master/data and modified to match
the COCO annotation category names. Weights for yolov3, yolov3-tiny, and
yolov3-spp can be found on Joseph Redmon's website at
https://pjreddie.com/darknet/yolo/. Sample images were taken from the original
COCO dataset (http://cocodataset.org).
