from datetime import datetime
import time

import cv2


def webcam_record(src=0, output_path=None):
    """
    Record video from a video capture source and write to .mp4 file. Output
    FPS is equal to average FPS over the duration of the recording.
    """
    if output_path is None:
        output_path = datetime.now().strftime("%Y%m%d%H%M%s.mp4")

    if not output_path.endswith(".mp4"):
        output_path += ".mp4"

    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), "VideoCapture not opened"

    frames = []
    start_time = time.time()
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        frames.append(frame)
        cv2.imshow("Stream", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    end_time = time.time()
    cap.release()

    assert frames, "No frames captured"

    average_fps = int(1 / ((end_time - start_time) / len(frames)))
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), average_fps, (w, h)
    )

    for frame in frames:
        writer.write(frame)
    writer.release()
