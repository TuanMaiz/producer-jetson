from datetime import datetime
from confluent_kafka import Producer
from serde import encodeToRaw
from dotenv import load_dotenv
import concurrent.futures
import torch
import argparse
import cv2
import sys
import os
import numpy as np
import time

# List of video are used for object detection
videos = [("Traffic1.m4v",0), ("Traffic2.m4v",1), ("Traffic3.m4v",2), ("Traffic4.mp4",3)]

########################### Convert model to np.array (frame has been detected)
def score_frame(frame):
    model.to('cuda')
    frame = [frame]
    results = model(frame)
    
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def class_to_label(x):
    return int(x)

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, str(model.names[class_to_label(labels[i])]) + " " + str(round(float(row[4]),2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame
###############################################################

def delivery_callback(err, msg):
    if err:
        sys.stderr.write("%% Message failed delivery: %s\n" % err)
    else:
        sys.stderr.write(
            "%% Message delivered to %s [%d] @ %d\n"
            % (msg.topic(), msg.partition(), msg.offset())
        )
# Send list of detected frame to each partition in kafka
def send(video):
    url, id = video
    vid = cv2.VideoCapture(url)
    while True:
        start_time = time.perf_counter()
        ret, frm = vid.read()
        if ret is False:
            cv2.destroyAllWindows()
            vid.release()
            break

        # Resize frame to 416x416
        gray = cv2.resize(frm, (416, 416))

        # pass gray frame has been resized to covert class
        results = score_frame(gray)
        frame = plot_boxes(results, gray)
        end_time = time.perf_counter()

        # display fps and time using opencv
        dt = str(datetime.now())
        fps = 1 / np.round(end_time - start_time, 3)
        cv2.putText(frame, f'FPS: {int(fps)}' + " " + dt, (20, 70),cv2.FONT_HERSHEY_DUPLEX, 0.7, (172, 7, 237), 2)

        # encode msg and send to kafka topic
        p.produce(
            args.topic, 
            encodeToRaw(frame, str(id)),
            callback=delivery_callback, 
            partition=id
        )
        p.poll(1)
    vid.release()

if __name__ == "__main__":

    # pass the CLI argument 
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", type=str)
    args = parser.parse_args()

    # Load yolov5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    # Load variables from .env file
    load_dotenv()

    # kafka config
    broker = os.environ.get("BROKER")
    conf = {"bootstrap.servers": broker}
    p = Producer(**conf)
    
    # Send messages to multiple partition in topic with parallel method 
    with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(send, videos)

