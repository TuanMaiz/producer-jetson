# type: ignore
from datetime import datetime
from cam_pb2 import Raw, Result
import numpy as np
import cv2 as cv
import torch

def encodeToRaw(frame, id):
    m = Raw()
    gray = cv.resize(frame, [416, 416])
    _, buffer = cv.imencode(".jpg", gray)
    m.cameraID = id
    m.frame = buffer.tobytes()
    m.timestamp = str(int(datetime.now().timestamp()))
    return m.SerializeToString()

def decodeFromRaw(buffer):
    m = Raw()
    m.ParseFromString(buffer)
    # decode frame to img
    nparr = np.frombuffer(m.frame, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return {
        "cameraID": m.cameraID,
        "frame": m.frame,
        "timestamp": m.timestamp,
        "img": img,
    }

def encodeResult(result):
    m = Result()
    m.cameraID = result["cameraID"]
    m.frame = encodeToBytes(result["frame"])
    m.timestamp = result["timestamp"]
    m.result = result["result"]
    return m.SerializeToString()

def decodeResult(buffer):
    m = Result()
    m.ParseFromString(buffer)
    nparr = np.frombuffer(m.frame, np.uint8)
    # decode image
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return {
        "cameraID": m.cameraID,
        "frame": m.frame,
        "timestamp": m.timestamp,
        "img": img,
        "result": m.result,
    }

def encodeToBytes(frame):
    _, buffer = cv.imencode(".jpg", frame)
    return buffer.tobytes()

def decodeFromBytes(buffer):
    nparr = np.frombuffer(buffer, np.uint8)
    # decode image
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return img

# raw = cam_pb2.Raw()
# raw.ParseFromString(msg.value())
# id = raw.cameraID
# time = datetime.fromtimestamp(int(raw.timestamp))
# print(id, time)
# frame = decodeFromBytes(raw.frame)
# if not (frame is None):
#     cv2.imshow("frame", frame)
# if cv2.waitKey(1) == ord("q"):
#     break