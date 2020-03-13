#!/usr/bin/env python
from plen_real.socket_comms import SocketServer
from plen_real.object_detector import ObjectDetector

if __name__ == "__main__":
    obj_detect = ObjectDetector()
    server = SocketServer()

    while True:
        obj_detect.detect()
        msg = [
            obj_detect.x_position, obj_detect.y_position, obj_detect.z_position
        ]
        server.broadcast(msg)
