"""Grab one camera frame and test which words YOLO-World responds to."""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLOWorld
import cv2

WORDS = ['bench', 'wooden bench', 'long bench', 'seat', 'sofa', 'couch', 'counter',
         'sign', 'signboard', 'shop sign', 'store sign', 'banner', 'billboard', 'poster', 'nameplate',
         'trash can', 'trash bin', 'garbage bin', 'green bin', 'bucket',
         'potted plant', 'plant', 'bush',
         'shelf', 'bookshelf', 'cabinet', 'table', 'desk', 'cardboard box', 'crate']


def main():
    rclpy.init()
    node = Node('prompt_test')
    frame = {}
    node.create_subscription(Image, '/camera/image', lambda m: frame.setdefault('msg', m), 10)
    while 'msg' not in frame:
        rclpy.spin_once(node, timeout_sec=0.5)
    rgb = CvBridge().imgmsg_to_cv2(frame['msg'], desired_encoding='rgb8')
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    name = sys.argv[1] if len(sys.argv) > 1 else 'frame'
    cv2.imwrite(f'{name}.png', bgr)
    print(f'saved {name}.png')

    model = YOLOWorld('yolov8s-worldv2.pt')
    print(f'{"word":16s} {"found":>5s} {"best":>6s}  boxes')
    for w in WORDS:
        model.set_classes([w])
        r = model.predict(bgr, conf=0.05, verbose=False)[0]
        confs = [float(c) for c in r.boxes.conf]
        boxes = ['(%d,%d)-(%d,%d)' % tuple(int(v) for v in b) for b in r.boxes.xyxy[:2]]
        best = max(confs) if confs else 0.0
        print(f'{w:16s} {len(confs):5d} {best:6.2f}  {" ".join(boxes)}')
    rclpy.shutdown()
