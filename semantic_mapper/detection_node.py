import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import numpy as np
from ultralytics import YOLOWorld

CLASSES = ['bench', 'chair', 'table', 'potted plant', 'trash can', 'sign',
           'fire extinguisher', 'vending machine', 'person']

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.bridge = CvBridge()
        self.K = None            # fx, fy, cx, cy once we have them
        self.last_print = 0.0
        
        self.model = YOLOWorld('yolov8s-worldv2.pt')
        self.model.set_classes(CLASSES)
        self.get_logger().info(f'YOLO-World ready, classes: {CLASSES}')
        
        # camera_info: needed once
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 10)

        # colour + depth must arrive as a matched pair
        rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.frame_cb)

    def info_cb(self, msg):
        if self.K is None:
            self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])   # fx, fy, cx, cy
            self.get_logger().info(f'K = {self.K}')

    def frame_cb(self, rgb_msg, depth_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_print < 1.0:
            return
        self.last_print = now

        h, w = depth.shape[:2]
        centre = float(depth[h // 2, w // 2])
        
        valid = np.isfinite(depth)
        pct = 100.0 * valid.mean()
        nearest = float(depth[valid].min()) if valid.any() else float('nan')
        median = float(np.median(depth[valid])) if valid.any() else float('nan')
                       
        result = self.model.predict(rgb, conf=0.25, verbose=False)[0]
        if len(result.boxes) == 0:
            self.get_logger().info('no objects')
            return
        for box in result.boxes:
            name = CLASSES[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            self.get_logger().info(f'{name:18s} conf={conf:.2f}  box=({x1},{y1})-({x2},{y2})')


def main():
    rclpy.init()
    rclpy.spin(DetectionNode())
    rclpy.shutdown()
