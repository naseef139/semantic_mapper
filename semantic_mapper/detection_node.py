import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import numpy as np
from ultralytics import YOLOWorld
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs   # lets tf2 transform PointStamped messages

CLASSES = ['bench', 'chair', 'table', 'potted plant', 'trash can', 'sign',
           'fire extinguisher', 'vending machine', 'person']

def box_depth(depth, x1, y1, x2, y2, shrink=0.2, bin_size=0.1, min_share=0.15):
    """Nearest-mode depth (metres) of the object inside a box, or None."""
    # 1. middle part of the box
    dx, dy = int((x2 - x1) * shrink), int((y2 - y1) * shrink)
    patch = depth[y1 + dy:y2 - dy, x1 + dx:x2 - dx]      # rows first, then columns
    # 2. valid depths only
    vals = patch[np.isfinite(patch)]
    if vals.size < 10:
        return None
    # 3. histogram with 0.1 m bins
    edges = np.arange(vals.min(), vals.max() + bin_size, bin_size)
    if edges.size < 2:
        return float(np.median(vals))
    counts, edges = np.histogram(vals, bins=edges)
    # 4. nearest bin that holds enough pixels
    for i, c in enumerate(counts):
        if c >= min_share * vals.size:
            near = vals[(vals >= edges[i]) & (vals <= edges[i] + 2 * bin_size)]
            return float(np.median(near))
    return float(np.median(vals))
    

def back_project(u, v, z, K):
    """Pixel (u, v) with depth z -> point (X, Y, Z) in the camera optical frame."""
    fx, fy, cx, cy = K
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return X, Y, z
    

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.bridge = CvBridge()
        self.K = None            # fx, fy, cx, cy once we have them
        self.last_print = 0.0
        
        self.model = YOLOWorld('yolov8s-worldv2.pt')
        self.model.set_classes(CLASSES)
        self.get_logger().info(f'YOLO-World ready, classes: {CLASSES}')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_mapper/markers', 10)        
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
        if self.K is None:
            return
                                   
        result = self.model.predict(rgb, conf=0.25, verbose=False)[0]
        if len(result.boxes) == 0:
            self.get_logger().info('no objects')
            return
            
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL          # remove last frame's markers
        markers.markers.append(clear)            
        for box in result.boxes:
            name = CLASSES[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            z = box_depth(depth, x1, y1, x2, y2)
            if z is None:
                self.get_logger().info(f'{name:18s} conf={conf:.2f}  no valid depth, skipped')
                continue

            u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0          # box centre
            X, Y, Z = back_project(u, v, z, self.K)
            dist = float(np.sqrt(X * X + Y * Y + Z * Z))      # straight-line distance
            # camera frame -> map frame
            p_cam = PointStamped()
            p_cam.header = rgb_msg.header            # frame = camera_optical_frame, time = image time
            p_cam.point.x, p_cam.point.y, p_cam.point.z = float(X), float(Y), float(Z)
            try:
                p_map = self.tf_buffer.transform(p_cam, 'map', timeout=Duration(seconds=0.2))
            except Exception as e:
                self.get_logger().warn(f'no transform to map yet: {e}')
                continue
            mx, my, mz = p_map.point.x, p_map.point.y, p_map.point.z
            self.get_logger().info(
                f'{name:18s} conf={conf:.2f}  cam=({X:+.2f}, {Y:+.2f}, {Z:+.2f})  '
                f'map=({mx:+.2f}, {my:+.2f}, {mz:+.2f})  dist={dist:.2f} m')

            # a red ball on the object and its name above it
            for kind in (Marker.SPHERE, Marker.TEXT_VIEW_FACING):
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = rgb_msg.header.stamp
                m.ns = 'detections'
                m.id = len(markers.markers)
                m.type = kind
                m.action = Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = mx, my, mz
                m.pose.orientation.w = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.1, 1.0
                if kind == Marker.SPHERE:
                    m.scale.x = m.scale.y = m.scale.z = 0.3
                else:
                    m.pose.position.z = mz + 0.5
                    m.scale.z = 0.35
                    m.color.r = m.color.g = m.color.b = 1.0
                    m.text = f'{name} {conf:.2f}'
                markers.markers.append(m)
                self.marker_pub.publish(markers)

def main():
    rclpy.init()
    rclpy.spin(DetectionNode())
    rclpy.shutdown()
