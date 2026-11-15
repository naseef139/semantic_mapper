"""Detection node: finds objects with YOLO-World, places them in the map frame.

Publishes two versions of every detection, so both position methods can be compared in one run:
  /semantic_mapper/detections            box centre + nearest-mode depth (original method)
  /semantic_mapper/detections_centroid   median 3D point of the object's pixels (Fix 2)
"""
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2
import easyocr
from ultralytics import YOLOWorld
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs   # lets tf2 transform PointStamped messages
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

# words given to the detector (chosen by the prompt tests on our mall)
WORDS = ['bench', 'garbage bin', 'desk', 'potted tree', 'signboard', 'person']
# detector word -> name stored on the map
LABEL = {'garbage bin': 'bin', 'desk': 'table', 'potted tree': 'potted plant', 'signboard': 'sign'}
# how far an object can reach in depth (m): used to collect all of its pixels
EXTENT = {'sign': 2.4, 'bench': 1.8, 'table': 0.8, 'potted plant': 0.8, 'bin': 0.5}
LONG = {'sign', 'bench'}            # long objects: skip their centroid if the box is cut off at the image edge


def box_depth(depth, x1, y1, x2, y2, shrink=0.2, bin_size=0.1, min_share=0.06):
    """Nearest-mode depth (metres) of the object inside a box, or None."""
    dx, dy = int((x2 - x1) * shrink), int((y2 - y1) * shrink)
    patch = depth[y1 + dy:y2 - dy, x1 + dx:x2 - dx]      # rows first, then columns
    vals = patch[np.isfinite(patch)]
    if vals.size < 10:
        return None
    edges = np.arange(vals.min(), vals.max() + bin_size, bin_size)
    if edges.size < 2:
        return float(np.median(vals))
    counts, edges = np.histogram(vals, bins=edges)
    for i, c in enumerate(counts):
        if c >= min_share * vals.size:
            near = vals[(vals >= edges[i]) & (vals <= edges[i] + 2 * bin_size)]
            return float(np.median(near))
    return float(np.median(vals))


def back_project(u, v, z, K):
    """Pixel (u, v) with depth z -> point (X, Y, Z) in the camera optical frame."""
    fx, fy, cx, cy = K
    return (u - cx) * z / fx, (v - cy) * z / fy, z


def object_point(depth, x1, y1, x2, y2, K, z_near, extent, shrink=0.1):
    """Fix 2: median 3D point of all pixels that belong to the object (near end to far end, no floor)."""
    fx, fy, cx, cy = K
    dx, dy = int((x2 - x1) * shrink), int((y2 - y1) * shrink)
    v, u = np.mgrid[y1 + dy:y2 - dy, x1 + dx:x2 - dx]
    z = depth[y1 + dy:y2 - dy, x1 + dx:x2 - dx]
    keep = np.isfinite(z) & (z >= z_near - 0.1) & (z <= z_near + extent)
    if keep.sum() < 10:
        return None
    z = z[keep]
    X = (u[keep] - cx) * z / fx
    Y = (v[keep] - cy) * z / fy
    not_floor = Y < 0.5                  # camera is 0.60 m up; Y > 0.5 m is within 0.1 m of the floor
    if not_floor.sum() >= 10:
        X, Y, z = X[not_floor], Y[not_floor], z[not_floor]
    return float(np.median(X)), float(np.median(Y)), float(np.median(z))


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.bridge = CvBridge()
        self.K = None            # fx, fy, cx, cy once we have them
        self.last_print = 0.0

        self.declare_parameter('model', 'yolov8l-worldv2.pt')
        self.model = YOLOWorld(self.get_parameter('model').value)
        self.model.set_classes(WORDS)
        self.get_logger().info(f'YOLO-World ready, words: {WORDS}')
        self.ocr = easyocr.Reader(['en'])
        self.get_logger().info('EasyOCR ready')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_mapper/markers', 10)
        self.det_pub = self.create_publisher(Detection3DArray, '/semantic_mapper/detections', 10)
        self.det_c_pub = self.create_publisher(Detection3DArray, '/semantic_mapper/detections_centroid', 10)
        self.annot_pub = self.create_publisher(Image, '/semantic_mapper/annotated', 10)
        self.max_range = self.declare_parameter('max_range', 8.0).value
        self.conf = self.declare_parameter('conf', 0.20).value

        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 10)
        rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.frame_cb)

    def read_sign(self, rgb, x1, y1, x2, y2):
        """Read the text on a detected sign. Returns '' if nothing is read."""
        h, w = rgb.shape[:2]
        pad = 6
        crop = rgb[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            return ''
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        results = self.ocr.readtext(crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        pieces = [(box[0][0], t) for box, t, c in results if c >= 0.3 and len(t) >= 2]
        text = ''.join(t for _, t in sorted(pieces))        # join the pieces from left to right
        return text if len(text) >= 3 else ''

    def info_cb(self, msg):
        if self.K is None:
            self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])   # fx, fy, cx, cy
            self.get_logger().info(f'K = {self.K}')

    def to_map(self, X, Y, Z, header):
        """Camera-frame point -> PointStamped in the map frame, or None."""
        p = PointStamped()
        p.header.frame_id = header.frame_id
        p.header.stamp = header.stamp
        p.point.x, p.point.y, p.point.z = float(X), float(Y), float(Z)
        try:
            return self.tf_buffer.transform(p, 'map')
        except Exception:
            try:
                p.header.stamp = Time().to_msg()             # use the newest transform
                return self.tf_buffer.transform(p, 'map')
            except Exception as e:
                self.get_logger().warn(f'no transform to map yet: {e}')
                return None

    @staticmethod
    def make_det(header, name, conf, text, p_map, width, height):
        det = Detection3D()
        det.header = header
        det.id = text
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = name
        hyp.hypothesis.score = conf
        q = p_map.point
        hyp.pose.pose.position.x, hyp.pose.pose.position.y, hyp.pose.pose.position.z = q.x, q.y, q.z
        hyp.pose.pose.orientation.w = 1.0
        det.results.append(hyp)
        det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.center.position.z = q.x, q.y, q.z
        det.bbox.center.orientation.w = 1.0
        det.bbox.size.x = det.bbox.size.y = width
        det.bbox.size.z = height
        return det

    def frame_cb(self, rgb_msg, depth_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_print < 1.0:
            return
        self.last_print = now
        if self.K is None:
            return
        h, w = depth.shape[:2]

        bgr = np.ascontiguousarray(rgb[:, :, ::-1])          # the detector expects blue-green-red
        result = self.model.predict(bgr, conf=self.conf, verbose=False)[0]
        annot = self.bridge.cv2_to_imgmsg(result.plot(), encoding='bgr8')
        annot.header = rgb_msg.header
        self.annot_pub.publish(annot)

        dets, dets_c = Detection3DArray(), Detection3DArray()
        for arr in (dets, dets_c):
            arr.header.frame_id = 'map'
            arr.header.stamp = rgb_msg.header.stamp
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for box in result.boxes:
            word = WORDS[int(box.cls[0])]
            name = LABEL.get(word, word)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            z = box_depth(depth, x1, y1, x2, y2)
            if z is None:
                self.get_logger().info(f'{name:14s} conf={conf:.2f}  no valid depth, skipped')
                continue
            u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0           # box centre
            X, Y, Z = back_project(u, v, z, self.K)
            dist = float(np.sqrt(X * X + Y * Y + Z * Z))
            if dist > self.max_range:
                self.get_logger().info(f'{name:14s} conf={conf:.2f}  {dist:.1f} m away, beyond max range')
                continue

            p_map = self.to_map(X, Y, Z, rgb_msg.header)      # original method
            if p_map is None:
                continue
            cut = name in LONG and (x1 <= 2 or x2 >= w - 3)
            pc = None if cut else object_point(depth, x1, y1, x2, y2, self.K, z, EXTENT.get(name, 0.6))
            pc_map = self.to_map(*pc, rgb_msg.header) if pc is not None else None   # Fix 2

            text = self.read_sign(rgb, x1, y1, x2, y2) if name == 'sign' else ''
            a = p_map.point
            line = f'{name:14s} conf={conf:.2f}  box=({a.x:+.2f}, {a.y:+.2f}, {a.z:+.2f})'
            if pc_map is not None:
                c = pc_map.point
                line += f'  centroid=({c.x:+.2f}, {c.y:+.2f}, {c.z:+.2f})'
            elif cut:
                line += '  centroid=cut off at image edge'
            self.get_logger().info(line + (f'  text="{text}"' if text else ''))

            width = float((x2 - x1) * z / self.K[0])
            height = float((y2 - y1) * z / self.K[1])
            dets.detections.append(self.make_det(dets.header, name, conf, text, p_map, width, height))
            if pc_map is not None:
                dets_c.detections.append(self.make_det(dets_c.header, name, conf, text, pc_map, width, height))

            q = (pc_map or p_map).point                        # marker at the best estimate
            for kind in (Marker.SPHERE, Marker.TEXT_VIEW_FACING):
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = rgb_msg.header.stamp
                m.ns = 'detections'
                m.id = len(markers.markers)
                m.type = kind
                m.action = Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = q.x, q.y, q.z
                m.pose.orientation.w = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.1, 1.0
                if kind == Marker.SPHERE:
                    m.scale.x = m.scale.y = m.scale.z = 0.3
                else:
                    m.pose.position.z = q.z + 0.5
                    m.scale.z = 0.35
                    m.color.r = m.color.g = m.color.b = 1.0
                    m.text = f'{text or name} {conf:.2f}'
                markers.markers.append(m)

        self.marker_pub.publish(markers)
        self.det_pub.publish(dets)
        self.det_c_pub.publish(dets_c)


def main():
    rclpy.init()
    rclpy.spin(DetectionNode())
    rclpy.shutdown()
