#!/usr/bin/env python3
"""Depth image -> slim XYZ point cloud for the Nav2 costmaps (replaces depth_image_proc's 256k points).

Two kinds of points, chosen per pixel:
  * obstacle points: height above the floor in [mark_min_h, mark_max_h] and within mark_range
    (= the costmap's min/max_obstacle_height and obstacle_max_range in nav.launch.py). Taken at FULL
    pixel resolution, then reduced to one point per `voxel` cube (2.5 cm, half a costmap cell), so
    every cell the full cloud would mark is still marked (e.g. the underside of a café tabletop,
    seen at a grazing angle from the 0.60 m camera).
  * all other points (floor, far walls): only every `clear_stride`-th pixel in u and v. The costmaps
    use them just to clear (raytrace), so a sparse set of rays is enough.

  sub: /camera/depth_image (32FC1 metres or 16UC1 millimetres), /camera/camera_info, TF
  pub: /camera/points (sensor_msgs/PointCloud2, x y z float32, frame = depth image frame)

Height is computed with the (static) transform depth frame -> base_frame, looked up once.
Non-finite and out-of-range depths are dropped (as before: 'inf' beyond 12 m does not clear).
"""
from array import array

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import Buffer, TransformListener

FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float32)


class DepthToCloud(Node):
    def __init__(self):
        super().__init__('depth_to_cloud')
        p = lambda n, v: self.declare_parameter(n, v).value   # noqa: E731
        self.clear_stride = int(p('clear_stride', 4))
        self.voxel = float(p('voxel', 0.025))
        self.mark_min_h = float(p('mark_min_h', 0.10))
        self.mark_max_h = float(p('mark_max_h', 1.0))
        self.mark_range = float(p('mark_range', 3.0))
        self.min_d = float(p('min_depth', 0.2))
        self.max_d = float(p('max_depth', 12.0))
        self.base_frame = p('base_frame', 'base_footprint')
        self.max_rate = float(p('max_rate', 5.0))   # Hz (sim time); the local costmap updates at 5 Hz; 0 = no cap
        self.last_stamp = None
        self.rays = None         # (ray_x, ray_y), full resolution, from camera_info
        self.key = None
        self.height_row = None    # (r20, r21, r22, tz): height = r20*x + r21*y + r22*z + tz
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)   # dropped after the one lookup (/tf is ~100 Hz)
        self.pub = self.create_publisher(PointCloud2, '/camera/points', qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth_image', self.on_depth, qos_profile_sensor_data)
        self.n_in = self.n_out = self.n_msgs = 0
        self.create_timer(30.0, self.report)
        self.get_logger().info(f'obstacle points {self.mark_min_h}-{self.mark_max_h} m high, <= {self.mark_range} m, '
                               f'voxel {self.voxel} m; others every {self.clear_stride}th pixel; '
                               f'max {self.max_rate} Hz')

    def report(self):
        if self.n_msgs:
            self.get_logger().info(f'{self.n_msgs} clouds, {self.n_out / self.n_msgs:.0f} points each '
                                   f'(from {self.n_in / self.n_msgs:.0f} pixels within mark_range)')
            self.n_in = self.n_out = self.n_msgs = 0

    def on_info(self, msg):
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        key = (msg.width, msg.height, fx, fy, cx, cy)
        if key == self.key:
            return
        uu, vv = np.meshgrid(np.arange(msg.width, dtype=np.float32), np.arange(msg.height, dtype=np.float32))
        self.rays = ((uu - cx) / fx, (vv - cy) / fy)
        self.key = key

    def on_depth(self, msg):
        if self.rays is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.max_rate > 0 and self.last_stamp is not None and 0 <= stamp - self.last_stamp < 0.9 / self.max_rate:
            return                                   # too soon after the last processed frame
        if self.height_row is None:
            try:
                t = self.tf.lookup_transform(self.base_frame, msg.header.frame_id, Time())
            except Exception as e:   # noqa: BLE001  (TF not there yet)
                self.get_logger().warn(f'waiting for TF {self.base_frame} <- {msg.header.frame_id}: {e}',
                                       throttle_duration_sec=5.0)
                return
            r = quat_to_rot(t.transform.rotation)
            self.height_row = (*r[2], t.transform.translation.z)
            self.tf_listener.unregister()            # camera is fixed on the robot: no more TF needed
            self.get_logger().info(f'height = {r[2].round(3).tolist()} . p + {t.transform.translation.z:.3f}')
        if msg.encoding == '32FC1':
            d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width]
        elif msg.encoding == '16UC1':
            d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
            d = d.astype(np.float32) * 0.001
        else:
            self.get_logger().warn(f'unsupported depth encoding {msg.encoding}', throttle_duration_sec=10.0)
            return
        if d.shape != self.rays[0].shape:
            return
        self.last_stamp = stamp
        r20, r21, r22, tz = self.height_row
        rx, ry = self.rays
        # 1) obstacle candidates: depth <= mark_range (z <= distance), full resolution
        with np.errstate(invalid='ignore'):
            near = (d >= self.min_d) & (d <= self.mark_range)       # NaN/inf compare False
        z = d[near]
        x, y = rx[near] * z, ry[near] * z
        h = r20 * x + r21 * y + r22 * z + tz
        m = (h >= self.mark_min_h) & (h <= self.mark_max_h) & (x * x + y * y + z * z <= self.mark_range ** 2)
        mp = np.stack([x[m], y[m], z[m]], axis=1)
        if len(mp):                                                 # one point per voxel (1-D key: fast)
            k = np.floor(mp / self.voxel).astype(np.int64) + 2048   # |coord| < 51 m at 2.5 cm
            _, first = np.unique((k[:, 0] << 24) | (k[:, 1] << 12) | k[:, 2], return_index=True)
            mp = mp[first]
        # 2) clearing rays: sparse grid, every valid pixel that is not an obstacle point
        s = self.clear_stride
        ds = d[::s, ::s]
        with np.errstate(invalid='ignore'):
            ok = (ds >= self.min_d) & (ds <= self.max_d)
        zs = ds[ok]
        xs, ys = rx[::s, ::s][ok] * zs, ry[::s, ::s][ok] * zs
        hs = r20 * xs + r21 * ys + r22 * zs + tz
        c = ~((hs >= self.mark_min_h) & (hs <= self.mark_max_h) & (xs * xs + ys * ys + zs * zs <= self.mark_range ** 2))
        pts = np.concatenate([mp, np.stack([xs[c], ys[c], zs[c]], axis=1)]).astype(np.float32)
        self.n_in += int(near.sum()); self.n_out += len(pts); self.n_msgs += 1
        out = PointCloud2()
        out.header = msg.header
        out.height, out.width = 1, len(pts)
        out.fields = FIELDS
        out.is_bigendian = False
        out.point_step, out.row_step = 12, 12 * len(pts)
        out.is_dense = True
        out.data = array('B', pts.tobytes())     # array('B') avoids a per-byte check
        self.pub.publish(out)


def main():
    rclpy.init()
    n = DepthToCloud()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
