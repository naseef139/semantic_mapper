#!/usr/bin/env python3
"""P1 check on a nav bag: SLAM corrections (map->odom jumps), close scan ranges, café-table approach.

  python3 nav_bag_check.py runs/runN_nav [jump_m=0.05]

Map frame = world - (-10, 0) (start pose -10,0,0). Café tables (world): (+-1.7|+-4.3, +-3.9).
"""
import math
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

TABLES = {'s1_0 (Bakery)': (5.7, -3.9), 's1_1 (Bakery)': (8.3, -3.9),
          'n2_0': (11.7, 3.9), 'n2_1': (14.3, 3.9)}
TABLE_TOP_R = 0.913 / 2


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    """2D pose a ∘ b, poses = (x, y, th)."""
    x, y, t = a
    return (x + math.cos(t) * b[0] - math.sin(t) * b[1], y + math.sin(t) * b[0] + math.cos(t) * b[1], t + b[2])


def main():
    path = sys.argv[1]
    jump = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=['/tf', '/scan']))

    m2o, o2b, scans = [], [], []      # (t, x, y, th) and (t, min_range)
    while r.has_next():
        topic, data, t_ns = r.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == '/tf':
            for tf in msg.transforms:
                st = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                tr, q = tf.transform.translation, tf.transform.rotation
                row = (st, tr.x, tr.y, yaw(q))
                if tf.header.frame_id == 'map' and tf.child_frame_id == 'odom':
                    m2o.append(row)
                elif tf.header.frame_id == 'odom' and tf.child_frame_id == 'base_footprint':
                    o2b.append(row)
        else:
            rg = np.array(msg.ranges)
            rg = rg[np.isfinite(rg) & (rg >= msg.range_min)]
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            scans.append((st, float(rg.min()) if rg.size else math.inf))

    m2o, o2b, scans = np.array(m2o), np.array(o2b), np.array(scans)
    print(f'bag: {len(m2o)} map->odom, {len(o2b)} odom->base, {len(scans)} scans; '
          f'sim span {o2b[-1,0]-o2b[0,0]:.0f} s')

    # robot pose in map at each odom sample (latest map->odom before it)
    idx = np.clip(np.searchsorted(m2o[:, 0], o2b[:, 0]) - 1, 0, len(m2o) - 1)
    pose = np.array([compose(tuple(m2o[i, 1:]), tuple(b[1:])) for i, b in zip(idx, o2b)])
    dist = np.sum(np.hypot(np.diff(pose[:, 0]), np.diff(pose[:, 1])))
    print(f'path length {dist:.1f} m')

    # SLAM corrections: change of map->odom between consecutive updates, expressed as where
    # the robot "jumps" in the map (translation of robot pose, not of the odom origin)
    print(f'\nSLAM corrections > {jump*100:.0f} cm (robot pose jump in map):')
    n = 0
    for k in range(1, len(m2o)):
        if m2o[k, 0] - m2o[k - 1, 0] <= 0:
            continue
        j = np.searchsorted(o2b[:, 0], m2o[k, 0]) - 1
        if j < 0:
            continue
        b = tuple(o2b[j, 1:])
        p0, p1 = compose(tuple(m2o[k - 1, 1:]), b), compose(tuple(m2o[k, 1:]), b)
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if d > jump:
            n += 1
            s = np.searchsorted(scans[:, 0], m2o[k, 0])
            w = scans[max(0, s - 50):s + 1, 1]        # ~5 s of scans before the correction
            near = min((math.hypot(p1[0] - tx, p1[1] - ty), nm) for nm, (tx, ty) in TABLES.items())
            print(f'  t={m2o[k,0]-o2b[0,0]:7.1f} s  jump {d*100:5.1f} cm  dθ {math.degrees(m2o[k,3]-m2o[k-1,3]):+5.1f}°  '
                  f'at ({p1[0]:5.2f},{p1[1]:5.2f})  scan min (5 s before) {w.min() if w.size else float("nan"):.2f} m  '
                  f'nearest table {near[1]} {near[0]:.2f} m')
    print(f'  total: {n}')

    print('\nscan min range:')
    for th in (0.15, 0.20, 0.25):
        print(f'  < {th:.2f} m: {np.sum(scans[:, 1] < th)} scans ({100*np.mean(scans[:, 1] < th):.1f}%)')
    print(f'  overall min {scans[:, 1].min():.3f} m')

    print('\ncafé tables (robot base_footprint centre vs table centre; top radius 0.46 m):')
    for nm, (tx, ty) in TABLES.items():
        d = np.hypot(pose[:, 0] - tx, pose[:, 1] - ty)
        i = int(d.argmin())
        within = np.sum(d < 1.0)
        print(f'  {nm:15s} closest {d[i]:.2f} m at t={o2b[i,0]-o2b[0,0]:.0f} s; '
              f'{within} odom samples within 1.0 m')


if __name__ == '__main__':
    main()
