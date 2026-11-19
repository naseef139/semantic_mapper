#!/usr/bin/env python3
"""Why was the robot 'in lethal space'? Looks at the global costmap around the robot in a nav bag.

  python3 costmap_trap_check.py runs/runN_nav [half_m=1.5]

For the first time the robot centre is lethal/inscribed in /global_costmap/costmap (and a few
later snapshots), prints an ASCII patch around the robot:
  '#' occupied in the SLAM /map (walls, lidar)   'X' lethal in costmap but NOT in /map (camera / stale)
  'o' inscribed (cost 99)  '+' cost 50-98  '.' free/low   ' ' unknown   'R' robot centre
  't' café tabletop outline (ground truth)
"""
import math
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

TABLES = [(5.7, -3.9), (8.3, -3.9), (11.7, 3.9), (14.3, 3.9)]   # map frame
TOP_R = 0.913 / 2
# Nav2 footprint (nav.launch.py FOOTPRINT), base_link = base_footprint in x/y
FP_X, FP_Y = (-0.28, 0.10), (-0.17, 0.17)


def footprint_cells(cm, pose):
    """World points (2 cm grid) inside the footprint at pose."""
    pts = []
    c, s = math.cos(pose[2]), math.sin(pose[2])
    for fx in np.arange(FP_X[0], FP_X[1] + 1e-9, 0.02):
        for fy in np.arange(FP_Y[0], FP_Y[1] + 1e-9, 0.02):
            pts.append((pose[0] + c * fx - s * fy, pose[1] + s * fx + c * fy))
    return pts


def footprint_max(cm, pose):
    return max((cell_val(cm, x, y) or -1) for x, y in footprint_cells(cm, pose))


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    x, y, t = a
    return (x + math.cos(t) * b[0] - math.sin(t) * b[1], y + math.sin(t) * b[0] + math.cos(t) * b[1], t + b[2])


def grid(msg):
    g = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
    return g, msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y


def cell_val(gr, x, y):
    g, res, ox, oy = gr
    i, j = int((x - ox) / res), int((y - oy) / res)
    return int(g[j, i]) if 0 <= i < g.shape[1] and 0 <= j < g.shape[0] else None


def patch(cm, slam, pose, half):
    g, res, ox, oy = cm
    rows = []
    n_extra = 0
    fp = {(int((x - ox) / res), int((y - oy) / res)) for x, y in footprint_cells(cm, pose)}
    for y in np.arange(pose[1] + half, pose[1] - half, -res):          # north at the top
        line = ''
        for x in np.arange(pose[0] - half, pose[0] + half, res):
            c = cell_val(cm, x, y)
            s = cell_val(slam, x, y) if slam else None
            if abs(x - pose[0]) < res / 2 + 1e-9 and abs(y - pose[1]) < res / 2 + 1e-9:
                ch = 'R'
            elif s is not None and s >= 65:
                ch = '#'
            elif c == 100:
                ch = 'X'; n_extra += 1
            elif c == 99:
                ch = 'o'
            elif c is not None and c >= 50:
                ch = '+'
            elif c == -1 or c is None:
                ch = ' '
            else:
                ch = '.'
            if (int((x - ox) / res), int((y - oy) / res)) in fp and ch != 'R':
                ch = {'X': '@', '#': '%'}.get(ch, 'f')    # footprint; '@'/'%' = footprint ON lethal
            if ch in '.+ ' and any(abs(math.hypot(x - tx, y - ty) - TOP_R) < res * 0.6 for tx, ty in TABLES):
                ch = 't'
            line += ch
        rows.append(line)
    return rows, n_extra


def main():
    path = sys.argv[1]
    half = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if '/global_costmap/costmap' not in types:
        sys.exit('no /global_costmap/costmap in this bag')
    r.set_filter(rosbag2_py.StorageFilter(topics=['/tf', '/map', '/global_costmap/costmap']))

    m2o, o2b, slam = None, None, None
    t0, shown, first, lethal_times = None, 0, None, []
    while r.has_next():
        topic, data, _ = r.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == '/tf':
            for tf in msg.transforms:
                tr, q = tf.transform.translation, tf.transform.rotation
                p = (tr.x, tr.y, yaw(q))
                if tf.header.frame_id == 'map' and tf.child_frame_id == 'odom':
                    m2o = p
                elif tf.header.frame_id == 'odom' and tf.child_frame_id == 'base_footprint':
                    o2b = p
                    t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                    t0 = t if t0 is None else t0
        elif topic == '/map':
            slam = grid(msg)
        elif m2o and o2b:
            cm = grid(msg)
            # base_footprint is below base_link (front axle); footprint centre ~0.09 m behind it
            pose = compose(m2o, o2b)
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - (t0 or 0)
            v = footprint_max(cm, pose)
            if v == 100:
                lethal_times.append(st)
                if first is None or (shown < 4 and st - lethal_times[-2 if len(lethal_times) > 1 else -1] > 20):
                    first = first or st
                    rows, n_extra = patch(cm, slam, pose, half)
                    print(f'\n=== t={st:.0f} s  robot ({pose[0]:.2f},{pose[1]:.2f}) yaw {math.degrees(pose[2]):.0f}°  '
                          f'max cost under footprint {v}; lethal cells NOT in SLAM map within ±{half} m: {n_extra}')
                    print('\n'.join(rows))
                    shown += 1
    if not lethal_times:
        print('footprint never on a lethal cell in the global costmap')
    else:
        lt = np.array(lethal_times)
        print(f'\nfootprint on a lethal cell in {len(lt)} costmap updates, '
              f'from t={lt.min():.0f} s to t={lt.max():.0f} s')


if __name__ == '__main__':
    main()
