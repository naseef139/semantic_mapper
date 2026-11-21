#!/usr/bin/env python3
"""P2 check on a nav bag: how close do the global plans (/plan) come to obstacles?

  python3 plan_clearance_check.py runs/runN_nav [runs/runN.log] [clear_m=0.35]

Clearance of a plan pose = distance from the pose to the nearest obstacle cell, where obstacles are
  walls  : SLAM /map cells >= 65
  costmap: /global_costmap/costmap cells == 100 (lidar + depth camera, e.g. café tabletops)
The first 0.5 m of each plan is skipped (the robot is already there; that part is not a planner choice).
A close pose (< clear_m) is 'avoidable' when a point within 0.5 m of it has clearance >= clear_m
(the space would have allowed a wider berth). Acceptance (P2): >= 95% of plans with no avoidable close pose.
With the run log, also prints the robot clearance at each stuck / escape / 'lethal space' event.
"""
import math
import re
import sys
from collections import Counter

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.ndimage import distance_transform_edt, maximum_filter

SKIP_START = 0.5
NEAR = 0.5


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    x, y, t = a
    return (x + math.cos(t) * b[0] - math.sin(t) * b[1], y + math.sin(t) * b[0] + math.cos(t) * b[1], t + b[2])


class Clear:
    """Distance-to-obstacle field of one grid + 'space allows' field."""

    def __init__(self, msg, obstacle, clear_m):
        g = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.res = msg.info.resolution
        self.ox, self.oy = msg.info.origin.position.x, msg.info.origin.position.y
        self.d = distance_transform_edt(~obstacle(g)) * self.res
        r = int(round(NEAR / self.res))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        self.wide = maximum_filter(self.d, footprint=(xx ** 2 + yy ** 2) <= r * r) >= clear_m

    def at(self, x, y):
        i, j = int((x - self.ox) / self.res), int((y - self.oy) / self.res)
        if 0 <= i < self.d.shape[1] and 0 <= j < self.d.shape[0]:
            return float(self.d[j, i]), bool(self.wide[j, i])
        return math.inf, False


def region(x, y):
    """Mall layout (map frame): corridor |y| < 2 (walls at +-2), doorways at the wall line, shops beyond."""
    ay = abs(y)
    side = 'N' if y > 0 else 'S'
    if ay < 1.6:
        return 'corridor'
    if ay < 2.4:
        return f'doorway/wall line {side}'
    return f'inside shop {side}'


def log_events(path):
    ev = []
    pat = re.compile(r'\[(\d{10}\.\d+)\]')
    for line in open(path, errors='replace'):
        kind = ('stuck' if 'stuck' in line else 'escape' if 'escaping' in line
                else 'lethal-start' if 'Starting point in lethal' in line else None)
        m = pat.search(line)
        if kind and m:
            t = float(m.group(1))
            if kind == 'lethal-start' and ev and ev[-1][1] == kind and t - ev[-1][2] < 10:
                ev[-1] = (ev[-1][0], kind, t, ev[-1][3] + 1)     # merge a burst
                continue
            ev.append((t, kind, t, 1))
    return ev


def main():
    path = sys.argv[1]
    log = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].replace('.', '').isdigit() else None
    clear_m = float(sys.argv[-1]) if sys.argv[-1].replace('.', '').isdigit() else 0.35

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=['/tf', '/map', '/global_costmap/costmap', '/plan']))

    walls = cmap = None
    m2o = o2b = None
    poses = []                         # (wall_t, x, y)
    plans = []                         # per plan: dict
    close_pts = Counter()
    while r.has_next():
        topic, data, t_ns = r.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        tw = t_ns * 1e-9
        if topic == '/tf':
            for tf in msg.transforms:
                tr, q = tf.transform.translation, tf.transform.rotation
                p = (tr.x, tr.y, yaw(q))
                if tf.header.frame_id == 'map' and tf.child_frame_id == 'odom':
                    m2o = p
                elif tf.header.frame_id == 'odom' and tf.child_frame_id == 'base_footprint':
                    o2b = p
                    if m2o:
                        x, y, _ = compose(m2o, o2b)
                        poses.append((tw, x, y))
        elif topic == '/map':
            walls = Clear(msg, lambda g: g >= 65, clear_m)
        elif topic == '/global_costmap/costmap':
            cmap = Clear(msg, lambda g: g == 100, clear_m)
        elif topic == '/plan' and walls and len(msg.poses) > 1:
            xy = np.array([(p.pose.position.x, p.pose.position.y) for p in msg.poses])
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
            res = []
            for (x, y), si in zip(xy, s):
                if si < SKIP_START:
                    continue
                dw, ww = walls.at(x, y)
                dc, wc = cmap.at(x, y) if cmap else (math.inf, True)
                d = min(dw, dc)
                res.append((d, ww and wc, x, y))
            if not res:
                continue
            dmin = min(q[0] for q in res)
            avoid = [q for q in res if q[0] < clear_m and q[1]]
            for q in avoid:
                close_pts[(round(q[2]), round(q[3]))] += 1
            worst = min(res)
            plans.append(dict(t=tw, dmin=dmin, avoid=bool(avoid), len=s[-1], worst=worst,
                              goal=(round(xy[-1, 0], 1), round(xy[-1, 1], 1))))

    if not plans:
        sys.exit('no /plan messages with a map')
    d = np.array([p['dmin'] for p in plans])
    av = np.array([p['avoid'] for p in plans])
    print(f'{len(plans)} plans ({len({p["goal"] for p in plans})} distinct goals), clearance threshold {clear_m} m, '
          f'first {SKIP_START} m of each plan skipped')
    print(f'  plans with min clearance >= {clear_m}: {np.mean(d >= clear_m)*100:5.1f}%')
    print(f'  plans with no AVOIDABLE close pose:  {np.mean(~av)*100:5.1f}%   (P2 accept: >= 95%)')
    print(f'  min clearance: overall {d.min():.2f} m, median of per-plan minima {np.median(d):.2f} m')
    for lo, hi in ((0, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 9)):
        print(f'    per-plan min in [{lo:.2f}, {hi:.2f}): {np.sum((d >= lo) & (d < hi)):4d}')

    print('\nwhere the avoidable close passes are (1 m cells, # plan poses):')
    by_region = Counter()
    for (x, y), n in close_pts.items():
        by_region[region(x, y)] += n
    for k, n in by_region.most_common():
        print(f'  {k:22s} {n}')
    for (x, y), n in close_pts.most_common(12):
        print(f'    ({x:3d},{y:3d}) {region(x, y):22s} {n}')

    if log:
        pt = np.array([p[0] for p in poses])
        print('\nexplorer / planner events (robot clearance to walls | costmap obstacles; last plan before):')
        t_first = pt[0]
        for t, kind, t_end, n in log_events(log):
            k = min(np.searchsorted(pt, t), len(pt) - 1)
            _, x, y = poses[k]
            last = [p for p in plans if p['t'] <= t]
            lp = last[-1] if last else None
            dw, _ = walls.at(x, y)
            dc, _ = cmap.at(x, y) if cmap else (math.inf, True)
            extra = f' ×{n}' if n > 1 else ''
            lps = (f'last plan min {lp["dmin"]:.2f} m at ({lp["worst"][2]:.1f},{lp["worst"][3]:.1f})'
                   if lp else '')
            print(f'  t={t - t_first:5.0f} s  {kind:12s}{extra:4s} robot ({x:5.2f},{y:5.2f}) {region(x, y):22s} '
                  f'clear {dw:.2f} | {dc:.2f} m   {lps}')
        print('  (note: final-map clearance; walls seen later may differ slightly)')


if __name__ == '__main__':
    main()
