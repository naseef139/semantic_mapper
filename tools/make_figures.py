#!/usr/bin/env python3
"""Figures for the README (docs/): results over runs, object map of one run, camera coverage.

  python3 make_figures.py [run=run27] [compare=run26]     (source ROS first; reads ~/mallbot_ws/runs)

  docs/results.png          recall / precision / position error per run (centroid method)
  docs/map_<run>.png        SLAM map + robot path + mapped objects vs ground truth
  docs/coverage.png         where the camera looked (<= 5 m, 95 deg FOV, rays stop at walls), two runs
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rosbag2_py  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from semantic_mapper.evaluate_map import read_truth  # noqa: E402

WS = os.path.expanduser('~/mallbot_ws')
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
WORLD = os.path.join(WS, 'src/mallbot_sim/worlds/mall_v2.sdf')
COLORS = {'sign': '#d62728', 'bench': '#8c564b', 'potted plant': '#2ca02c', 'bin': '#7f7f7f',
          'table': '#1f77b4'}

# evaluate_map results, centroid method (see CLAUDE.md "Results so far")
RESULTS = [  # run, recall %, precision %, position error m, sign names /8, what changed
    ('run4', 85, 100, 0.21, 5, 'keyboard driving'),
    ('run10', 90, 100, 0.24, 6, 'autonomous, manual cancels'),
    ('run15', 50, 71, 0.29, 3, 'collision + drift'),
    ('run19', 90, 82, 0.16, 8, 'depth in costmaps'),
    ('run22', 100, 83, 0.18, 5, 'slim depth cloud'),
    ('run24', 65, 93, 0.20, 4, 'class-share rule (explorer quit early)'),
    ('run25', 90, 100, 0.19, 6, 'explorer on sim time'),
    ('run26', 80, 94, 0.16, 4, 'trap check'),
    ('run27', 90, 100, 0.20, 5, 'keep target when trapped'),
    ('run28', 90, 90, 0.29, 5, 'look_around on'),
]


def read_nav(run):
    """Last /map, and the robot path in the map frame (odom composed with map->odom)."""
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=os.path.join(WS, 'runs', f'{run}_nav'), storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(topics=['/odom', '/tf', '/map']))
    mo, path, grid = (0.0, 0.0, 0.0), [], None
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == '/map':
            grid = deserialize_message(data, OccupancyGrid)
        elif topic == '/tf':
            for tr in deserialize_message(data, TFMessage).transforms:
                if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                    q = tr.transform.rotation
                    mo = (tr.transform.translation.x, tr.transform.translation.y, 2 * math.atan2(q.z, q.w))
        else:
            p = deserialize_message(data, Odometry).pose.pose
            c, s = math.cos(mo[2]), math.sin(mo[2])
            path.append((mo[0] + c * p.position.x - s * p.position.y,
                         mo[1] + s * p.position.x + c * p.position.y,
                         2 * math.atan2(p.orientation.z, p.orientation.w) + mo[2]))
    return grid, np.array(path)


def grid_image(grid):
    g = np.array(grid.data, dtype=np.int16).reshape(grid.info.height, grid.info.width)
    img = np.full(g.shape, 0.75)            # unknown: grey
    img[(g >= 0) & (g < 25)] = 1.0          # free: white
    img[g >= 65] = 0.0                      # occupied: black
    res, ox, oy = grid.info.resolution, grid.info.origin.position.x, grid.info.origin.position.y
    return img, (ox, ox + g.shape[1] * res, oy, oy + g.shape[0] * res), g


def fig_results():
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = np.arange(len(RESULTS))
    ax[0].plot(x, [r[1] for r in RESULTS], 'o-', label='recall %')
    ax[0].plot(x, [r[2] for r in RESULTS], 's-', label='precision %')
    ax[0].set_ylim(40, 105)
    ax[0].legend(loc='lower right')
    ax[0].grid(alpha=0.3)
    ax[1].bar(x, [r[3] for r in RESULTS], color='#999999')
    ax[1].set_ylabel('position error (m)')
    ax[1].grid(alpha=0.3, axis='y')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f'{r[0]}\n{r[5]}' for r in RESULTS], rotation=35, ha='right', fontsize=8)
    fig.suptitle('Object map quality per run (evaluate_map, centroid method)')
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, 'results.png'), dpi=120)


def fig_map(run):
    grid, path = read_nav(run)
    img, ext, _ = grid_image(grid)
    truth = [t for t in read_truth(WORLD, -10.0, 0.0, 0.0) if t['label'] != 'person']
    objs = [o for o in json.load(open(os.path.join(WS, 'runs', run, 'map_centroid.json')))['objects']
            if o['confirmed']]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', extent=ext)
    ax.plot(path[:, 0], path[:, 1], '-', color='#ff7f0e', lw=1, alpha=0.8, label='robot path')
    for t in truth:
        ax.plot(t['x'], t['y'], 'o', mfc='none', mec=COLORS[t['label']], ms=14, mew=1.5)
    for o in objs:
        ax.plot(o['x'], o['y'], 'x', color=COLORS.get(o['label'], 'k'), ms=9, mew=2.5)
        if o.get('text'):
            ax.annotate(o['text'], (o['x'], o['y']), xytext=(4, 6), textcoords='offset points', fontsize=7)
    for lab, c in COLORS.items():
        ax.plot([], [], 's', color=c, label=lab)
    ax.plot([], [], 'o', mfc='none', mec='k', label='ground truth')
    ax.plot([], [], 'x', color='k', mew=2, label='mapped (confirmed)')
    ax.set_xlim(-3, 23)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m, map frame)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title(f'{run}: SLAM map, robot path and object map vs ground truth')
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, f'map_{run}.png'), dpi=120)


def camera_seen(grid, path, rng=5.0):
    _, _, g = grid_image(grid)
    res, ox, oy = grid.info.resolution, grid.info.origin.position.x, grid.info.origin.position.y
    seen = np.zeros(g.shape, bool)
    for x, y, th in path[::10]:
        cx, cy = x - 0.155 * math.cos(th), y - 0.155 * math.sin(th)     # camera 0.155 m behind base_link
        for a in np.radians(np.arange(-47, 48, 2)):
            for k in np.arange(0, rng, res):
                i = int((cy + k * math.sin(th + a) - oy) / res)
                j = int((cx + k * math.cos(th + a) - ox) / res)
                if not (0 <= i < g.shape[0] and 0 <= j < g.shape[1]) or g[i, j] >= 65:
                    break
                seen[i, j] = True
    return seen


def fig_coverage(runs):
    fig, axs = plt.subplots(len(runs), 1, figsize=(10, 4.2 * len(runs)))
    for ax, run in zip(np.atleast_1d(axs), runs):
        grid, path = read_nav(run)
        img, ext, g = grid_image(grid)
        seen = camera_seen(grid, path)
        rgb = np.dstack([img] * 3)
        free = (g >= 0) & (g < 25)
        rgb[seen & free] = (0.55, 0.8, 1.0)                 # seen by the camera: light blue
        ax.imshow(rgb, origin='lower', extent=ext)
        ax.plot(path[:, 0], path[:, 1], '-', color='#ff7f0e', lw=1)
        pct = 100 * (seen & free).sum() / free.sum()
        ax.set_title(f'{run}: camera saw {pct:.0f}% of the free map (blue), robot path (orange)')
        ax.set_xlim(-3, 23)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, 'coverage.png'), dpi=110)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'run27'
    other = sys.argv[2] if len(sys.argv) > 2 else 'run26'
    os.makedirs(DOCS, exist_ok=True)
    fig_results()
    fig_map(run)
    fig_coverage([other, run])
    print('written to', os.path.abspath(DOCS))


if __name__ == '__main__':
    main()
