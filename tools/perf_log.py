#!/usr/bin/env python3
"""Log simulation speed and CPU per process every PERIOD seconds while a run is going.

  python3 perf_log.py OUT.csv RUN.log [duration_s=300] [period_s=10]

Columns: wall_s, sim_s, rtf (= d sim / d wall over the period, from ROS /clock),
CPU % per process group (100 = one core), total CPU % of the machine, explorer goals so far.
"""
import csv
import re
import sys
import time

import psutil
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock

GROUPS = {   # column -> substring of the process command line
    'gazebo': 'ign gazebo',
    'bridge': 'parameter_bridge',
    'slam': 'async_slam_toolbox_node',
    'controller': 'controller_server',
    'planner': 'planner_server',
    'bt_nav': 'bt_navigator',
    'explorer': 'frontier_explorer',
    'detection': 'detection_node',
    'object_maps': 'object_map_node',
    'cloud': ('point_cloud_xyz_node', 'depth_to_cloud'),
    'rviz': 'rviz2',
}


_node = None


def sim_time():
    """Sim time (s) from one ROS /clock message, or None.
    (Was `ign topic -e -t /stats -n 1`: a Ruby process at ~124% CPU for up to 5 s per sample.)"""
    global _node
    if _node is None:
        rclpy.init()
        _node = rclpy.create_node('perf_log')
    got = []
    sub = _node.create_subscription(Clock, '/clock', got.append, qos_profile_sensor_data)
    end = time.time() + 2.0
    while not got and time.time() < end:
        rclpy.spin_once(_node, timeout_sec=0.1)
    _node.destroy_subscription(sub)
    return got[0].clock.sec + got[0].clock.nanosec * 1e-9 if got else None


def thermal():
    """(CPU package temperature in C, average core clock in MHz) - to spot thermal throttling."""
    temp = ''
    for name, entries in psutil.sensors_temperatures().items():
        for e in entries:
            if e.label == 'Package id 0' or name == 'k10temp':
                temp = round(e.current, 1)
                break
    try:   # psutil.cpu_freq() gives wrong units on this laptop; /proc/cpuinfo is reliable
        f = [float(l.split(':')[1]) for l in open('/proc/cpuinfo') if l.startswith('cpu MHz')]
        mhz = round(sum(f) / len(f))
    except (OSError, ValueError, ZeroDivisionError):
        mhz = ''
    fans = {f.label: f.current for v in psutil.sensors_fans().values() for f in v}
    return temp, mhz, fans.get('cpu_fan', ''), fans.get('gpu_fan', '')


def procs():
    found = {k: [] for k in GROUPS}
    for p in psutil.process_iter(['cmdline']):
        cmd = ' '.join(p.info['cmdline'] or [])
        if 'bash -c' in cmd or 'perf_log' in cmd:
            continue
        for k, pat in GROUPS.items():
            if any(q in cmd for q in ((pat,) if isinstance(pat, str) else pat)):
                found[k].append(p)
    return found


def main():
    out, log = sys.argv[1], sys.argv[2]
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    period = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    t0 = time.time()
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wall_s', 'sim_s', 'rtf'] + list(GROUPS) + ['total_cpu', 'goals', 'cpu_temp_c', 'cpu_mhz', 'cpu_fan_rpm', 'gpu_fan_rpm'])
        psutil.cpu_percent(None)
        found = procs()
        for ps in found.values():
            for p in ps:
                p.cpu_percent(None)
        last_wall, last_sim = time.time(), sim_time()
        while time.time() - t0 < duration:
            time.sleep(period)
            wall, sim = time.time(), sim_time()
            rtf = (sim - last_sim) / (wall - last_wall) if sim is not None and last_sim is not None else ''
            row = []
            for k in GROUPS:
                s = 0.0
                for p in found[k]:
                    try:
                        s += p.cpu_percent(None)
                    except psutil.NoSuchProcess:
                        pass
                row.append(round(s, 1))
            try:
                goals = len(re.findall(r'goal #\d+:', open(log, errors='ignore').read()))
            except OSError:
                goals = ''
            w.writerow([round(wall - t0, 1), round(sim, 2) if sim else '',
                        round(rtf, 3) if rtf != '' else ''] + row
                       + [round(psutil.cpu_percent(None), 1), goals, *thermal()])
            f.flush()
            last_wall, last_sim = wall, sim
            # pick up processes that started late (nav at 15 s, our nodes at 5 s)
            new = procs()
            for k, ps in new.items():
                known = {p.pid for p in found[k]}
                for p in ps:
                    if p.pid not in known:
                        p.cpu_percent(None)
                        found[k].append(p)


if __name__ == '__main__':
    main()
