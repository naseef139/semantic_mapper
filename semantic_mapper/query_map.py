"""Ask the saved object map a question.

  ros2 run semantic_mapper query_map <label> [from_x from_y]
  e.g.  ros2 run semantic_mapper query_map bench 5.0 0.0
"""
import json
import math
import os
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    label = sys.argv[1]
    fx, fy = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) >= 4 else (0.0, 0.0)
    data = json.load(open(os.path.expanduser('~/mallbot_ws/object_map.json')))
    confirmed = [o for o in data['objects'] if o['confirmed']]
    objs = [o for o in confirmed if o['label'] == label]
    if not objs:
        print(f'No "{label}" on the map. Known labels: {sorted(set(o["label"] for o in confirmed))}')
        return
    objs.sort(key=lambda o: math.hypot(o['x'] - fx, o['y'] - fy))
    print(f'{len(objs)} x "{label}" on the map, nearest to ({fx:.1f}, {fy:.1f}) first:')
    for o in objs:
        d = math.hypot(o['x'] - fx, o['y'] - fy)
        print(f'  #{o["id"]:<3d} at ({o["x"]:6.2f}, {o["y"]:6.2f})  {d:5.1f} m away  seen {o["sightings"]} times')
