"""Score a saved object map against the ground truth in a Gazebo world file.

  ros2 run semantic_mapper evaluate_map <object_map.json> <world.sdf> [start_x start_y start_yaw]
"""
import json
import math
import re
import sys

# model-name pattern in the world file -> label used on our map
TRUTH_NAMES = [(r'^sign_', 'sign'), (r'^bench_', 'bench'), (r'^plant_', 'potted plant'),
               (r'^trash_can_', 'bin'), (r'table', 'table'), (r'^person_', 'person')]
MATCH_DIST = 1.5          # a map object counts as found if within this distance (m) of the true one


def read_truth(sdf_path, sx, sy, syaw):
    text = open(sdf_path).read()
    found = re.findall(r'<model name="([^"]+)">.*?<pose>([^<]+)</pose>', text, flags=re.S)
    found += [(n, p) for n, p in re.findall(r'<name>([^<]+)</name>.*?<pose>([^<]+)</pose>', text, flags=re.S)]
    truth = []
    for name, pose in found:
        for pattern, label in TRUTH_NAMES:
            if re.search(pattern, name):
                wx, wy = [float(v) for v in pose.split()[:2]]
                dx, dy = wx - sx, wy - sy                      # world frame -> map frame
                mx = math.cos(syaw) * dx + math.sin(syaw) * dy
                my = -math.sin(syaw) * dx + math.cos(syaw) * dy
                truth.append(dict(name=name, label=label, x=mx, y=my, used=False))
                break
    return truth


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    sx, sy, syaw = [float(v) for v in sys.argv[3:6]] if len(sys.argv) >= 6 else (-10.0, 0.0, 0.0)
    truth = read_truth(sys.argv[2], sx, sy, syaw)
    objs = [o for o in json.load(open(sys.argv[1]))['objects'] if o['confirmed']]

    pairs = sorted((math.hypot(o['x'] - t['x'], o['y'] - t['y']), i, j)
                   for i, o in enumerate(objs) for j, t in enumerate(truth)
                   if o['label'] == t['label'])
    match = {}
    for d, i, j in pairs:                                      # nearest pairs first
        if d <= MATCH_DIST and i not in match and not truth[j]['used']:
            match[i] = (j, d)
            truth[j]['used'] = True

    labels = [l for _, l in TRUTH_NAMES if l != 'person']
    print(f'\n{"label":14s} {"truth":>5s} {"found":>5s} {"false":>5s} {"mean err (m)":>13s}')
    tot_t = tot_f = tot_false = 0
    errs_all = []
    for lab in labels:
        n_t = sum(t['label'] == lab for t in truth)
        errs = [d for i, (j, d) in match.items() if objs[i]['label'] == lab]
        n_false = sum(o['label'] == lab for i, o in enumerate(objs) if i not in match)
        mean = f'{sum(errs) / len(errs):.2f}' if errs else '-'
        print(f'{lab:14s} {n_t:5d} {len(errs):5d} {n_false:5d} {mean:>13s}')
        tot_t += n_t
        tot_f += len(errs)
        tot_false += n_false
        errs_all += errs
    n_map = len(objs)
    print(f'{"ALL":14s} {tot_t:5d} {tot_f:5d} {tot_false:5d} {sum(errs_all) / max(len(errs_all), 1):13.2f}')
    print(f'\nrecall    = {tot_f}/{tot_t} = {100 * tot_f / max(tot_t, 1):.0f}%   (real objects that are on the map)')
    print(f'precision = {tot_f}/{n_map} = {100 * tot_f / max(n_map, 1):.0f}%   (map objects that are real)')

    people = [t for t in truth if t['label'] == 'person']
    print('\nmissed:')
    for t in truth:
        if not t['used'] and t['label'] != 'person':
            print(f'  {t["name"]:20s} {t["label"]:13s} at map ({t["x"]:.2f}, {t["y"]:.2f})')
    print('false or mislabelled:')
    for i, o in enumerate(objs):
        if i not in match:
            near = min(truth, key=lambda t: math.hypot(o['x'] - t['x'], o['y'] - t['y']))
            d = math.hypot(o['x'] - near['x'], o['y'] - near['y'])
            print(f'  #{o["id"]} {o["label"]} at ({o["x"]:.2f}, {o["y"]:.2f})  -> nearest real thing: {near["name"]} ({d:.2f} m away)')
    stored_people = sum(o['label'] == 'person' for o in objs)
    print(f'people in the world: {len(people)}, stored on the map: {stored_people}')
    signs = [(objs[i], truth[j]) for i, (j, d) in match.items() if objs[i]['label'] == 'sign']
    right = sum(o.get('text', '').lower() == t['name'].replace('sign_', '') for o, t in signs)
    print(f'\nsign names read correctly: {right}/{sum(t["label"] == "sign" for t in truth)}')
    for o, t in sorted(signs, key=lambda s: s[1]['name']):
        print(f'  {t["name"]:18s} read as "{o.get("text", "")}"')    
    
    
