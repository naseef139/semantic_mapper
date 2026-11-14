"""Object map: merges repeated detections into single map objects."""
from collections import Counter
import json
import os
import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

MOVING = {'person'}          # detected, but never stored on the map
# long objects: the box centre shifts with the viewpoint, so allow a larger distance (m)
MERGE_DIST = {'sign': 2.5, 'bench': 1.8}

class ObjectMapNode(Node):
    def __init__(self):
        super().__init__('object_map_node')
        self.merge_dist = self.declare_parameter('merge_dist', 1.0).value        # same label: same object if closer than this (m)
        self.cross_dist = self.declare_parameter('cross_label_dist', 0.4).value  # different label: only if very close (m)
        self.min_sightings = self.declare_parameter('min_sightings', 3).value
        self.min_gap = self.declare_parameter('min_gap', 1.0).value              # seconds between counted sightings
        self.objects = []
        self.people_seen = 0
        self.next_id = 0
        self.people = []                 # places where people were seen: position and count
        self.person_radius = self.declare_parameter('person_radius', 0.8).value        
        self.save_path = os.path.expanduser(
            self.declare_parameter('save_path', '~/mallbot_ws/object_map.json').value)
        load_path = os.path.expanduser(self.declare_parameter('load_path', '').value)
        if load_path:
            self.load_map(load_path)
        self.create_timer(5.0, self.save_map)                
        self.create_subscription(Detection3DArray, '/semantic_mapper/detections', self.det_cb, 10)
        self.pub = self.create_publisher(MarkerArray, '/semantic_mapper/object_map', 10)
        self.create_timer(1.0, self.publish_markers)

    @staticmethod
    def label_of(obj):
        return obj['votes'].most_common(1)[0][0]         # class vote: the label with the most weight

    def find_match(self, label, pos):
        best, best_d = None, 1e9
        for obj in self.objects:
            d = float(np.linalg.norm(pos - obj['pos']))
            limit = MERGE_DIST.get(label, self.merge_dist) if label == self.label_of(obj) else self.cross_dist
            if d < limit and d < best_d:
                best, best_d = obj, d
        return best

    def det_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for det in msg.detections:
            hyp = det.results[0].hypothesis
            label, conf = hyp.class_id, hyp.score
            c = det.bbox.center.position
            pos = np.array([c.x, c.y, c.z])

            if label in MOVING:
                self.people_seen += 1
                for p in self.people:                       # remember where people stand
                    if np.linalg.norm(pos[:2] - p['pos'][:2]) < 1.0:
                        p['n'] += 1
                        p['pos'] += (pos - p['pos']) / p['n']
                        break
                else:
                    self.people.append(dict(pos=pos.copy(), n=1))
                continue

            if self.at_person(pos):                         # do not start an object where a person stands
                continue

            obj = self.find_match(label, pos)
            if obj is None:                                # first time we see it: a candidate
                self.next_id += 1
                self.objects.append(dict(id=self.next_id, pos=pos, updates=1, n=1,
                                         votes=Counter({label: conf}), last_t=t, confirmed=False))
                self.get_logger().info(f'candidate #{self.next_id}: {label} at ({pos[0]:.2f}, {pos[1]:.2f})')
                continue

            obj['updates'] += 1
            obj['pos'] += (pos - obj['pos']) / obj['updates']       # running average of the position
            obj['votes'][label] += conf                             # class vote, weighted by confidence
            if t - obj['last_t'] >= self.min_gap:                   # count only sightings spread over time
                obj['n'] += 1
                obj['last_t'] = t
            if not obj['confirmed'] and obj['n'] >= self.min_sightings:
                obj['confirmed'] = True
                p = obj['pos']
                self.get_logger().info(
                    f'CONFIRMED #{obj["id"]}: {self.label_of(obj)} at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) '
                    f'after {obj["n"]} sightings')
        self.fuse_close()
        self.drop_people_objects()
        
    def at_person(self, pos):
        """True if people were seen repeatedly at this spot (and pos is not high up, like a sign)."""
        if pos[2] > 1.5:
            return False
        return any(p['n'] >= 3 and np.linalg.norm(pos[:2] - p['pos'][:2]) < self.person_radius
                   for p in self.people)

    def drop_people_objects(self):
        """Remove objects that were stored before we knew a person stands there."""
        for obj in list(self.objects):
            if self.at_person(obj['pos']):
                self.get_logger().info(f'removed #{obj["id"]} {self.label_of(obj)}: a person stands there')
                self.objects.remove(obj)
                        
    def fuse_close(self):
        """Join stored objects of the same label that have ended up close together."""
        i = 0
        while i < len(self.objects):
            a = self.objects[i]
            j = i + 1
            while j < len(self.objects):
                b = self.objects[j]
                la = self.label_of(a)
                near = np.linalg.norm(a['pos'] - b['pos']) < MERGE_DIST.get(la, self.merge_dist)
                if la == self.label_of(b) and near:
                    w = a['updates'] + b['updates']
                    a['pos'] = (a['pos'] * a['updates'] + b['pos'] * b['updates']) / w
                    a['updates'] = w
                    a['n'] += b['n']
                    a['votes'] += b['votes']
                    a['confirmed'] = a['confirmed'] or b['confirmed'] or a['n'] >= self.min_sightings
                    self.get_logger().info(f'fused #{b["id"]} into #{a["id"]} ({la})')
                    del self.objects[j]
                else:
                    j += 1
            i += 1

    def save_map(self):
        data = {'frame': 'map', 'objects': [
            {'id': o['id'], 'label': self.label_of(o),
             'x': round(float(o['pos'][0]), 3), 'y': round(float(o['pos'][1]), 3), 'z': round(float(o['pos'][2]), 3),
             'sightings': o['n'], 'confirmed': o['confirmed'],
             'votes': {k: round(v, 2) for k, v in o['votes'].items()}}
            for o in self.objects]}
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_map(self, path):
        with open(path) as f:
            data = json.load(f)
        for d in data['objects']:
            self.objects.append(dict(id=d['id'], pos=np.array([d['x'], d['y'], d['z']]),
                                     updates=d['sightings'], n=d['sightings'],
                                     votes=Counter(d['votes']), last_t=0.0, confirmed=d['confirmed']))
            self.next_id = max(self.next_id, d['id'])
        self.get_logger().info(f'loaded {len(self.objects)} objects from {path}')
                    
    def publish_markers(self):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for obj in self.objects:
            p = obj['pos']
            for kind in (Marker.CUBE, Marker.TEXT_VIEW_FACING):
                if kind == Marker.TEXT_VIEW_FACING and not obj['confirmed']:
                    continue                                # candidates get no label
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = 'object_map'
                m.id = len(arr.markers)
                m.type = kind
                m.action = Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p[0]), float(p[1]), float(p[2])
                m.pose.orientation.w = 1.0
                if kind == Marker.CUBE:
                    size = 0.35 if obj['confirmed'] else 0.15
                    m.scale.x = m.scale.y = m.scale.z = size
                    if obj['confirmed']:
                        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.8, 0.2, 1.0    # green = confirmed
                    else:
                        m.color.r, m.color.g, m.color.b, m.color.a = 0.6, 0.6, 0.6, 0.8    # grey = candidate
                else:
                    m.pose.position.z = float(p[2]) + 0.5
                    m.scale.z = 0.35
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                    m.text = f'{self.label_of(obj)} x{obj["n"]}'
                arr.markers.append(m)
        self.pub.publish(arr)


def main():
    rclpy.init()
    rclpy.spin(ObjectMapNode())
    rclpy.shutdown()
