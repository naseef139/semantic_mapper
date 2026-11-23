#!/usr/bin/env python3
"""Replay a run's recorded detections through object_map_node offline (seconds, no sim) and write
the box and centroid maps, to test map rules on old runs.

  python3 replay_map.py runs/runN/bag OUT_PREFIX [min_class_share=0.7]
  -> OUT_PREFIX_box.json, OUT_PREFIX_centroid.json (score them with evaluate_map)

Source ROS and the workspace first. Faithful: run22 replayed with min_class_share 0 (the rule was not
in the live run yet) gives the live map exactly (83% precision, 0.18 m); with 0.7 it gives 95%.
"""
import sys
import rclpy, rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.parameter import Parameter
from vision_msgs.msg import Detection3DArray
from semantic_mapper.object_map_node import ObjectMapNode
bag, out = sys.argv[1], sys.argv[2]
share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
rclpy.init()
nodes = {}
for topic, name in (('/semantic_mapper/detections', 'box'), ('/semantic_mapper/detections_centroid', 'centroid')):
    n = ObjectMapNode()
    n.set_parameters([Parameter('min_class_share', value=share)])
    n.min_share = share
    n.save_path = f'{out}_{name}.json'; nodes[topic] = n
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
while r.has_next():
    t, data, _ = r.read_next()
    if t in nodes:                       # read_next() gives (topic, data, t)
        nodes[t].det_cb(deserialize_message(data, Detection3DArray))
for n in nodes.values():
    n.save_map()
