"""
Test Coordinate Boxes.
"""
import numpy as np
import unittest
from deepchem.utils import coordinate_box_utils as box_utils


class TestCoordinateBoxUtils(unittest.TestCase):

    def test_make_box(self):
        x_range = (-10, 10)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box = box_utils.CoordinateBox(x_range, y_range, z_range)
        assert box.x_range == x_range
        assert box.y_range == y_range
        assert box.z_range == z_range

    def test_union(self):
        x_range = (-10, 10)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box = box_utils.CoordinateBox(x_range, y_range, z_range)

        x_range = (-1, 1)
        y_range = (-2, 2)
        z_range = (-3, 3)
        interior_box = box_utils.CoordinateBox(x_range, y_range, z_range)

        merged_box = box_utils.union(box, interior_box)
        assert merged_box.x_range == box.x_range
        assert merged_box.y_range == box.y_range
        assert merged_box.z_range == box.z_range

        x_range = (-10, 10)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box1 = box_utils.CoordinateBox(x_range, y_range, z_range)

        x_range = (-11, 9)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box2 = box_utils.CoordinateBox(x_range, y_range, z_range)

        merged_box = box_utils.union(box1, box2)
        assert merged_box.x_range == (-11, 10)
        assert merged_box.y_range == (-20, 20)
        assert merged_box.z_range == (-30, 30)

    def test_get_face_boxes(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        boxes = box_utils.get_face_boxes(coords)
        # There are 4 faces to the shape created by coords
        assert len(boxes) == 4

    def test_point_containment(self):
        box = box_utils.CoordinateBox((0, 1), (0, 1), (0, 1))
        assert (0, 0, 0) in box
        assert (-0.1, -0.1, -0.1) not in box
        assert (0.5, 0.5, 0.5) in box
        assert (5, 5, 5) not in box

    def test_volume(self):
        box = box_utils.CoordinateBox((0, 1), (0, 1), (0, 1))
        assert box.volume() == 1.0
        box = box_utils.CoordinateBox((0, 0), (0, 1), (0, 1))
        assert box.volume() == 0

    def test_box_containment(self):
        box = box_utils.CoordinateBox((0, 1), (0, 1), (0, 1))
        int_box = box_utils.CoordinateBox((0, 1), (0, 1), (0, 1))
        assert box.contains(int_box)
        ext_box = box_utils.CoordinateBox((0, 2), (0, 1), (0, 1))
        assert not box.contains(ext_box)

    def test_box_hash(self):
        box1 = box_utils.CoordinateBox((0, 1), (0, 1), (0, 1))
        box2 = box_utils.CoordinateBox((0, 2), (0, 2), (0, 2))
        mapping = {}
        mapping[box1] = 1
        mapping[box2] = 2
        assert len(mapping) == 2

    def test_intersect_interval(self):
        int1 = (0, 1)
        int2 = (0.5, 2)
        inter = box_utils.intersect_interval(int1, int2)
        assert inter == (0.5, 1)

        int1 = (0, 1)
        int2 = (1.5, 2)
        inter = box_utils.intersect_interval(int1, int2)
        assert inter == (0, 0)

        int1 = (1.5, 2)
        int2 = (0, 1)
        inter = box_utils.intersect_interval(int1, int2)
        assert inter == (0, 0)

    def test_intersection(self):
        x_range = (-10, 10)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box = box_utils.CoordinateBox(x_range, y_range, z_range)

        x_range = (-1, 1)
        y_range = (-2, 2)
        z_range = (-3, 3)
        interior_box = box_utils.CoordinateBox(x_range, y_range, z_range)

        int_box = box_utils.intersection(box, interior_box)
        assert int_box == interior_box

        x_range = (-10, 10)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box1 = box_utils.CoordinateBox(x_range, y_range, z_range)

        x_range = (-11, 9)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box2 = box_utils.CoordinateBox(x_range, y_range, z_range)

        int_box = box_utils.intersection(box1, box2)
        assert int_box.x_range == (-10, 9)
        assert int_box.y_range == (-20, 20)
        assert int_box.z_range == (-30, 30)

    def test_merge_overlapping_boxes(self):
        x_range = (-1, 1)
        y_range = (-2, 2)
        z_range = (-3, 3)
        interior_box = box_utils.CoordinateBox(x_range, y_range, z_range)

        x_range = (-10, 10)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box = box_utils.CoordinateBox(x_range, y_range, z_range)
        boxes = [interior_box, box]
        merged_boxes = box_utils.merge_overlapping_boxes(boxes)
        assert len(merged_boxes) == 1

        x_range = (-10, 10)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box1 = box_utils.CoordinateBox(x_range, y_range, z_range)

        x_range = (-11, 9)
        y_range = (-20, 20)
        z_range = (-30, 30)
        box2 = box_utils.CoordinateBox(x_range, y_range, z_range)
        boxes = [box1, box2]
        merged_boxes = box_utils.merge_overlapping_boxes(boxes)
        assert len(merged_boxes) == 1

        box1 = box_utils.CoordinateBox((0, 1), (0, 1), (0, 1))
        box2 = box_utils.CoordinateBox((2, 3), (2, 3), (2, 3))
        boxes = [box1, box2]
        merged_boxes = box_utils.merge_overlapping_boxes(boxes)
        assert len(merged_boxes) == 2

        box1 = box_utils.CoordinateBox((1, 2), (1, 2), (1, 2))
        box2 = box_utils.CoordinateBox((1, 3), (1, 3), (1, 3))
        boxes = [box1, box2]
        merged_boxes = box_utils.merge_overlapping_boxes(boxes)
        assert len(merged_boxes) == 1
        assert merged_boxes[0] == box_utils.CoordinateBox((1, 3), (1, 3),
                                                          (1, 3))

        box1 = box_utils.CoordinateBox((1, 3), (1, 3), (1, 3))
        box2 = box_utils.CoordinateBox((1, 2), (1, 2), (1, 2))
        boxes = [box1, box2]
        merged_boxes = box_utils.merge_overlapping_boxes(boxes)
        assert len(merged_boxes) == 1
        assert merged_boxes[0] == box_utils.CoordinateBox((1, 3), (1, 3),
                                                          (1, 3))

        box1 = box_utils.CoordinateBox((1, 3), (1, 3), (1, 3))
        box2 = box_utils.CoordinateBox((1, 2), (1, 2), (1, 2))
        box3 = box_utils.CoordinateBox((1, 2.5), (1, 2.5), (1, 2.5))
        boxes = [box1, box2, box3]
        merged_boxes = box_utils.merge_overlapping_boxes(boxes)
        assert len(merged_boxes) == 1
        assert merged_boxes[0] == box_utils.CoordinateBox((1, 3), (1, 3),
                                                          (1, 3))
