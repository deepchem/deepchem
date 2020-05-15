"""This module adds utilities for coordinate boxes"""
import numpy as np
from scipy.spatial import ConvexHull


def intersect_interval(interval1, interval2):
  """Computes the intersection of two intervals.

  Parameters
  ----------
  interval1: tuple[int]
    Should be `(x1_min, x1_max)`
  interval2: tuple[int]
    Should be `(x2_min, x2_max)`

  Returns
  -------
  x_intersect: tuple[int]
    Should be the intersection. If the intersection is empty returns
    `(0, 0)` to represent the empty set. Otherwise is `(max(x1_min,
    x2_min), min(x1_max, x2_max))`.
  """
  x1_min, x1_max = interval1
  x2_min, x2_max = interval2
  if x1_max < x2_min:
    # If interval1 < interval2 entirely
    return (0, 0)
  elif x2_max < x1_min:
    # If interval2 < interval1 entirely
    return (0, 0)
  x_min = max(x1_min, x2_min)
  x_max = min(x1_max, x2_max)
  return (x_min, x_max)


def intersection(box1, box2):
  """Computes the intersection box of provided boxes.

  Parameters
  ----------
  box1: `CoordinateBox`
    First `CoordinateBox`
  box2: `CoordinateBox`
    Another `CoordinateBox` to intersect first one with.

  Returns
  -------
  A `CoordinateBox` containing the intersection. If the intersection is empty, returns the box with 0 bounds.
  """
  x_intersection = intersect_interval(box1.x_range, box2.x_range)
  y_intersection = intersect_interval(box1.y_range, box2.y_range)
  z_intersection = intersect_interval(box1.z_range, box2.z_range)
  return CoordinateBox(x_intersection, y_intersection, z_intersection)


def union(box1, box2):
  """Merges provided boxes to find the smallest union box. 

  This method merges the two provided boxes.

  Parameters
  ----------
  box1: `CoordinateBox`
    First box to merge in
  box2: `CoordinateBox`
    Second box to merge into this box

  Returns
  -------
  Smallest `CoordinateBox` that contains both `box1` and `box2`
  """
  x_min = min(box1.x_range[0], box2.x_range[0])
  y_min = min(box1.y_range[0], box2.y_range[0])
  z_min = min(box1.z_range[0], box2.z_range[0])
  x_max = max(box1.x_range[1], box2.x_range[1])
  y_max = max(box1.y_range[1], box2.y_range[1])
  z_max = max(box1.z_range[1], box2.z_range[1])
  return CoordinateBox((x_min, x_max), (y_min, y_max), (z_min, z_max))


def merge_overlapping_boxes(boxes, threshold=.8):
  """Merge boxes which have an overlap greater than threshold.

  Parameters
  ----------
  boxes: list[CoordinateBox]
    A list of `CoordinateBox` objects.
  threshold: float, optional (default 0.8)
    The volume fraction of the boxes that must overlap for them to be
    merged together. 
  
  Returns
  -------
  list[CoordinateBox] of merged boxes. This list will have length less
  than or equal to the length of `boxes`.
  """
  outputs = []
  for box in boxes:
    for other in boxes:
      if box == other:
        continue
      intersect_box = intersection(box, other)
      if (intersect_box.volume() >= threshold * box.volume() or
          intersect_box.volume() >= threshold * other.volume()):
        box = union(box, other)
    unique_box = True
    for output in outputs:
      if output.contains(box):
        unique_box = False
    if unique_box:
      outputs.append(box)
  return outputs


def get_face_boxes(coords, pad=5):
  """For each face of the convex hull, compute a coordinate box around it.

  The convex hull of a macromolecule will have a series of triangular
  faces. For each such triangular face, we construct a bounding box
  around this triangle. Think of this box as attempting to capture
  some binding interaction region whose exterior is controlled by the
  box. Note that this box will likely be a crude approximation, but
  the advantage of this technique is that it only uses simple geometry
  to provide some basic biological insight into the molecule at hand.

  The `pad` parameter is used to control the amount of padding around
  the face to be used for the coordinate box.

  Parameters
  ----------
  coords: np.ndarray
    Of shape `(N, 3)`. The coordinates of a molecule.
  pad: float, optional (default 5)
    The number of angstroms to pad.

  Examples
  --------
  >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
  >>> boxes = get_face_boxes(coords, pad=5)
  """
  hull = ConvexHull(coords)
  boxes = []
  # Each triangle in the simplices is a set of 3 atoms from
  # coordinates which forms the vertices of an exterior triangle on
  # the convex hull of the macromolecule.
  for triangle in hull.simplices:
    # Points is the set of atom coordinates that make up this
    # triangular face on the convex hull
    points = np.array(
        [coords[triangle[0]], coords[triangle[1]], coords[triangle[2]]])
    # Let's extract x/y/z coords for this face
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]

    # Let's compute min/max points
    x_min, x_max = np.amin(x_coords), np.amax(x_coords)
    x_min, x_max = int(np.floor(x_min)) - pad, int(np.ceil(x_max)) + pad
    x_bounds = (x_min, x_max)

    y_min, y_max = np.amin(points[:, 1]), np.amax(points[:, 1])
    y_min, y_max = int(np.floor(y_min)) - pad, int(np.ceil(y_max)) + pad
    y_bounds = (y_min, y_max)
    z_min, z_max = np.amin(points[:, 2]), np.amax(points[:, 2])
    z_min, z_max = int(np.floor(z_min)) - pad, int(np.ceil(z_max)) + pad
    z_bounds = (z_min, z_max)
    box = CoordinateBox(x_bounds, y_bounds, z_bounds)
    boxes.append(box)
  return boxes


class CoordinateBox(object):
  """A coordinate box that represents a block in space.

  Molecular complexes are typically represented with atoms as
  coordinate points. Each complex is naturally associated with a
  number of different box regions. For example, the bounding box is a
  box that contains all atoms in the molecular complex. A binding
  pocket box is a box that focuses in on a binding region of a protein
  to a ligand. A interface box is the region in which two proteins
  have a bulk interaction.

  The `CoordinateBox` class is designed to represent such regions of
  space. It consists of the coordinates of the box, and the collection
  of atoms that live in this box alongside their coordinates.
  """

  def __init__(self, x_range, y_range, z_range):
    """Initialize this box.

    Parameters
    ----------
    x_range: tuple
      A tuple of `(x_min, x_max)` with max and min x-coordinates.
    y_range: tuple
      A tuple of `(y_min, y_max)` with max and min y-coordinates.
    z_range: tuple
      A tuple of `(z_min, z_max)` with max and min z-coordinates.

    Raises
    ------
    `ValueError` if this interval is malformed
    """
    if not isinstance(x_range, tuple) or not len(x_range) == 2:
      raise ValueError("x_range must be a tuple of length 2")
    else:
      x_min, x_max = x_range
      if not x_min <= x_max:
        raise ValueError("x minimum must be <= x maximum")
    if not isinstance(y_range, tuple) or not len(y_range) == 2:
      raise ValueError("y_range must be a tuple of length 2")
    else:
      y_min, y_max = y_range
      if not y_min <= y_max:
        raise ValueError("y minimum must be <= y maximum")
    if not isinstance(z_range, tuple) or not len(z_range) == 2:
      raise ValueError("z_range must be a tuple of length 2")
    else:
      z_min, z_max = z_range
      if not z_min <= z_max:
        raise ValueError("z minimum must be <= z maximum")
    self.x_range = x_range
    self.y_range = y_range
    self.z_range = z_range

  def __repr__(self):
    """Create a string representation of this box"""
    x_str = str(self.x_range)
    y_str = str(self.y_range)
    z_str = str(self.z_range)
    return "Box[x_bounds=%s, y_bounds=%s, z_bounds=%s]" % (x_str, y_str, z_str)

  def __str__(self):
    """Create a string representation of this box."""
    return self.__repr__()

  def __contains__(self, point):
    """Check whether a point is in this box.

    Parameters
    ----------
    point: 3-tuple or list of length 3 or  np.ndarray of shape `(3,)`
      The `(x, y, z)` coordinates of a point in space.
    """
    (x_min, x_max) = self.x_range
    (y_min, y_max) = self.y_range
    (z_min, z_max) = self.z_range
    x_cont = (x_min <= point[0] and point[0] <= x_max)
    y_cont = (y_min <= point[1] and point[1] <= y_max)
    z_cont = (z_min <= point[2] and point[2] <= z_max)
    return x_cont and y_cont and z_cont

  def __eq__(self, other):
    """Compare two boxes to see if they're equal.

    Parameters
    ----------
    other: `CoordinateBox`
      Compare this coordinate box to the other one.

    Returns
    -------
    bool that's `True` if all bounds match.

    Raises
    ------
    `ValueError` if attempting to compare to something that isn't a
    `CoordinateBox`.
    """
    if not isinstance(other, CoordinateBox):
      raise ValueError("Can only compare to another box.")
    return (self.x_range == other.x_range and self.y_range == other.y_range and
            self.z_range == other.z_range)

  def __hash__(self):
    """Implement hashing function for this box.

    Uses the default `hash` on `self.x_range, self.y_range,
    self.z_range`.

    Returns
    -------
    Unique integeer
    """
    return hash((self.x_range, self.y_range, self.z_range))

  def center(self):
    """Computes the center of this box.

    Returns
    -------
    `(x, y, z)` the coordinates of the center of the box.

    Examples
    --------
    >>> box = CoordinateBox((0, 1), (0, 1), (0, 1))
    >>> box.center()
    (0.5, 0.5, 0.5)
    """
    x_min, x_max = self.x_range
    y_min, y_max = self.y_range
    z_min, z_max = self.z_range
    return (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2,
            z_min + (z_max - z_min) / 2)

  def volume(self):
    """Computes and returns the volume of this box.

    Returns
    -------
    float, the volume of this box. Can be 0 if box is empty

    Examples
    --------
    >>> box = CoordinateBox((0, 1), (0, 1), (0, 1))
    >>> box.volume()
    1
    """
    x_min, x_max = self.x_range
    y_min, y_max = self.y_range
    z_min, z_max = self.z_range
    return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

  def contains(self, other):
    """Test whether this box contains another.

    This method checks whether `other` is contained in this box.

    Parameters
    ----------
    other: `CoordinateBox`
      The box to check is contained in this box.

    Returns
    -------
    bool, `True` if `other` is contained in this box.

    Raises
    ------
    `ValueError` if `not isinstance(other, CoordinateBox)`.
    """
    if not isinstance(other, CoordinateBox):
      raise ValueError("other must be a CoordinateBox")
    other_x_min, other_x_max = other.x_range
    other_y_min, other_y_max = other.y_range
    other_z_min, other_z_max = other.z_range
    self_x_min, self_x_max = self.x_range
    self_y_min, self_y_max = self.y_range
    self_z_min, self_z_max = self.z_range
    return (self_x_min <= other_x_min and other_x_max <= self_x_max and
            self_y_min <= other_y_min and other_y_max <= self_y_max and
            self_z_min <= other_z_min and other_z_max <= self_z_max)
