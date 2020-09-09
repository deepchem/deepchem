# The number of elements to print for dataset ids/tasks
_print_threshold = 10


def get_print_threshold() -> int:
  """Return the printing threshold for datasets.

  The print threshold is the number of elements from ids/tasks to
  print when printing representations of `Dataset` objects.

  Returns
  ----------
  threshold: int
    Number of elements that will be printed
  """
  return _print_threshold


def set_print_threshold(threshold: int):
  """Set print threshold

  The print threshold is the number of elements from ids/tasks to
  print when printing representations of `Dataset` objects.

  Parameters
  ----------
  threshold: int
    Number of elements to print.
  """
  global _print_threshold
  _print_threshold = threshold


# If a dataset contains more than this number of elements, it won't
# print any dataset ids
_max_print_size = 1000


def get_max_print_size() -> int:
  """Return the max print size for a dataset.

  If a dataset is large, printing `self.ids` as part of a string
  representation can be very slow. This field controls the maximum
  size for a dataset before ids are no longer printed.

  Returns
  -------
  max_print_size: int
    Maximum length of a dataset for ids to be printed in string
    representation.
  """
  return _max_print_size


def set_max_print_size(max_print_size: int):
  """Set max_print_size

  If a dataset is large, printing `self.ids` as part of a string
  representation can be very slow. This field controls the maximum
  size for a dataset before ids are no longer printed.

  Parameters
  ----------
  max_print_size: int
    Maximum length of a dataset for ids to be printed in string
    representation.
  """
  global _max_print_size
  _max_print_size = max_print_size
