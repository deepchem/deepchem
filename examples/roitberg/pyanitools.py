# Written by Roman Zubatyuk and Justin S. Smith
# Modified by Yutong Zhao to make python2 compatible
import h5py
import numpy as np
import platform
import os

PY_VERSION = int(platform.python_version().split('.')[0]) > 3


class datapacker(object):

  def __init__(self, store_file, mode='w-', complib='gzip', complevel=6):
    """Wrapper to store arrays within HFD5 file
        """
    # opening file
    self.store = h5py.File(store_file, mode=mode)
    self.clib = complib
    self.clev = complevel

  def store_data(self, store_loc, **kwargs):
    """Put arrays to store
        """
    #print(store_loc)
    g = self.store.create_group(store_loc)
    for k, v, in kwargs.items():
      #print(type(v[0]))

      #print(k)
      if type(v) == list:
        if len(v) != 0:
          if type(v[0]) is np.str_ or type(v[0]) is str:
            v = [a.encode('utf8') for a in v]

      g.create_dataset(
          k, data=v, compression=self.clib, compression_opts=self.clev)

  def cleanup(self):
    """Wrapper to close HDF5 file
        """
    self.store.close()


class anidataloader(object):
  ''' Contructor '''

  def __init__(self, store_file):
    if not os.path.exists(store_file):
      exit('Error: file not found - ' + store_file)
    self.store = h5py.File(store_file)

  ''' Group recursive iterator (iterate through all groups in all branches and return datasets in dicts) '''

  def h5py_dataset_iterator(self, g, prefix=''):
    for key in g.keys():
      item = g[key]
      path = '{}/{}'.format(prefix, key)
      keys = [i for i in item.keys()]
      if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
        data = {'path': path}
        for k in keys:
          if not isinstance(item[k], h5py.Group):
            dataset = np.array(item[k].value)

            if type(dataset) is np.ndarray:
              if dataset.size != 0:
                if type(dataset[0]) is np.bytes_:
                  dataset = [a.decode('ascii') for a in dataset]

            data.update({k: dataset})

        yield data
      else:  # test for group (go down)
        for s in self.h5py_dataset_iterator(item, path):
          yield s

  ''' Default class iterator (iterate through all data) '''

  def __iter__(self):
    for data in self.h5py_dataset_iterator(self.store):
      yield data

  ''' Returns a list of all groups in the file '''

  def get_group_list(self):
    return [g for g in self.store.values()]

  ''' Allows interation through the data in a given group '''

  def iter_group(self, g):
    for data in self.h5py_dataset_iterator(g):
      yield data

  ''' Returns the requested dataset '''

  def get_data(self, path, prefix=''):
    item = self.store[path]
    path = '{}/{}'.format(prefix, path)
    keys = [i for i in item.keys()]
    data = {'path': path}
    # print(path)
    for k in keys:
      if not isinstance(item[k], h5py.Group):
        dataset = np.array(item[k].value)

        if type(dataset) is np.ndarray:
          if dataset.size != 0:
            if type(dataset[0]) is np.bytes_:
              dataset = [a.decode('ascii') for a in dataset]

        data.update({k: dataset})
    return data

  ''' Returns the number of groups '''

  def group_size(self):
    return len(self.get_group_list())

  def size(self):
    count = 0
    for g in self.store.values():
      count = count + len(g.items())
    return count

  ''' Close the HDF5 file '''

  def cleanup(self):
    self.store.close()


if __name__ == "__main__":
  base_dir = os.environ["ROITBERG_ANI"]

  # Number of conformations in each file increases exponentially.
  # Start with a smaller dataset before continuing. Use all of them
  # for production
  hdf5files = [
      'ani_gdb_s01.h5', 'ani_gdb_s02.h5', 'ani_gdb_s03.h5', 'ani_gdb_s04.h5',
      'ani_gdb_s05.h5', 'ani_gdb_s06.h5', 'ani_gdb_s07.h5', 'ani_gdb_s08.h5'
  ]

  hdf5files = [os.path.join(base_dir, f) for f in hdf5files]

  for hdf5file in hdf5files:
    print("processing", hdf5file)
    adl = anidataloader(hdf5file)
    for data in adl:

      # Extract the data
      P = data['path']
      R = data['coordinates']
      E = data['energies']
      S = data['species']
      smi = data['smiles']
