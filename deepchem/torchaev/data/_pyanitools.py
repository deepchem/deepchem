import h5py
import numpy as np
import os


class datapacker:
    def __init__(self, store_file, mode='w-', complib='gzip', complevel=6):
        """Wrapper to store arrays within HFD5 file"""
        self.store = h5py.File(store_file, mode=mode)
        self.clib = complib
        self.clev = complevel

    def store_data(self, store_loc, **kwargs):
        """Put arrays to store"""
        g = self.store.create_group(store_loc)
        for k, v, in kwargs.items():
            if isinstance(v, list):
                if len(v) != 0:
                    if isinstance(v[0], np.str_) or isinstance(v[0], str):
                        v = [a.encode('utf8') for a in v]

            g.create_dataset(k, data=v, compression=self.clib,
                             compression_opts=self.clev)

    def cleanup(self):
        """Wrapper to close HDF5 file"""
        self.store.close()


class anidataloader:

    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - ' + store_file)
        self.store = h5py.File(store_file, 'r')

    def h5py_dataset_iterator(self, g, prefix=''):
        """Group recursive iterator

        Iterate through all groups in all branches and return datasets in dicts)
        """
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
                data = {'path': path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k][()])

                        if isinstance(dataset, np.ndarray):
                            if dataset.size != 0:
                                if isinstance(dataset[0], np.bytes_):
                                    dataset = [a.decode('ascii')
                                               for a in dataset]
                        data.update({k: dataset})
                yield data
            else:  # test for group (go down)
                yield from self.h5py_dataset_iterator(item, path)

    def __iter__(self):
        """Default class iterator (iterate through all data)"""
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    def get_group_list(self):
        """Returns a list of all groups in the file"""
        return [g for g in self.store.values()]

    def iter_group(self, g):
        """Allows interation through the data in a given group"""
        for data in self.h5py_dataset_iterator(g):
            yield data

    def get_data(self, path, prefix=''):
        """Returns the requested dataset"""
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k][()])

                if isinstance(dataset, np.ndarray):
                    if dataset.size != 0:
                        if isinstance(dataset[0], np.bytes_):
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    def group_size(self):
        """Returns the number of groups"""
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    def cleanup(self):
        """Close the HDF5 file"""
        self.store.close()
