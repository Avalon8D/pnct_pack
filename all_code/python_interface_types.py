from ctypes import *
from os import environ
from os.path import join

import numpy

c_double_p = POINTER(c_double)
NULL_double = c_double_p()

c_long_p = POINTER(c_long)
NULL_long = c_long_p()

c_size_t_p = POINTER(c_size_t)
NULL_size_t = c_size_t_p()

NULL_void = c_void_p()

c_cluster_lib_name = environ['CLUSTERING_LIB_SO']
c_cluster_lib = CDLL(c_cluster_lib_name)

c_cluster_lib.init_rand(None)
c_cluster_lib.malloc.restype = c_void_p


# general C-utility functions

class C_NUMPY_UTIL():

    def keep_if_C_behaved(numpy_array, c_dtype=c_double_p):

        if not numpy_array.flags["CARRAY"]:
            numpy_array = numpy.ascontiguousarray(numpy_array, dtype=numpy_array.dtype)

        return numpy_array.ctypes.data_as(c_dtype), numpy_array

    def copy_numpy_array_from_c_array(c_array, array_length, numpy_dtype=numpy.float64, shape=None):

        numpy_array = numpy.fromiter(cast(c_array, c_void_p), dtype=numpy_dtype, count=array_length)

        if shape is not None:
            return numpy_array.reshape(shape)

        return numpy_array

    def buffer_numpy_array_from_c_array(c_array, array_length, numpy_dtype=numpy.float64, shape=None):
        c_array_address = addressof(c_array.contents)
        c_array_fixed = (type(c_array.contents) * array_length).from_address(c_array_address)
        numpy_array = numpy.frombuffer(c_array_fixed, dtype=numpy_dtype, count=array_length)

        if shape is not None:
            return numpy_array.reshape(shape)

        return numpy_array

    def kill_memory_block(c_struct):

        if hasattr(c_struct, 'memory_block') and bool(c_struct.memory_block):
            c_cluster_lib.free(cast(c_struct.memory_block, c_void_p))


# data_types

class FLAG_DATA(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("len", c_long), ("point_set", c_double_p),
                ("flen", c_long), ("set", c_double_p)]

    def alloc(data_len, data_flen):
        flags = FLAG_DATA()
        c_cluster_lib.flag_data_alloc(byref(flags), data_len, data_flen)

        return flags

    def self_alloc(self, data_len, data_flen):
        c_cluster_lib.flag_data_alloc(byref(self), data_len, data_flen)

    def load_from_numpy_data(
            self, flags_matrix,
            point_flags_vector=None
    ):
        flat_len = self.len * self.flen
        flags_set = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(self.set, flat_len)
        numpy.copyto(flags_set, flags_matrix.reshape(flat_len))

        if point_flags_vector is not None:
            point_flags_set = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(self.point_set, self.len)
            numpy.copyto(point_flags_set, point_flags_vector.reshape(self.len))

    def alloc_from_numpy_data(flags_matrix, point_flags_vector=None):
        flags = FLAG_DATA.alloc(flags_matrix.shape[0], flags_matrix.shape[1])
        flags.load_from_numpy_data(flags_matrix, point_flags_vector)

        return flags

    # loads flags matrix only
    def alloc_from_path(flags_path, delimiter=','):
        flags_matrix = numpy.loadtxt(flags_path, delimiter=delimiter, dtype=float)

        return FLAG_DATA.alloc_from_numpy_data(flags_matrix, point_flags_vector=None), flags_matrix

    def numpy_data(self, copy=False):
        flat_len = self.len * self.flen
        flags_set = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.set, flat_len, shape=(self.len, self.flen)
        )
        point_flags_set = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.point_set, self.len
        )

        if copy:
            return flags_set.copy(), point_flags_set.copy()

        return flags_set, point_flags_set

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data

    def __str__(self):
        flags_str = ""
        flags_id = iter(range(self.len * self.flen))

        for flags_row in range(self.len):
            flags_str += "{:.2f}".format(self.set[next(flags_id)])

            for flags_col in range(1, self.flen):
                flags_str += ", {:.2f}".format(self.set[next(flags_id)])

            flags_str += "\n"

        point_flags_str = "{:.2f}".format(self.point_set[0])

        for point_flag_id in range(1, self.len):
            point_flags_str += ", {:.2f}".format(self.point_set[point_flag_id])

        point_flags_str += "\n"

        return point_flags_str + '\n' + flags_str + '\n'


class DATA_WORKSPC(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("len", c_long), ("flen", c_long), ("set", c_double_p),
                ("row_stride", c_long), ("col_stride", c_long)]

    def alloc(data_len, data_flen):
        data = DATA_WORKSPC()
        c_cluster_lib.data_workspc_alloc(byref(data), data_len, data_flen)

        return data

    def self_alloc(self, data_len, data_flen):
        c_cluster_lib.data_workspc_alloc(byref(self), data_len, data_flen)

    def load_from_numpy_data(self, data_matrix):
        flat_len = self.len * self.row_stride
        data_set = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.set, flat_len, shape=(self.len, self.row_stride)
        )
        data_set = data_set[:, :self.flen:self.col_stride]
        numpy.copyto(data_set, data_matrix[:self.len, :self.flen])

    def alloc_from_numpy_data(data_matrix):
        data = DATA_WORKSPC.alloc(data_matrix.shape[0], data_matrix.shape[1])
        data.load_from_numpy_data(data_matrix)

        return data

    def alloc_from_path(data_path, delimiter=','):
        data_matrix = numpy.loadtxt(data_path, delimiter=delimiter, dtype=float)

        return DATA_WORKSPC.alloc_from_numpy_data(data_matrix), data_matrix

    def create_as_mask_of_matrix(data_matrix):
        data = DATA_WORKSPC()
        data_len, data_flen = data_matrix.shape

        c_data_matrix = data_matrix.ctypes.data_as(c_double_p)
        c_cluster_lib.data_workspc_mem_load(
            byref(data), c_data_matrix, data_len, data_flen,
            data_matrix.strides[0] // sizeof(c_double),
            data_matrix.strides[1] // sizeof(c_double)
        )

        return data

    def numpy_data(self, copy=False):
        flat_len = self.len * self.row_stride
        data_set = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.set, flat_len, shape=(self.len, self.row_stride)
        )
        data_set = data_set[:, :self.flen:self.col_stride]

        if copy:
            return data_set.copy()

        return data_set

    def normalize2_rows(self):
        c_cluster_lib.mat_normalize2_rows(
            self.set, self.len, self.flen, self.row_stride, self.col_stride, 0
        )

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data

    def __str__(self):
        return str(self.numpy_data())


DATA_WORKSPC_p = POINTER(DATA_WORKSPC)
NULL_DATA_WORKSPC = DATA_WORKSPC_p()


class DATA_FEATURE_RANGE(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("len", c_long), ("set", c_long_p)]

    def alloc(feature_len):
        features = DATA_FEATURE_RANGE()
        c_cluster_lib.data_feature_range_alloc(byref(features), feature_len)

        return features

    def self_alloc(self, feature_len):
        c_cluster_lib.data_feature_range_alloc(byref(self), feature_len)

    def load_from_numpy_data(self, features_vector):
        features_set = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.set, self.len, numpy_dtype=numpy.int64
        )
        numpy.copyto(features_set, features_vector.reshape(self.len))

    def alloc_from_numpy_data(features_vector):
        features = DATA_FEATURE_RANGE.alloc(features_vector.shape[0])
        features.load_from_numpy_data(features_vector)

        return features

    def create_as_mask_of_vector(features_vector):
        c_features_vector, features_vector = C_NUMPY_UTIL.keep_if_C_behaved(features_vector)
        feature_len = len(features_vector.flat)

        features = DATA_FEATURE_RANGE()
        c_cluster_lib.data_feature_range_mem_load(byref(features), c_features_vector, feature_len)

        return features, features_vector

    def alloc_as_mask_of_vector(feature_len):
        features_vector = numpy.empty(feature_len, dtype=numpy.int64)

        features = DATA_FEATURE_RANGE()
        c_cluster_lib.data_feature_range_mem_load(
            byref(features), features_vector.ctypes.data_as(c_long_p), feature_len
        )

        return features, features_vector

    def numpy_data(self, copy=False):
        features_set = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.set, self.len, numpy_dtype=numpy.int64
        )

        if copy:
            return features_set.copy()

        return features_set

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data

    def __str__(self):
        features_str = "{:d}".format(self.set[0])

        for feature_id in range(1, self.len):
            features_str += ", {:d}".format(self.set[feature_id])

        return features_str + '\n\n'


DATA_FEATURE_RANGE_p = POINTER(DATA_FEATURE_RANGE)
NULL_FEATURE_RANGE = DATA_FEATURE_RANGE_p()


class DATA_CLUSTERING(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("data_len", c_long), ("data_cluster_ids", c_long_p),
                ("cluster_count", c_long), ("cluster_sizes", c_long_p)]

    def alloc(data_len, cluster_count):
        clustering = DATA_CLUSTERING()
        c_cluster_lib.data_clustering_alloc(byref(clustering), data_len, cluster_count)

        return clustering

    def self_alloc(self, data_len, cluster_count):
        c_cluster_lib.data_feature_range_alloc(byref(self), data_len, cluster_count)

    def load_from_numpy_data(self, clustering_vector, eval_cluster_sizes=True):
        data_cluster_ids = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.data_cluster_ids, self.len, numpy_dtype=numpy.int64
        )
        numpy.copyto(data_cluster_ids, clustering_vector.reshape(self.len))

        if eval_cluster_sizes:

            for cluster_id in range(self.cluster_count):
                self.cluster_sizes[cluster_id] = (data_cluster_ids == i).sum()

    def alloc_from_numpy_data(clustering_vector, eval_cluster_sizes=True):
        clustering = DATA_CLUSTERING.alloc(
            clustering_vector.shape[0], int(clustering_vector.max() + 1)
        )
        clustering.load_from_numpy_data(clustering_vector, eval_cluster_sizes=True)

        return clustering

    def alloc_from_path(clustering_path, delimiter=','):
        clustering_vector = numpy.loadtxt(clustering_path, delimiter=delimiter, dtype=numpy.int64)

        return DATA_CLUSTERING.alloc_from_numpy_data(clustering_vector), clustering_vector

    def create_as_mask_of_vector(clustering_vector, cluster_sizes_vector):
        c_clustering_vector, clustering_vector = C_NUMPY_UTIL.keep_if_C_behaved(clustering_vector)
        c_cluster_sizes_vector, cluster_sizes_vector = C_NUMPY_UTIL.keep_if_C_behaved(cluster_sizes_vector)
        data_len, cluster_count = len(clustering_vector.flat), len(cluster_sizes_vector.flat)

        clustering = DATA_CLUSTERING()
        c_cluster_lib.data_clustering_mem_load(byref(clustering), c_clustering_vector, data_len,
                                               c_cluster_sizes_vector, cluster_count)

        return clustering, clustering_vector, cluster_sizes_vector

    def alloc_as_mask_of_vector(data_len, cluster_count):
        clustering_vector = numpy.empty(data_len, dtype=numpy.int64)
        cluster_sizes_vector = numpy.empty(cluster_count, dtype=numpy.int64)

        clustering = DATA_CLUSTERING()
        c_cluster_lib.data_clustering_mem_load(
            byref(clustering), clustering_vector.ctypes.data_as(c_long_p),
            data_len, cluster_sizes_vector.ctypes.data_as(c_long_p), cluster_count
        )

        return clustering, clustering_vector, cluster_sizes_vector

    def numpy_data(self, copy=False):
        data_cluster_ids = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.data_cluster_ids, self.data_len, numpy_dtype=numpy.int64
        )
        cluster_sizes = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.cluster_sizes, self.cluster_count, numpy_dtype=numpy.int64
        )

        if copy:
            return data_cluster_ids.copy(), cluster_sizes.copy()

        return data_cluster_ids, cluster_sizes

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data

    def __str__(self):
        data_cluster_ids_str = "{:d}".format(self.data_cluster_ids[0])

        for data_id in range(1, self.data_len):
            data_cluster_ids_str += ", {:d}".format(self.data_cluster_ids[data_id])

        data_cluster_ids_str += '\n'

        cluster_sizes_str = "{:d}".format(self.cluster_sizes[0])

        for cluster_id in range(1, self.cluster_count):
            cluster_sizes_str += ", {:d}".format(self.cluster_sizes[cluster_id])

        cluster_sizes_str += '\n'

        return data_cluster_ids_str + '\n' + cluster_sizes_str + '\n'


# rng_types

class PAIR_SAMPLE(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("count", c_long), ("sample", c_long_p)]

    def alloc(sample_len):
        pairs = PAIR_SAMPLE()
        PAIR_SAMPLE.pair_sample_alloc(byref(pairs), sample_len)

        return pairs

    def self_alloc(self, sample_len):
        PAIR_SAMPLE.pair_sample_alloc(byref(self), sample_len)

    def gen_sample(sample_len, id_len):
        pairs = PAIR_SAMPLE()
        c_cluster_lib.gen_pair_sample(byref(pairs), sample_len, id_len)

        return pairs

    def gen_prufer_sample(sample_len, id_len):
        pairs = PAIR_SAMPLE()
        c_cluster_lib.gen_prufer_pair_sample(byref(pairs), sample_len, id_len)

        return pairs

    def numpy_data(self, copy=False):
        pairs_sample = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.sample, 2 * self.count, numpy_dtype=numpy.int64, shape=(self.count, 2)
        )

        if copy:
            return pairs_sample.copy()

        return pairs_sample

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data

    def __str__(self):
        pairs_str = "({:d}, {:d})".format(self.sample[0], self.sample[1])

        for sample_id in range(1, self.count):
            pairs_str += ", ({:d}, {:d})".format(
                self.sample[2 * sample_id], self.sample[2 * sample_id + 1]
            )

        return pairs_str + '\n\n'


class SAMPLE_WEIGHTS_TABLES(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("sample_range", c_long), ("w_threshold_table", c_double_p),
                ("w_id_table", c_long_p)]

    def alloc(sample_range):
        sample_tables = SAMPLE_WEIGHTS_TABLES()
        c_cluster_lib.sample_weights_tables_alloc(byref(sample_tables), sample_range)

        return sample_tables

    def self_alloc(self, sample_range):
        c_cluster_lib.sample_weights_tables_alloc(byref(self), sample_range)

    def alloc_from_weights(weights_vector):
        sample_range = weights_vector.shape[0]
        sample_tables = SAMPLE_WEIGHTS_TABLES.alloc(sample_range)
        c_weights = sample_tables.w_threshold_table

        for i in range(sample_range):
            c_weights[i] = weights_vector[i]
            print(c_weights[i])

        c_cluster_lib.rng_weighted_integer_lookup_tables(sample_tables, c_weights)

        return sample_tables

    def set_weights(self, weights):
        c_weights = self.w_threshold_table

        for i in range(sample_range):
            c_weights[i] = weights_vector[i]
            print(c_weights[i])

        c_cluster_lib.rng_weighted_integer_lookup_tables(self, c_weights)

    def isample(self, offset=0):
        return c_cluster_lib.rng_weighted_integer(self, offset)

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy


# affinity_types

class LAPLACIAN_COMPONENTS(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("data_len", c_long), ("vectors", c_double_p),
                ("values", c_double_p)]

    def alloc(data_len):
        components = LAPLACIAN_COMPONENTS()
        c_cluster_lib.laplacian_components_alloc(byref(components), data_len)

        return components

    def self_alloc(self, data_len):
        c_cluster_lib.laplacian_components_alloc(byref(components), data_len)

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def __str__(self):
        data_len = self.data_len
        values_str = "{:.2f}".format(self.values[0])

        for value_id in range(1, data_len):
            values_str += ", {:.2f}".format(self.values[0])

        values_str += '\n'

        vectors_str = ""
        vectors_id = iter(range(data_len * data_len))

        for row_id in range(data_len):
            vectors_str += "{:.2f}".format(self.vectors[next(vectors_id)])

            for col_id in range(1, data_len):
                vectors_str += ", {:.2f}".format(self.vectors[next(vectors_id)])

            vectors_str += '\n'

        return values_str + '\n' + vectors_str + '\n'


# cluster_stats_types

class CLUSTERING_MARGINS(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("count", c_long), ("flen", c_long),
                ("set_up", c_double_p), ("set_down", c_double_p)]

    def alloc(cluster_count, data_flen):
        margins = CLUSTERING_MARGINS()
        c_cluster_lib.clustering_margins_alloc(byref(margins), cluster_count, data_flen)

        return margins

    def self_alloc(self, cluster_count, data_flen):
        c_cluster_lib.clustering_margins_alloc(byref(margins), cluster_count, data_flen)

    def load_from_numpy_data(self, margins_matrix):
        flat_len = self.count * self.data_flen
        margins_up = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(self.set_up, flat_len)
        numpy.copyto(margins_up, margins_matrix[0].reshape(flat_len))
        margins_down = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(self.set_down, flat_len)
        numpy.copyto(margins_down, margins_matrix[1].reshape(flat_len))

    def alloc_from_numpy_data(margins_matrix):
        margins = CLUSTERING_MARGINS()
        margins.load_from_numpy_data(margins_matrix)

        return margins

    def alloc_from_path(path, delimiter=','):
        margins_matrix = numpy.loadtxt(path, delimiter=',', dtype=floabyreft)

        return CLUSTERING_MARGINS.alloc_from_numpy_data(margins_matrix), margins_matrix

    def numpy_data(self, copy=False):
        flat_len = self.count * self.flen
        margins_up = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.set_up, flat_len, shape=(self.count, self.flen)
        )
        margins_down = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.set_down, flat_len, shape=(self.count, self.flen)
        )

        if copy:
            return numpy.array((margins_up, margins_down))

        return (margins_up, margins_down)

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data


class CLUSTERING_CENTROIDS(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("count", c_long), ("flen", c_long), ("set", c_double_p)]

    def alloc(cluster_count, data_flen):
        centroids = CLUSTERING_CENTROIDS()
        c_cluster_lib.clustering_centroids_alloc(byref(centroids), cluster_count, data_flen)

        return centroids

    def self_alloc(self, cluster_count, data_flen):
        c_cluster_lib.clustering_centroids_alloc(byref(self), cluster_count, data_flen)

    def load_from_numpy_data(self, centroids_matrix):
        flat_len = self.count * self.flen
        centroids_matrix = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(self.set, flat_len)
        numpy.copyto(centroids_matrix, centroids_matrix.reshape(flat_len))

    def alloc_from_numpy_data(centroids_matrix):
        centroids = CLUSTERING_CENTROIDS.alloc(centroids_matrix.shape[0], centroids_matrix.shape[1])
        centroids.load_from_numpy_data(centroids_matrix)

        return centroids

    def numpy_data(self, copy=False):
        flat_len = self.count * self.flen
        centroids_matrix = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.set, flat_len, shape=(self.count, self.flen)
        )

        if copy:
            return centroids_matrix.copy()

        return centroids_matrix

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data


class CLUSTERING_SAMPLE_SPACE(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("cluster_count", c_long), ("cluster_range_size", c_long),
                ("cluster_ranges", c_long_p), ("data_len", c_long),
                ("cluster_data_ids", c_long_p)]

    def alloc(data_len, cluster_count):
        sample_space = CLUSTERING_SAMPLE_SPACE()
        c_cluster_lib.sample_space_alloc(byref(sample_space), data_len, cluster_count)

        return sample_space

    def alloc_from_clustering(c_clustering):
        sample_space = CLUSTERING_SAMPLE_SPACE()
        c_cluster_lib.sample_space_alloc_from(byref(sample_space), c_clustering)

        return sample_space

    def load_form_clustering(self, c_clustering):
        c_cluster_lib.sample_space_set(sample_space, c_clustering)

    def set(self, c_clustering):
        c_cluster_lib.sample_space_set(self, c_clustering)

    def numpy_data(self, copy=False):
        data_len = self.data_len
        cluster_range_size = self.cluster_range_size

        cluster_ranges = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.cluster_ranges, cluster_range_size, numpy_dtype=numpy.int64
        )
        cluster_data_ids = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.cluster_data_ids, data_len
        )

        if copy:
            cluster_data_ids = cluster_data_ids.copy()

        return [cluster_data_ids[cluster_beg:cluster_end]
                for cluster_beg, cluster_end in zip(cluster_ranges[:-1], cluster_ranges[1:])]

    def kill_memory_block(self):
        c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bail_to_numpy_data(self):
        new_data = self.numpy_data()
        self.kill_memory_block()

        return new_data

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy


# proclus_types

class MEDOID_WORKSPC(Structure):
    _fields_ = [("count", c_long), ("sample_size", c_long),
                ("sample", c_long_p), ("off_currents", c_long_p),
                ("currents", c_long_p), ("bad_count", c_long),
                ("bad", c_long_p)]


class BUFFER_WORKSPC(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("data_flen", c_long), ("range", c_double_p),
                ("point", c_double_p), ("subspaces_scores", c_double_p),
                ("subspace_count", c_size_t), ("subspace_arg", c_size_t_p),
                ("least_cluster_size", c_double), ("cluster_sizes", c_double_p)]

    def alloc(data_flen, cluster_count, subspace_count):
        proclus_buffer = BUFFER_WORKSPC()
        c_cluster_lib.proclus_buffer_workspc_alloc(
            byref(proclus_buffer), data_flen, cluster_count, subspace_count
        )

        return proclus_buffer

    def self_alloc(self, data_flen, cluster_count, subspace_count):
        c_cluster_lib.proclus_buffer_workspc_alloc(byref(self), data_flen, cluster_count, subspace_count)

    def kill_memory_block(self):
        c_cluster_lib.free(cast(self.range, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy


class PROCLUS_WORKSPC(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("data_cluster_ids", c_long_p), ("medoid", MEDOID_WORKSPC),
                ("buffer", BUFFER_WORKSPC)]

    def alloc(data_len, data_flen, cluster_count, subspace_count,
              sample_size, least_cluster_size):
        proclus = PROCLUS_WORKSPC()
        c_cluster_lib.proclus_workspc_alloc(
            byref(proclus), data_len, data_flen, cluster_count,
            subspace_count, sample_size, least_cluster_size
        )

        return proclus

    def self_alloc(self, data_len, data_flen, cluster_count, subspace_count,
                   sample_size, least_cluster_size):
        c_cluster_lib.proclus_workspc_alloc(
            byref(self), data_len, data_flen, cluster_count,
            subspace_count, sample_size, least_cluster_size
        )

    def kill_memory_block(self):
        c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy


class MEDOID_SET(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("count", c_long), ("ids", c_long_p)]

    def alloc(medoid_count):
        medoid = MEDOID_SET()
        c_cluster_lib.medoid_set_alloc(byref(medoid), medoid_count)

        return medoid

    def self_alloc(self, medoid_count):
        c_cluster_lib.medoid_set_alloc(byref(self), medoid_count)

    def load_from_numpy_data(self, medoid_ids_vector):
        medoid_ids = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.ids, self.count, numpy_dtype=numpy.int64
        )
        numpy.copyto(medoid_ids, medoid_ids_vector.reshape(self.count))

    def alloc_from_numpy_data(medoid_ids_vector):
        medoid = DATA_FEATURE_RANGE.alloc(medoid_ids_vector.shape[0])
        medoid.load_from_numpy_data(medoid_ids_vector)

        return medoid

    def create_as_mask_of_vector(medoid_ids_vector):
        c_medoid_vector, medoid_ids_vector = C_NUMPY_UTIL.keep_if_C_behaved(medoid_ids_vector)
        medoid_count = count(medoid_ids_vector.flat)

        medoid = MEDOID_SET()
        c_cluster_lib.medoid_set_mem_load(byref(medoid), c_medoid_vector, medoid_count)

        return medoid, medoid_ids_vector

    def alloc_as_mask_of_vector(medoid_count):
        medoid_ids_vector = numpy.empty(medoid_count, dtype=numpy.int64)

        medoid = MEDOID_SET()
        c_cluster_lib.medoid_set_mem_load(byref(medoid), medoid_ids_vector.ctypes.data_as(c_long_p),
                                          medoid_count)

        return medoid, medoid_ids_vector

    def numpy_data(self, copy=False):
        medoid_ids = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.ids, self.count, numpy_dtype=numpy.int64
        )

        if copy:
            return medoid_ids.copy()

        return medoid_ids

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data


class PROCLUS_SUBSPACES(Structure):
    _fields_ = [("memory_len", c_long), ("memory_block", c_void_p),
                ("cluster_count", c_long), ("subspace_count", c_long),
                ("subspace_ranges", c_long_p), ("subspaces", c_long_p)]

    def alloc(subspace_count, cluster_count):
        subspcs = PROCLUS_SUBSPACES()
        c_cluster_lib.proclus_subspaces_alloc(byref(subspcs), subspace_count, cluster_count)

        return subspcs

    def self_alloc(self, subspace_count, cluster_count):
        c_cluster_lib.proclus_subspaces_alloc(byref(self), subspace_count, cluster_count)

    def load_from_numpy_data(self, subspace_ranges_vector, subspaces_vector):
        subspace_ranges = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.subspace_ranges, self.cluster_count + 1,
            numpy_dtype=numpy.int64
        )
        numpy.copyto(subspace_ranges, subspace_ranges_vector.reshape(self.cluster_count + 1))

        subspaces = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.subspaces, self.cluster_count * self.subspace_count,
            numpy_dtype=numpy.int64
        )
        numpy.copyto(subspaces, subspaces_vector.reshape(self.cluster_count * self.subspace_count))

    def alloc_from_numpy_data(subspace_ranges_vector, subspaces_vector):
        subspaces = PROCLUS_SUBSPACES.alloc(
            subspaces_vector.shape[0] // (subspace_ranges_vector.shape[0] - 1),
            (subspace_ranges_vector.shape[0] - 1)
        )
        subspaces.load_from_numpy_data(subspace_ranges_vector, subspaces_vector)

        return subspaces

    def create_as_mask_of_vector(subspace_ranges_vector, subspaces_vector):
        c_subspace_ranges_vector, subspace_ranges_vector = C_NUMPY_UTIL.keep_if_C_behaved(
            subspace_ranges_vector
        )
        c_subspaces_vector, subspaces_vector = C_NUMPY_UTIL.keep_if_C_behaved(subspaces_vector)
        cluster_count = (subspace_ranges_vector.shape[0] - 1)
        subspace_count = subspaces_vector.shape[0] // cluster_count

        subspcs = PROCLUS_SUBSPACES()
        c_cluster_lib.proclus_subspaces_mem_load(
            byref(subspcs), c_subspaces_vector, subspace_count,
            c_subspace_ranges_vector, cluster_count
        )

        return subspcs, subspace_ranges_vector, subspaces_vector

    def alloc_as_mask_of_vector(cluster_count, subspace_count):
        subspace_ranges_vector = numpy.empty(cluster_count, dtype=numpy.int64)
        subspaces_vector = numpy.empty(subspace_count)

        subspcs = PROCLUS_SUBSPACES()
        c_cluster_lib.proclus_subspaces_mem_load(
            byref(subspcs), subspaces_vector.ctypes.data_as(c_long_p),
            subspace_count, subspace_ranges_vector.ctypes.data_as(c_long_p),
            cluster_count
        )

        return subspcs, subspace_ranges_vector, subspaces_vector

    def numpy_data(self, copy=False):
        subspace_ranges_vector = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.subspace_ranges, self.cluster_count + 1,
            numpy_dtype=numpy.int64
        )
        subspaces_vector = C_NUMPY_UTIL.buffer_numpy_array_from_c_array(
            self.subspaces, self.cluster_count * self.subspace_count,
            numpy_dtype=numpy.int64
        )

        if copy:
            return subspace_ranges_vector.copy(), subspaces_vector.copy()

        return subspace_ranges_vector, subspaces_vector

    def kill_memory_block(self):

        if bool(self.memory_block):
            c_cluster_lib.free(cast(self.memory_block, c_void_p))

    def bitw_copy(self):
        self_copy = type(self)()
        pointer(self_copy)[0] = self

        return self_copy

    def bail_to_numpy_data(self):
        new_data = self.numpy_data(copy=True)
        self.kill_memory_block()

        return new_data


# wrapping function pointer for sampler functions used on proclus_bootstrap

PROCLUS_SAMPLER_F = CFUNCTYPE(c_long, c_double, c_void_p)
