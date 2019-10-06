from numpy import *
from pandas import read_csv
from os import sys, listdir
from os.path import exists, join

from argparse import ArgumentParser

from all_code import python_interface_funcs as cluster_lib
import builtins

from subprocess import run

parser = ArgumentParser(description="""\
    Batch script to cluster and detect outliers from directory containing equipments data.
""")

parser.add_argument('system_tag', help="Type of classes used, for example: pnct, sgp, hdm4. Used only for filename lookup and writing.")
parser.add_argument('eqs', help="Directory containing folders for each equipment data.")
parser.add_argument(
    '--list-eqs-done', default='',
    help="File on which to save and check list of equipments already ran. Defaults to <eqs>/.list_eqs_done."
)

args, _ = parser.parse_known_args ()
print (args)

tables_in_path = args.eqs
assert exists (tables_in_path), f'{tables_in_path} directory does not exist'

system_tag = args.system_tag

list_eqs_done_fn = args.list_eqs_done if args.list_eqs_done else join (tables_in_path, '.list_eqs_done')

eqs = [eq for eq in listdir (tables_in_path) if eq.startswith ('eq_')]

if exists (list_eqs_done_fn):
    with open (list_eqs_done_fn, 'r') as list_eqs_done:
        done_eqs = list_eqs_done.read ().split ('\n')[:-1]
    
    eqs = setdiff1d (eqs, done_eqs)
    random.shuffle (eqs)
else:
    try:
        with open (list_eqs_done_fn, 'w') as f:
            pass
    except FileNotFoundError as e:
        raise Exception (f'Supplied {list_eqs_done_fn} file could no be opened.')

def run_eq (
    eq, eq_table,
    data_flen, relevant_features_count, features_vector,
    c_features_p, features_bins
):
    eq_tag = '_'.join (eq_table.split('_')[1:-1]) + '.csv'
    eq_table_tag = '_'.join (eq_table.split('_')[1:])

    class_matrix = read_csv (join (tables_in_path, eq, eq_table), index_col=0, header=None).values.astype (float)
    heudson_matrix = read_csv (join (tables_in_path, eq, 'outl_' + eq_tag), index_col=0).values.astype (float)

    data_len = class_matrix.shape[0]

    data_matrix = class_matrix[:, :data_flen]
    c_data = cluster_lib.DATA_WORKSPC.create_as_mask_of_matrix (data_matrix)

    margins_masses = [.625, .625, .375, .375, 3.5, 2, 1000, 0.9]
    c_margins = cluster_lib.CLUSTER_STATS.data_margins_assym_bounded_interinterquantile (c_data, *margins_masses)

    margins = c_margins.bail_to_numpy_data ()

    flags_matrix = cluster_lib.CLUSTER_STATS.beyond_margins_flags (data_matrix, margins, nonzero=True)
    # safety step. If all of any bin is outlier, invalidate classification
    flags_matrix[:, (flags_matrix < 1).sum (axis=0) == flags_matrix.shape[0]] = 1

    flags_vector = dot (flags_matrix, features_bins)

    valid_flags = flags_vector >= .6 * features_vector.shape[0]
    print (valid_flags.sum (), valid_flags.shape[0])

    c_data_subset = cluster_lib.DATA_WORKSPC.alloc_from_numpy_data (data_matrix[valid_flags])
    subset_original_ids = arange (data_len)[valid_flags]
    c_subset_original_ids = subset_original_ids.ctypes.data_as (cluster_lib.c_long_p)

    max_cluster_count = 16;

    if c_data_subset.len > max_cluster_count * 32:
        trial_max = 16;
        non_improv_max = 4;
        max_subspace_count = features_vector.shape[0];
        data_log = int ((log (data_len) + 1))
        initial_sample_size = min (max_cluster_count * data_log * 4, max_cluster_count * 16, data_len)
        medoid_sample_size = initial_sample_size // 4
        inter_intra_sample_size, prufer = 1, 1
        cluster_avg_frac = .25
        bootstrap_max = data_log * 512
        weighted = initial_sample_size // 2

        c_cluster_sampler = cluster_lib.PROCLUS_SAMPLER_F (
            lambda a,b: 2 + random.randint (max_cluster_count - 2)
        )
        c_subspace_sampler = cluster_lib.PROCLUS_SAMPLER_F (
            lambda a,b: 2 + random.randint (max_subspace_count - 2)
        )

        affinity_matrix = cluster_lib.BASE_CLUSTERING_ALGOS.proclus_bootstrap (
            c_data, max_cluster_count, max_subspace_count,
            c_cluster_sampler, c_subspace_sampler, 
            trial_max, non_improv_max, bootstrap_max,
            cluster_avg_frac,
            inter_intra_sample_size, medoid_sample_size, 
            initial_sample_size, 
            c_features_p=c_features_p, weighted=weighted, 
            prufer=prufer, 
            c_data_subset_p=cluster_lib.byref (c_data_subset), 
            c_subset_original_ids=c_subset_original_ids
        )

        cumm_frac=.9
        small_fraction = .05
        cluster_count = None
        c_clustering, affinity_vals = cluster_lib.AFFINITY_ALGOS.affinity_laplacian_eigen_kmeans (
            affinity_matrix, 
            2 * trial_max, 
            cluster_count=cluster_count, 
            inplace=True
        )

    else :
        max_cluster_count = 4
        trial_max = 128;
        small_fraction = .05

        c_clustering, c_centroids = cluster_lib.BASE_CLUSTERING_ALGOS.kmeans_reassign_small (
            c_data, max_cluster_count, 
            trial_max, small_fraction
        )

        c_centroids.kill_memory_block ()

    cluster_count = int (c_clustering.cluster_count)
    data_cluster_ids, cluster_sizes = c_clustering.numpy_data ()
    data_clusters = [data_matrix[data_cluster_ids == i] for i in range (cluster_count)]

    c_sample_space = cluster_lib.CLUSTERING_SAMPLE_SPACE.alloc_from_clustering (c_clustering)

    sets_len = 3
    masses_set = [(.125, .375), (.25, .5), (.375, .625)]
    ranges_set = [(5.5, 2.5), (4.5, 2), (3, 1.5)]
    relevant_bound_up, relevant_bound_down = 2000, .9

    masses_labels = ['Inner', 'Broader', 'Outer']

    interinter_margins_set = []
    data_flags_set = []

    for (sub_mass, mass), (range_up, range_down) in zip (masses_set, ranges_set):
        c_margins = cluster_lib.CLUSTERING_MARGINS.alloc (cluster_count, data_flen)
        cluster_lib.CLUSTER_STATS.margins_initd_base_assym_bounded_interinterquantile (
            c_data, c_margins, c_sample_space, mass, mass,
            sub_mass, sub_mass, range_up, range_down, 
            relevant_bound_up, relevant_bound_down
        )

        interinter_margins_set.append (c_margins.bail_to_numpy_data ())

    mean_margins = mean (interinter_margins_set[:-1], axis=0)

    data_flags = cluster_lib.CLUSTER_STATS.beyond_margins_flags (
        data_matrix, mean_margins, 
        data_cluster_ids, nonzero=True
    )

    # cummulative flags
    flags_matrix *= data_flags

    least_invalid = 8
    unaware_matrix = (heudson_matrix > 0) & (flags_matrix > 0) & (
        (heudson_matrix[:,features_vector] > 0).sum (axis=1) > least_invalid
    ).reshape (-1,1)# & feature_bins_bool.reshape (1, -1)
    
    flags_matrix[unaware_matrix] = 0

    # i.o. stuff #

    labeled_ids = arange (class_matrix.shape[0])[valid_flags]

    flags_matrix = flags_matrix < 1

    mean_margins = mean_margins.reshape ((prod (mean_margins.shape[:-1]), mean_margins.shape[-1]))

    flags_path = join (tables_in_path, eq, 'outli_' + eq_tag)
    labels_path = join (tables_in_path, eq, 'labels_' + eq_tag)
    labeled_path = join (tables_in_path, eq, 'labeled_ids_' + eq_tag)
    margins_path = join (tables_in_path, eq, 'margins_' + eq_tag)

    for path, table, fmt in zip ([flags_path, labels_path, labeled_path, margins_path], 
                                 [flags_matrix, data_cluster_ids, labeled_ids, mean_margins], 
                                 ['%ld', '%ld', '%ld', '%.2e']):
        savetxt (path, table, fmt=fmt, delimiter=',')

    c_clustering.kill_memory_block ()
    c_sample_space.kill_memory_block ()

def run_eq_loop (eqs, list_eqs_done, system_tag):
    data_flen = 96
    relevant_features_count = data_flen - 24

    features_vector = array (range(relevant_features_count), dtype=int)
    features_vector += 24
    c_features_p = cluster_lib.DATA_FEATURE_RANGE_p (
        cluster_lib.DATA_FEATURE_RANGE.create_as_mask_of_vector (features_vector)[0]
    )
    features_bins = zeros (data_flen)
    features_bins[features_vector] = 1
    
    for eq in eqs:
        print (eq)

        eq_tables = [
            eq for eq in listdir (join (tables_in_path, eq))
            if eq.startswith ('freq') and eq.endswith (system_tag + '.csv')
        ]

        print (eq_tables)

        for eq_table in eq_tables:
            run_eq (
                eq, eq_table,
                data_flen, relevant_features_count, features_vector,
                c_features_p, features_bins
            )

        list_eqs_done.write (eq + '\n')

with open (list_eqs_done_fn, 'a', buffering=1) as list_eqs_done:
    run_eq_loop (eqs, list_eqs_done, system_tag)
