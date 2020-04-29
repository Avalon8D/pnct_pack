import datetime
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, exists, join

from numba import jit
from numpy import *
from pandas import read_csv

from all_code import python_interface_funcs as cluster_lib


@jit(forceobj=True)
def create_class_clusters(
        class_matrix, data_cluster_ids,
        flags_matrix, cluster_count, data_flen
):
    valid_flags = flags_matrix < 1
    return [
        [
            class_matrix[logical_and(data_cluster_ids == i, valid_flags[:, j]), j::data_flen]
            if logical_and(data_cluster_ids == i, valid_flags[:, j]).sum() > 0
            else class_matrix[valid_flags[:, j], j::data_flen]
            for j in range(data_flen)
        ] for i in range(cluster_count)
    ]


@jit(forceobj=True)
def bootstrap_class_imputation(
        class_matrix, imputed_matrix, data_cluster_ids,
        flags_matrix, cluster_count, data_flen, sample_len,
        change_vector
):
    class_clusters = create_class_clusters(
        class_matrix, data_cluster_ids,
        flags_matrix, cluster_count, data_flen
    )
    invalid_flags = flags_matrix > 0

    imputed_class_matrix = class_matrix.copy()
    imputed_class_matrix[:, :data_flen] = imputed_matrix

    for class_label, class_point, point_flags, has_changed in zip(
        data_cluster_ids, imputed_class_matrix[:], invalid_flags[:], change_vector
    ):
        if not has_changed:
            continue

        for i, flag in enumerate(point_flags):
            if flag and class_clusters[class_label][i].shape[0] > 0:
                sample_bound = class_clusters[class_label][i].shape[0]
                sample_classes = class_clusters[class_label][i][random.choice(sample_bound, size=sample_len)]
                sample_classes = sample_classes[sample_classes[:, 0] > 0]

                if sample_classes.size:
                    sample_classes[:, 1:] /= sample_classes[:, 0].reshape(-1, 1)
                    class_point[i + data_flen::data_flen] = sample_classes[:, 1:].mean(axis=0)

                else:
                    class_point[i + data_flen::data_flen] = 0

                class_point[i + data_flen::data_flen] *= class_point[i]

    return imputed_class_matrix


@jit(forceobj=True)
def truncate_class_matrix(imputed_class_matrix, data_len, class_count, data_flen):
    imputed_class_matrix_int = imputed_class_matrix.astype(int, copy=True). \
        reshape(data_len, class_count, data_flen)
    bin_arg_maxes = empty(data_len, dtype=int)

    for i in range(data_flen):
        bin_view = imputed_class_matrix_int[..., i]
        bin_arg_maxes[...] = bin_view[:, 1:].argmax(axis=1)
        bin_arg_maxes[...] += 1

        for point_bin_view, bin_arg_max in zip(bin_view[:], bin_arg_maxes):
            point_bin_view[bin_arg_max] -= point_bin_view[1:].sum()
            point_bin_view[bin_arg_max] += point_bin_view[0]

        bin_view[..., 1:].sum(axis=1, out=bin_arg_maxes)
        bin_arg_maxes[...] -= bin_view[:, 0]
        absolute(bin_arg_maxes, out=bin_arg_maxes)

        assert bin_arg_maxes.max() == 0

    return imputed_class_matrix_int.reshape(data_len, class_count * data_flen)


parser = ArgumentParser(description="""\
    Batch script to cluster and detect outliers from directory containing equipments data.
""")

parser.add_argument('system_tag',
                    help="Type of classes used, for example: pnct, sgp, hdm4. Used only for filename lookup and writing.")
parser.add_argument('year', type=int, help="Year for which days are being class imputed")
parser.add_argument('eqs', help="Directory containing folders for each equipment data.")
parser.add_argument(
    '--list-eqs-done', default='',
    help="File on which to save and check list of equipments already ran. Defaults to <eqs>/.list_eqs_done_imput."
)
parser.add_argument(
    '--list-eqs-unfeasible', default='',
    help="File on which to save list of equipments that are unfeasible for class imputation. Defaults to <eqs>/.list_eqs_unfeasible."
)

args, _ = parser.parse_known_args()
print(args)

tables_in_path = args.eqs
assert exists(tables_in_path), f'{tables_in_path} directory does not exist'

year = args.year
assert year > 0 and year <= datetime.date.today().year, f'Year {year} not between 0 and {datetime.date.today().year}'

system_tag = args.system_tag

list_eqs_unfeasible_fn = (
    args.list_eqs_unfeasible if args.list_eqs_unfeasible
    else join(tables_in_path, '.list_eqs_unfeasible')
)

if not exists(list_eqs_unfeasible_fn):
    try:
        with open(list_eqs_unfeasible_fn, 'w') as f:
            pass
    except FileNotFoundError as e:
        raise Exception(f'Supplied {list_eqs_unfeasible_fn} file could no be opened.')

eqs = [eq for eq in listdir(tables_in_path) if eq.startswith('eq_')]

list_eqs_done_fn = (
    args.list_eqs_done if args.list_eqs_done
    else join(tables_in_path, '.list_eqs_done_imput')
)

if exists(list_eqs_done_fn):
    with open(list_eqs_done_fn, 'r') as list_eqs_done:
        done_eqs = list_eqs_done.read().split('\n')[:-1]

    eqs = setdiff1d(eqs, done_eqs)
    random.shuffle(eqs)
else:
    try:
        with open(list_eqs_done_fn, 'w') as f:
            pass
    except FileNotFoundError as e:
        raise Exception(f'Supplied {list_eqs_done_fn} file could no be opened.')


def run_eq(
        eq, eq_table, list_eqs_unfeasible,
        data_flen, relevant_features_count, features_vector,
        c_features_p, features_bins
):
    eq_tag = '_'.join(eq_table.split('_')[1:-1]) + '.csv'
    eq_table_tag = '_'.join(eq_table.split('_')[1:])

    print(eq, eq_tag)

    if (
            not isfile(join(tables_in_path, eq, 'imput_' + eq_table_tag))
            or not isfile(join(tables_in_path, eq, 'lost_days_d_' + eq_tag))
            or not isfile(join(tables_in_path, eq, 'outli_' + eq_tag))
    ):
        print('Missing imputed frequency matrix or lost_days inidicators for equipament ' + str(eq.split('_')[-1]))
        list_eqs_unfeasible.write(eq + ',' + eq_tag.split('_')[-1] + '\n')

        return -1

    df_heudson_matrix = read_csv(
        join(tables_in_path, eq, 'outl_' + eq_tag),
        index_col=0, parse_dates=True
    )
    df_heudson_matrix = df_heudson_matrix[df_heudson_matrix.index.year <= year]

    heudson_matrix = df_heudson_matrix.values.astype(float)

    df_class_matrix = read_csv(
        join(tables_in_path, eq, eq_table),
        index_col=0, parse_dates=True
    )
    df_class_matrix = df_class_matrix[df_class_matrix.index.year <= year]

    class_matrix = df_class_matrix.values.astype(float)

    data_len = class_matrix.shape[0]
    class_count = class_matrix.shape[1] // data_flen
    data_whole_flen = data_flen * class_count
    data_class_flen = data_whole_flen - data_flen

    data_matrix = class_matrix[:, :data_flen]

    this_years_days = df_heudson_matrix.index.year == year

    imputed_matrix_this_year = loadtxt(
        join(tables_in_path, eq, 'imput_' + eq_table_tag),
        delimiter=',', dtype=float
    )

    if int(imputed_matrix_this_year.shape[0]) != int(this_years_days.sum()):
        print('There are less imputed days than this years days')
        list_eqs_unfeasible.write(eq + ',' + eq_tag.split('_')[-1] + '\n')

        return 1

    imputed_matrix = data_matrix.copy()
    imputed_matrix[this_years_days] = imputed_matrix_this_year

    lost_days_this_year = loadtxt(
        join(tables_in_path, eq, 'lost_days_d_' + eq_tag),
        delimiter=',', dtype=bool
    )

    lost_days = zeros(data_len, dtype=bool)
    lost_days[this_years_days] = lost_days_this_year

    del this_years_days, lost_days_this_year, imputed_matrix_this_year, df_heudson_matrix

    flags_matrix = loadtxt(join(tables_in_path, eq, 'outli_' + eq_tag), delimiter=',', dtype=float)
    print(flags_matrix.shape, heudson_matrix.shape)

    # places changed after imputation
    change_matrix = imputed_matrix != data_matrix

    # places where imputation is unecessary
    # reassign flags as 1
    false_alarm_matrix = (
            change_matrix & (heudson_matrix < 1)
            & (flags_matrix > 0)
            & (
                    abs(
                        (imputed_matrix - data_matrix) / (data_matrix + 1e-10)
                    ) < .1
            )
    )

    imputed_matrix[false_alarm_matrix] = data_matrix[false_alarm_matrix]
    flags_matrix[false_alarm_matrix] = 0

    change_matrix[...] = imputed_matrix != data_matrix

    over_imputed = (flags_matrix < 1) & change_matrix
    imputed_matrix[over_imputed] = data_matrix[over_imputed]
    change_matrix[...] = imputed_matrix != data_matrix

    estimated_days = (change_matrix).sum(axis=1) > .8 * data_flen

    flags_vector = dot(flags_matrix, features_bins)
    valid_flags = flags_vector < .2 * features_vector.shape[0]
    valid_flags &= ~lost_days
    valid_flags &= ~estimated_days
    print(valid_flags.sum(), flags_matrix.shape[0])

    c_data = cluster_lib.DATA_WORKSPC.create_as_mask_of_matrix(imputed_matrix)
    c_data_subset = cluster_lib.DATA_WORKSPC.alloc_from_numpy_data(imputed_matrix[valid_flags])
    subset_original_ids = arange(data_len)[valid_flags]
    c_subset_original_ids = subset_original_ids.ctypes.data_as(cluster_lib.c_long_p)

    max_cluster_count = 16;

    if c_data_subset.len > max_cluster_count * 32:
        trial_max = 16;
        non_improv_max = 4;
        max_subspace_count = features_vector.shape[0];
        data_log = int((log(data_len) + 1))
        initial_sample_size = min(max_cluster_count * data_log * 4, max_cluster_count * 16, data_len)
        medoid_sample_size = initial_sample_size // 4
        inter_intra_sample_size, prufer = 1, 1
        cluster_avg_frac = .25
        bootstrap_max = data_log * 512
        weighted = initial_sample_size // 2

        c_cluster_sampler = cluster_lib.PROCLUS_SAMPLER_F(
            lambda a, b: 2 + random.randint(max_cluster_count - 2)
        )
        c_subspace_sampler = cluster_lib.PROCLUS_SAMPLER_F(
            lambda a, b: 2 + random.randint(max_subspace_count - 2)
        )

        affinity_matrix = cluster_lib.BASE_CLUSTERING_ALGOS.proclus_bootstrap(
            c_data, max_cluster_count, max_subspace_count,
            c_cluster_sampler, c_subspace_sampler,
            trial_max, non_improv_max, bootstrap_max,
            cluster_avg_frac,
            inter_intra_sample_size, medoid_sample_size,
            initial_sample_size,
            c_features_p=c_features_p, weighted=weighted,
            prufer=prufer,
            c_data_subset_p=cluster_lib.byref(c_data_subset),
            c_subset_original_ids=c_subset_original_ids
        )

        cluster_count = None
        c_clustering, affinity_vals = cluster_lib.AFFINITY_ALGOS.affinity_laplacian_eigen_kmeans(
            affinity_matrix, 2 * trial_max,
            cluster_count=cluster_count,
            inplace=True
        )

    else:
        max_cluster_count = 4
        trial_max = 128
        small_fraction = .05

        c_clustering, c_centroids = cluster_lib.BASE_CLUSTERING_ALGOS.kmeans_reassign_small(
            c_data, max_cluster_count,
            trial_max, small_fraction
        )

        c_centroids.kill_memory_block()

    cluster_count = int(c_clustering.cluster_count)
    data_cluster_ids, cluster_sizes = c_clustering.numpy_data()
    c_sample_space = cluster_lib.CLUSTERING_SAMPLE_SPACE.alloc_from_clustering(c_clustering)

    feature_bins_bool = features_bins.astype(bool, copy=False)
    relevant_features_count = int(feature_bins_bool.sum())
    irrelevant_features_count = int((~feature_bins_bool).sum())

    relevant_data = imputed_matrix[:, feature_bins_bool]
    c_relevant_data = cluster_lib.DATA_WORKSPC.alloc_from_numpy_data(relevant_data)

    irrelevant_data = imputed_matrix[:, ~feature_bins_bool]
    c_irrelevant_data = cluster_lib.DATA_WORKSPC.alloc_from_numpy_data(irrelevant_data)

    masses_set = [(.125, .375), (.25, .5), (.375, .625)]
    ranges_set = [(5.5, 2.5), (4.5, 2), (3, 1.5)]
    relevant_bound_up, relevant_bound_down = 2000, .9

    interinter_margins_set = []

    for (sub_mass, mass), (range_up, range_down) in zip(masses_set, ranges_set):
        chunked_margin = empty((2, cluster_count, data_flen))

        c_margins = cluster_lib.CLUSTERING_MARGINS.alloc(cluster_count, relevant_features_count)
        cluster_lib.CLUSTER_STATS.margins_initd_base_assym_bounded_interinterquantile(
            c_relevant_data, c_margins, c_sample_space, mass, mass,
            sub_mass, sub_mass, range_up, range_down, relevant_bound_up, relevant_bound_down
        )

        chunked_margin[..., feature_bins_bool] = c_margins.bail_to_numpy_data()

        c_margins = cluster_lib.CLUSTERING_MARGINS.alloc(cluster_count, irrelevant_features_count)
        cluster_lib.CLUSTER_STATS.margins_initd_base_assym_interinterquantile(
            c_irrelevant_data, c_margins, c_sample_space, mass, mass,
            sub_mass, sub_mass, range_up, range_down
        )

        chunked_margin[..., ~feature_bins_bool] = c_margins.bail_to_numpy_data()

        interinter_margins_set.append(chunked_margin)

    mean_margins = mean(interinter_margins_set, axis=0)
    after_flags_matrix = cluster_lib.CLUSTER_STATS.beyond_margins_flags(
        imputed_matrix, mean_margins,
        data_cluster_ids, nonzero=True
    )

    rem_flags_matrix = (~change_matrix) & (after_flags_matrix < 1)
    rem_lost_days = (dot(rem_flags_matrix, features_bins) > .2 * features_vector.shape[0]) | lost_days

    boot_sample_len = 15

    class_clusters = create_class_clusters(
        class_matrix, data_cluster_ids,
        flags_matrix, cluster_count, data_flen
    )

    # at least 80% available for good sampling
    if (
            (flags_matrix[:, feature_bins_bool] < 1).sum(axis=0) > 3 * boot_sample_len
    ).sum() > .8 * feature_bins_bool.sum():
        change_vector = any(change_matrix, axis=1)

        imputed_class_matrix = bootstrap_class_imputation(
            class_matrix, imputed_matrix, data_cluster_ids,
            flags_matrix, cluster_count, data_flen, boot_sample_len,
            change_vector
        )

        class_rates_matrix = imputed_class_matrix[:, 1:].copy()

        for i in range(data_flen):
            nonzero = imputed_class_matrix[:, i] > 0
            class_rates_matrix[nonzero, i::data_flen] /= imputed_class_matrix[nonzero, i].reshape(-1, 1)
            class_rates_matrix[logical_not(nonzero), i::data_flen] = 0

        imputed_class_matrix_int = truncate_class_matrix(
            imputed_class_matrix, data_len, class_count, data_flen
        )

        mean_margins = mean_margins.reshape((
            prod(mean_margins.shape[:-1]), mean_margins.shape[-1]
        ))

        imput_float_path = join(tables_in_path, eq, 'imput_boot_float_' + eq_table_tag)
        imput_path = join(tables_in_path, eq, 'imput_boot_' + eq_table_tag)
        flags_path = join(tables_in_path, eq, 'outli_f_' + eq_tag)
        rates_path = join(tables_in_path, eq, 'imput_rates_' + eq_tag)
        margins_path = join(tables_in_path, eq, 'margins_f_' + eq_tag)
        rem_flags_path = join(tables_in_path, eq, 'rem_flags_' + eq_tag)
        rem_lost_path = join(tables_in_path, eq, 'lost_days_' + eq_tag)
        estimated_path = join(tables_in_path, eq, 'estimated_days_' + eq_tag)
        labels_path = join(tables_in_path, eq, 'labels_f_' + eq_tag)

        for path, table, fmt in zip(
                [imput_path, imput_float_path, flags_path, rates_path,
                 margins_path, rem_flags_path, rem_lost_path,
                 estimated_path, labels_path],
                [imputed_class_matrix_int, imputed_class_matrix, flags_matrix,
                 class_rates_matrix, mean_margins, rem_flags_matrix, rem_lost_days,
                 estimated_days, data_cluster_ids],
                ['%ld', '%.2f', '%ld', '%.4f', '%.4f', '%ld', '%ld', '%ld', '%ld']
        ):
            savetxt(path, table, fmt=fmt, delimiter=',')

    else:
        print("There are not enough non-outlier bins for the imputation to be feasible.")
        print('\n', (flags_matrix < 1).sum(axis=0), '\n')
        print('\n', (heudson_matrix < 1).sum(axis=0), '\n')
        # i.o. stuff #

        mean_margins = mean_margins.reshape((prod(mean_margins.shape[:-1]), mean_margins.shape[-1]))

        flags_path = join(tables_in_path, eq, 'outli_f_' + eq_tag)
        margins_path = join(tables_in_path, eq, 'margins_f_' + eq_tag)
        rem_flags_path = join(tables_in_path, eq, 'rem_flags_' + eq_tag)
        rem_lost_path = join(tables_in_path, eq, 'lost_days_' + eq_tag)
        estimated_path = join(tables_in_path, eq, 'estimated_days_' + eq_tag)
        labels_path = join(tables_in_path, eq, 'labels_f_' + eq_tag)

        for path, table, fmt in zip(
                [flags_path, margins_path,
                 rem_flags_path, rem_lost_path,
                 estimated_path, labels_path],
                [flags_matrix, mean_margins, rem_flags_matrix,
                 rem_lost_days, estimated_days, data_cluster_ids],
                ['%ld', '%.4f', '%ld', '%ld', '%ld', '%ld']
        ):
            savetxt(path, table, fmt=fmt, delimiter=',')

        list_eqs_unfeasible.write(eq + ',' + eq_tag.split('_')[-1] + '\n')

    c_clustering.kill_memory_block()
    c_sample_space.kill_memory_block()
    c_relevant_data.kill_memory_block()
    c_irrelevant_data.kill_memory_block()
    c_data_subset.kill_memory_block()

    return 1


def run_eq_loop(eqs, list_eqs_done, list_eqs_unfeasible, system_tag):
    data_flen = 96
    relevant_features_count = data_flen - 24

    features_vector = array(range(relevant_features_count), dtype=int)
    features_vector += 24
    c_features_p = cluster_lib.DATA_FEATURE_RANGE_p(cluster_lib.DATA_FEATURE_RANGE. \
                                                    create_as_mask_of_vector(features_vector)[0])
    features_bins = zeros(data_flen)
    features_bins[features_vector] = 1

    for eq in eqs:
        print(eq)

        eq_tables = [
            eq for eq in listdir(join(tables_in_path, eq))
            if eq.startswith('freq') and eq.endswith(system_tag + '.csv')
        ]

        flag = 0

        for eq_table in eq_tables:
            flag = run_eq(
                eq, eq_table, list_eqs_unfeasible,
                data_flen, relevant_features_count, features_vector,
                c_features_p, features_bins
            )

            if flag == -1:
                break

        if flag == 1:
            list_eqs_done.write(eq + '\n')


with open(list_eqs_done_fn, 'a', buffering=1) as list_eqs_done:
    with open(list_eqs_unfeasible_fn, 'a', buffering=1) as list_eqs_unfeasible:
        run_eq_loop(eqs, list_eqs_done, list_eqs_unfeasible, system_tag)
