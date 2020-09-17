"""
Copyright (c) 2020 Jonas K. Falkner
"""


#
import os
import sys
import pickle
import argparse
from collections import namedtuple
import numpy as np


dg = sys.modules[__name__]

CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }

TW_CAPACITIES = {
        10: 250.,
        20: 500.,
        50: 750.,
        100: 1000.
    }


CVRP_SET = namedtuple("CVRP_SET",
                      ["depot_loc",  # Depot location
                       "node_loc",   # Node locations
                       "demand",     # demand per node
                       "capacity"])  # vehicle capacity (homogeneous)


CVRPTW_SET = namedtuple("CVRPTW_SET",
                        ["depot_loc",    # Depot location
                         "node_loc",     # Node locations
                         "demand",       # demand per node
                         "capacity",     # vehicle capacity (homogeneous)
                         "depot_tw",     # depot time window (full horizon)
                         "node_tw",      # node time windows
                         "durations",    # service duration per node
                         "service_window",  # maximum of time units
                         "time_factor"])    # value to map from distances in [0, 1] to time units (transit times)


def generate_cvrp_data(size, graph_size, rnds=None, **kwargs):
    """Generate data for CVRP
    replicates the data from https://github.com/wouterkool/attention-learn-to-route

    Args:
        size (int): size of dataset
        graph_size (int): size of problem instance graph (number of customers without depot)
        rnds : numpy random state

    Returns:
        List of CVRP instances wrapped in named tuples
    """
    rnds = np.random if rnds is None else rnds
    return [CVRP_SET(*data) for data in zip(
        rnds.uniform(size=(size, 2)).tolist(),
        rnds.uniform(size=(size, graph_size, 2)).tolist(),
        rnds.randint(1, 10, size=(size, graph_size)).tolist(),
        np.full(size, CAPACITIES[graph_size]).tolist()
    )]


def generate_cvrptw_data(size, graph_size, rnds=None,
                         service_window=1000,
                         service_duration=10,
                         time_factor=100.0,
                         tw_expansion=3.0,
                         **kwargs):
    """Generate data for CVRP-TW

    Args:
        size (int): size of dataset
        graph_size (int): size of problem instance graph (number of customers without depot)
        rnds : numpy random state
        service_window (int): maximum of time units
        service_duration (int): duration of service
        time_factor (float): value to map from distances in [0, 1] to time units (transit times)
        tw_expansion (float): expansion factor of TW w.r.t. service duration

    Returns:
        List of CVRP-TW instances wrapped in named tuples
    """
    rnds = np.random if rnds is None else rnds

    # sample locations
    dloc = rnds.uniform(size=(size, 2))  # depot location
    nloc = rnds.uniform(size=(size, graph_size, 2))  # node locations

    # TW start needs to be feasibly reachable directly from depot
    min_t = np.ceil(np.linalg.norm(dloc[:, None, :]*time_factor - nloc*time_factor, axis=-1)) + 1
    # TW end needs to be early enough to perform service and return to depot until end of service window
    max_t = np.ceil(np.linalg.norm(dloc[:, None, :]*time_factor - nloc*time_factor, axis=-1) + service_duration) + 1

    # horizon allows for the feasibility of reaching nodes / returning from nodes within the global tw (service window)
    horizon = list(zip(min_t, service_window - max_t))
    epsilon = np.maximum(np.abs(rnds.standard_normal([size, graph_size])), 1 / time_factor)

    # sample earliest start times a
    a = [rnds.randint(*h) for h in horizon]
    # calculate latest start times b, which is
    # a + service_time_expansion x normal random noise, all limited by the horizon
    # and combine it with a to create the time windows
    tw = [np.transpose(np.vstack((rt,  # a
                                  np.minimum(rt + tw_expansion * time_factor * sd, h[-1]).astype(int)  # b
                                  ))).tolist()
          for rt, sd, h in zip(a, epsilon, horizon)]

    return [CVRPTW_SET(*data) for data in zip(
        dloc.tolist(),
        nloc.tolist(),
        np.minimum(np.maximum(np.abs(rnds.normal(loc=15, scale=10, size=[size, graph_size])).astype(int), 1), 42).tolist(),
        np.full(size, TW_CAPACITIES[graph_size]).tolist(),
        [[0, service_window]] * size,
        tw,
        np.full([size, graph_size], service_duration).tolist(),
        [service_window] * size,
        [time_factor] * size,
    )]


def format_save_path(directory, args=None, note=''):
    """Formats the save path for saving datasets"""
    directory = os.path.normpath(os.path.expanduser(directory))

    fname = ''
    if args is not None:
        for k, v in args.items():
            if isinstance(v, str):
                fname += f'_{v}'
            else:
                fname += f'_{k}_{v}'

    fpath = os.path.join(directory, str(note) + fname + '.pkl')

    if os.path.isfile(fpath):
        print('Dataset file with same name exists already. Overwrite file? (y/n)')
        a = input()
        if a != 'y':
            print('Could not write to file. Terminating program...')
            sys.exit()

    return fpath


def save_dataset(dataset, filepath):
    """Saves data set to file path"""
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # check file extension
    assert os.path.splitext(filepath)[1] == '.pkl', "Can only save as pickle. Please add extension '.pkl'!"

    # save with pickle
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


# ## MAIN ## #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='./data', help="Create datasets in dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name of dataset (test, validation, ...)")
    parser.add_argument("--problem", type=str, default='cvrp',
                        help="Problem to sample: 'cvrp', 'cvrptw' or 'all' to generate all")
    parser.add_argument("--size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default: 20, 50, 100)")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--service_window', type=int, default=1000, help="Global time window of CVRP-TW")
    parser.add_argument('--service_duration', type=int, default=10, help="Global duration of service")
    parser.add_argument('--time_factor', type=float, default=100.0,
                        help="Value to map from distances in [0, 1] to time units (transit times)")
    parser.add_argument('--tw_expansion', type=float, default=3.0,
                        help="Expansion factor of service tw compared to service duration")

    args = parser.parse_args()

    problem = args.problem
    problems = ['cvrp', 'cvrptw'] if problem=='all' else [problem]

    for problem in problems:
        for graph_size in args.graph_sizes:

            ddir = os.path.join(args.dir, problem)
            filename = format_save_path(ddir, note=f"{problem}{graph_size}_{args.name}_seed{args.seed}")

            rnds = np.random.RandomState(args.seed)
            dataset = getattr(dg, f"generate_{problem}_data")(graph_size=graph_size, **vars(args))

            save_dataset(dataset, filename)
