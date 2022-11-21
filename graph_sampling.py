import os
import pandas as pd
import argparse
import numpy as np
import random
import networkx
import torch
from networkx.algorithms import bipartite, degree_assortativity_coefficient
import csv
import math
from torch_geometric.utils.dropout import dropout_node, dropout_edge, dropout_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run graph sampling (Node Dropout, Edge Dropout, Random Walking).")
    parser.add_argument('--dataset', nargs='?', default='allrecipes', help='dataset name')
    parser.add_argument('--filename', nargs='?', default='dataset.tsv', help='filename')
    parser.add_argument('--sampling_strategies', nargs='+', type=str, default=['RW'],
                        help='graph sampling strategy')
    parser.add_argument('--num_samplings', nargs='?', type=int, default=300,
                        help='number of samplings')
    parser.add_argument('--num_walks', nargs='?', type=int, default=4,
                        help='number of walks (only for RW)')
    parser.add_argument('--random_seed', nargs='?', type=int, default=42,
                        help='random seed for reproducibility')

    return parser.parse_args()


args = parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True


def calculate_statistics(data, info):
    # mapping users and items
    num_users = torch.unique(data[0]).shape[0]
    current_public_to_private_users = {u.item(): idx for idx, u in enumerate(torch.unique(data[0]))}
    current_public_to_private_items = {i.item(): idx + num_users for idx, i in enumerate(torch.unique(data[1]))}
    current_private_to_public_users = {idx: u for u, idx in current_public_to_private_users.items()}
    current_private_to_public_items = {idx: i for i, idx in current_public_to_private_items.items()}

    # rescale nodes indices to feed the edge_index into networkx
    graph = networkx.Graph()
    graph.add_nodes_from([idx for idx, _ in enumerate(torch.unique(data[0]))], bipartite='users')
    graph.add_nodes_from([idx + num_users for idx, _ in enumerate(torch.unique(data[1]))], bipartite='items')
    graph.add_edges_from(list(zip(
        [current_public_to_private_users[u] for u in data[0].tolist()],
        [current_public_to_private_items[i] for i in data[1].tolist()]))
    )

    if networkx.is_connected(graph):
        # basic statistics
        user_nodes, item_nodes = bipartite.sets(graph)
        num_users = len(user_nodes)
        num_items = len(item_nodes)
        m = len(graph.edges())
        delta_g = m / (num_users * num_items)
        k = (2 * m) / (num_users + num_items)
        k_users = m / num_users
        k_items = m / num_items
        space_size = math.sqrt(num_users * num_items) / 1000
        shape = num_users / num_items

        dataset = pd.concat([pd.Series(data[0].tolist()), pd.Series(data[1].tolist())], axis=1)
        sorted_users = dataset.groupby(0).count().sort_values(by=[1]).to_dict()[1]
        sorted_items = dataset.groupby(1).count().sort_values(by=[0]).to_dict()[0]

        def gini_user_term():
            return (num_users + 1 - idx) / (num_users + 1) * sorted_users[user] / m

        def gini_item_term():
            return (num_items + 1 - idx) / (num_items + 1) * sorted_items[item] / m

        gini_terms = 0
        for idx, (user, ratings) in enumerate(sorted_users.items()):
            gini_terms += gini_user_term()

        gini_user = 1 - 2 * gini_terms

        gini_terms = 0
        for idx, (item, ratings) in enumerate(sorted_items.items()):
            gini_terms += gini_item_term()

        gini_item = 1 - 2 * gini_terms

        # TOO SLOW
        # calculate average eccentricity
        # average_eccentricity = sum(list(dict(networkx.eccentricity(graph)).values())) / (users + items)

        # calculate clustering coefficients
        average_clustering_dot = bipartite.average_clustering(graph, mode='dot')
        average_clustering_min = bipartite.average_clustering(graph, mode='min')
        average_clustering_max = bipartite.average_clustering(graph, mode='max')
        average_clustering_dot_users = bipartite.average_clustering(graph, mode='dot', nodes=user_nodes)
        average_clustering_dot_items = bipartite.average_clustering(graph, mode='dot', nodes=item_nodes)
        average_clustering_min_users = bipartite.average_clustering(graph, mode='min', nodes=user_nodes)
        average_clustering_min_items = bipartite.average_clustering(graph, mode='min', nodes=item_nodes)
        average_clustering_max_users = bipartite.average_clustering(graph, mode='max', nodes=user_nodes)
        average_clustering_max_items = bipartite.average_clustering(graph, mode='max', nodes=item_nodes)

        # TOO SLOW
        # calculate node redundancy
        # average_node_redundancy = sum(list(dict(bipartite.node_redundancy(graph)).values())) / (users + items)

        # calculate average assortativity
        average_assortativity = degree_assortativity_coefficient(graph)

        stats_dict = {
            'users': num_users,
            'items': num_items,
            'interactions': m,
            'delta_g': delta_g,
            'k': k,
            'k_users': k_users,
            'k_items': k_items,
            'space_size': space_size,
            'shape': shape,
            'gini_user': gini_user,
            'gini_item': gini_item,
            # 'eccentricity': average_eccentricity,
            'clustering_dot': average_clustering_dot,
            'clustering_min': average_clustering_min,
            'clustering_max': average_clustering_max,
            'clustering_dot_users': average_clustering_dot_users,
            'clustering_dot_items': average_clustering_dot_items,
            'clustering_min_users': average_clustering_min_users,
            'clustering_min_items': average_clustering_min_items,
            'clustering_max_users': average_clustering_max_users,
            'clustering_max_items': average_clustering_max_items,
            'assortativity': average_assortativity
            # 'node_redundancy': average_node_redundancy
        }

        return info.update(stats_dict), None
    else:
        # take the subgraph with maximum extension
        graph = graph.subgraph(max(networkx.connected_components(graph), key=len))

        # basic statistics
        user_nodes, item_nodes = bipartite.sets(graph)
        num_users = len(user_nodes)
        num_items = len(item_nodes)
        m = len(graph.edges())
        delta_g = m / (num_users * num_items)
        k = (2 * m) / (num_users + num_items)
        k_users = m / num_users
        k_items = m / num_items
        space_size = math.sqrt(num_users * num_items) / 1000
        shape = num_users / num_items

        connected_edges = list(graph.edges())
        connected_edges = [[current_private_to_public_users[i] for i, j in connected_edges],
                           [current_private_to_public_items[j] for i, j in connected_edges]]

        connected_dataset = pd.concat([pd.Series(connected_edges[0]), pd.Series(connected_edges[1])], axis=1)
        sorted_users = connected_dataset.groupby(0).count().sort_values(by=[1]).to_dict()[1]
        sorted_items = connected_dataset.groupby(1).count().sort_values(by=[0]).to_dict()[0]

        def gini_user_term():
            return (num_users + 1 - idx) / (num_users + 1) * sorted_users[user] / m

        def gini_item_term():
            return (num_items + 1 - idx) / (num_items + 1) * sorted_items[item] / m

        gini_terms = 0
        for idx, (user, ratings) in enumerate(sorted_users.items()):
            gini_terms += gini_user_term()

        gini_user = 1 - 2 * gini_terms

        gini_terms = 0
        for idx, (item, ratings) in enumerate(sorted_items.items()):
            gini_terms += gini_item_term()

        gini_item = 1 - 2 * gini_terms

        # TOO SLOW
        # calculate average eccentricity
        # average_eccentricity = sum(list(dict(networkx.eccentricity(graph)).values())) / (users + items)

        # calculate clustering coefficients
        average_clustering_dot = bipartite.average_clustering(graph, mode='dot')
        average_clustering_min = bipartite.average_clustering(graph, mode='min')
        average_clustering_max = bipartite.average_clustering(graph, mode='max')
        average_clustering_dot_users = bipartite.average_clustering(graph, mode='dot', nodes=user_nodes)
        average_clustering_dot_items = bipartite.average_clustering(graph, mode='dot', nodes=item_nodes)
        average_clustering_min_users = bipartite.average_clustering(graph, mode='min', nodes=user_nodes)
        average_clustering_min_items = bipartite.average_clustering(graph, mode='min', nodes=item_nodes)
        average_clustering_max_users = bipartite.average_clustering(graph, mode='max', nodes=user_nodes)
        average_clustering_max_items = bipartite.average_clustering(graph, mode='max', nodes=item_nodes)

        # TOO SLOW
        # calculate node redundancy
        # average_node_redundancy = sum(list(dict(bipartite.node_redundancy(graph)).values())) / (users + items)

        # calculate average assortativity
        average_assortativity = degree_assortativity_coefficient(graph)

        stats_dict = {
            'users': num_users,
            'items': num_items,
            'interactions': m,
            'delta_g': delta_g,
            'k': k,
            'k_users': k_users,
            'k_items': k_items,
            'space_size': space_size,
            'shape': shape,
            'gini_user': gini_user,
            'gini_item': gini_item,
            # 'eccentricity': average_eccentricity,
            'clustering_dot': average_clustering_dot,
            'clustering_min': average_clustering_min,
            'clustering_max': average_clustering_max,
            'clustering_dot_users': average_clustering_dot_users,
            'clustering_dot_items': average_clustering_dot_items,
            'clustering_min_users': average_clustering_min_users,
            'clustering_min_items': average_clustering_min_items,
            'clustering_max_users': average_clustering_max_users,
            'clustering_max_items': average_clustering_max_items,
            'assortativity': average_assortativity
            # 'node_redundancy': average_node_redundancy
        }

        edge_index = torch.tensor([connected_edges[0], connected_edges[1]], dtype=torch.int64)

        return info.update(stats_dict), edge_index


def graph_sampling():
    # load public dataset
    dataset = pd.read_csv(f'./data/{args.dataset}/{args.filename}', sep='\t', header=None)
    initial_num_users = dataset[0].nunique()
    initial_num_items = dataset[1].nunique()
    initial_users = dataset[0].unique().tolist()
    initial_items = dataset[1].unique().tolist()
    public_to_private_users = {u: idx for idx, u in enumerate(initial_users)}
    public_to_private_items = {i: idx + initial_num_users for idx, i in enumerate(initial_items)}
    private_to_public_users = {idx: u for u, idx in public_to_private_users.items()}
    private_to_public_items = {idx: i for i, idx in public_to_private_items.items()}

    # build undirected and bipartite graph with networkx
    graph = networkx.Graph()
    graph.add_nodes_from(list(range(initial_num_users)), bipartite='users')
    graph.add_nodes_from(list(range(initial_num_users, initial_num_users + initial_num_items)),
                         bipartite='items')
    graph.add_edges_from(list(zip(
        [public_to_private_users[u] for u in dataset[0].tolist()],
        [public_to_private_items[i] for i in dataset[1].tolist()]))
    )

    connected_edges = None

    # if graph is not connected, retain only the biggest connected portion
    if not networkx.is_connected(graph):
        graph = graph.subgraph(max(networkx.connected_components(graph), key=len))
        connected_edges = list(graph.edges())
        connected_edges = [[private_to_public_users[i] for i, j in connected_edges],
                           [private_to_public_items[j] for i, j in connected_edges]]

    # calculate statistics
    user_nodes, item_nodes = bipartite.sets(graph)
    num_users = len(user_nodes)
    num_items = len(item_nodes)
    m = len(graph.edges())
    delta_g = m / (num_users * num_items)
    k = (2 * m) / (num_users + num_items)
    k_users = m / num_users
    k_items = m / num_items
    space_size = math.sqrt(num_users * num_items) / 1000
    shape = num_users / num_items

    if not connected_edges:
        sorted_users = dataset.groupby(0).count().sort_values(by=[1]).to_dict()[1]
        sorted_items = dataset.groupby(1).count().sort_values(by=[0]).to_dict()[0]
    else:
        connected_dataset = pd.concat([pd.Series(connected_edges[0]), pd.Series(connected_edges[1])], axis=1)
        sorted_users = connected_dataset.groupby(0).count().sort_values(by=[1]).to_dict()[1]
        sorted_items = connected_dataset.groupby(1).count().sort_values(by=[0]).to_dict()[0]

    def gini_user_term():
        return (num_users + 1 - idx) / (num_users + 1) * sorted_users[user] / m

    def gini_item_term():
        return (num_items + 1 - idx) / (num_items + 1) * sorted_items[item] / m

    gini_terms = 0
    for idx, (user, ratings) in enumerate(sorted_users.items()):
        gini_terms += gini_user_term()

    gini_user = 1 - 2 * gini_terms

    gini_terms = 0
    for idx, (item, ratings) in enumerate(sorted_items.items()):
        gini_terms += gini_item_term()

    gini_item = 1 - 2 * gini_terms

    # calculate clustering coefficients
    average_clustering_dot = bipartite.average_clustering(graph, mode='dot')
    average_clustering_min = bipartite.average_clustering(graph, mode='min')
    average_clustering_max = bipartite.average_clustering(graph, mode='max')
    average_clustering_dot_users = bipartite.average_clustering(graph, mode='dot', nodes=user_nodes)
    average_clustering_dot_items = bipartite.average_clustering(graph, mode='dot', nodes=item_nodes)
    average_clustering_min_users = bipartite.average_clustering(graph, mode='min', nodes=user_nodes)
    average_clustering_min_items = bipartite.average_clustering(graph, mode='min', nodes=item_nodes)
    average_clustering_max_users = bipartite.average_clustering(graph, mode='max', nodes=user_nodes)
    average_clustering_max_items = bipartite.average_clustering(graph, mode='max', nodes=item_nodes)

    # calculate average assortativity
    average_assortativity = degree_assortativity_coefficient(graph)

    # print statistics
    print(f'DATASET: {args.dataset}')
    print(f'Number of users: {num_users}')
    print(f'Number of items: {num_items}')
    print(f'Number of interactions: {m}')
    print(f'Density: {delta_g}')
    print(f'Space size: {space_size}')
    print(f'Shape: {shape}')
    print(f'Gini user: {gini_user}')
    print(f'Gini item: {gini_item}')
    print(f'Average degree: {k}')
    print(f'Average user degree: {k_users}')
    print(f'Average item degree: {k_items}')
    print(f'Average clustering (dot): {average_clustering_dot}')
    print(f'Average clustering (min): {average_clustering_min}')
    print(f'Average clustering (max): {average_clustering_max}')
    print(f'Average user clustering (dot): {average_clustering_dot_users}')
    print(f'Average item clustering (dot): {average_clustering_dot_items}')
    print(f'Average user clustering (min): {average_clustering_min_users}')
    print(f'Average item clustering (min): {average_clustering_min_items}')
    print(f'Average user clustering (max): {average_clustering_max_users}')
    print(f'Average item clustering (max): {average_clustering_max_items}')
    print(f'Assortativity: {average_assortativity}')

    # get rows and cols for the dataset and convert them into private
    connected_edges = list(graph.edges())
    connected_edges = [[private_to_public_users[i] for i, j in connected_edges],
                       [private_to_public_items[j] for i, j in connected_edges]]

    # create edge index
    edge_index = torch.tensor([connected_edges[0], connected_edges[1]], dtype=torch.int64)

    # create dictionary to store
    filename_no_extension = args.filename.split('.')[0]
    extension = args.filename.split('.')[1]

    print('\n\nSTART GRAPH SAMPLING...')
    with open(f'./data/{args.dataset}/sampling-stats.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['strategy',
                         'dropout',
                         'users',
                         'items',
                         'interactions',
                         'delta_g',
                         'k',
                         'k_users',
                         'k_items',
                         'clustering_dot',
                         'clustering_min',
                         'clustering_max',
                         'clustering_dot_users',
                         'clustering_dot_items',
                         'clustering_min_users',
                         'clustering_min_items',
                         'clustering_max_users',
                         'clustering_max_items',
                         'assortativity'])
        for _ in range(args.num_samplings):
            gss = random.choice(args.sampling_strategies)
            dr = np.random.uniform(0.7, 0.9)
            if gss == 'ND':
                if not os.path.exists(f'./data/{args.dataset}/node-dropout/'):
                    os.makedirs(f'./data/{args.dataset}/node-dropout/')
                print(f'\n\nRunning NODE DROPOUT with dropout ratio {dr}')
                sampled_edge_index, _, _ = dropout_node(edge_index, p=dr, num_nodes=num_users + num_items)
                current_stats_dict, sampled_graph = calculate_statistics(sampled_edge_index,
                                                                         info={'strategy': 'node dropout',
                                                                               'dropout': dr})
                if sampled_graph:
                    sampled_rows = [private_to_public_users[r] for r in sampled_graph[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_graph[1].tolist()]
                else:
                    sampled_rows = [private_to_public_users[r] for r in sampled_edge_index[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_edge_index[1].tolist()]
                sampled_dataset = pd.concat([pd.Series(sampled_rows), pd.Series(sampled_cols)], axis=1)
                sampled_dataset.to_csv(
                    f'./data/{args.dataset}/node-dropout/{filename_no_extension}-{dr}.{extension}',
                    sep='\t', header=None, index=None)
                writer.writerow(current_stats_dict)
            elif gss == 'ED':
                if not os.path.exists(f'./data/{args.dataset}/edge-dropout/'):
                    os.makedirs(f'./data/{args.dataset}/edge-dropout/')
                print(f'\n\nRunning EDGE DROPOUT with dropout ratio {dr}')
                sampled_edge_index, _ = dropout_edge(edge_index, p=dr)
                current_stats_dict, sampled_graph = calculate_statistics(sampled_edge_index,
                                                                         info={'strategy': 'edge dropout',
                                                                               'dropout': dr})
                if sampled_graph:
                    sampled_rows = [private_to_public_users[r] for r in sampled_graph[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_graph[1].tolist()]
                else:
                    sampled_rows = [private_to_public_users[r] for r in sampled_edge_index[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_edge_index[1].tolist()]
                sampled_dataset = pd.concat([pd.Series(sampled_rows), pd.Series(sampled_cols)], axis=1)
                sampled_dataset.to_csv(
                    f'./data/{args.dataset}/edge-dropout/{filename_no_extension}-{dr}.{extension}',
                    sep='\t', header=None, index=None)
                writer.writerow(current_stats_dict)
            elif gss == 'RW':
                if not os.path.exists(f'./data/{args.dataset}/random-walk/'):
                    os.makedirs(f'./data/{args.dataset}/random-walk/')
                print(f'\n\nRunning RANDOM WALK with dropout ratio {dr}, '
                      f'{args.num_walks} walk length, and {round(k / 2)} walks per node')
                sampled_edge_index, _ = dropout_path(edge_index,
                                                     p=dr,
                                                     walks_per_node=round(k / 2),
                                                     walk_length=args.num_walks,
                                                     num_nodes=num_users + num_items)
                current_stats_dict, sampled_graph = calculate_statistics(sampled_edge_index,
                                                                         info={'strategy': 'random walk',
                                                                               'dropout': dr})
                if sampled_graph:
                    sampled_rows = [private_to_public_users[r] for r in sampled_graph[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_graph[1].tolist()]
                else:
                    sampled_rows = [private_to_public_users[r] for r in sampled_edge_index[0].tolist()]
                    sampled_cols = [private_to_public_items[c] for c in sampled_edge_index[1].tolist()]
                sampled_dataset = pd.concat([pd.Series(sampled_rows), pd.Series(sampled_cols)], axis=1)
                sampled_dataset.to_csv(
                    f'./data/{args.dataset}/random-walk/{filename_no_extension}-{dr}-{args.num_walks}.{extension}',
                    sep='\t', header=None, index=None)
                writer.writerow(current_stats_dict)
            else:
                raise NotImplementedError('This graph sampling strategy has not been implemented yet!')
    print('\n\nEND GRAPH SAMPLING...')


if __name__ == '__main__':
    graph_sampling()