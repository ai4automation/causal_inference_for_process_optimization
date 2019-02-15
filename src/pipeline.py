import argparse
from src.aggregate_logs import get_aggregate_df
import networkx as nx
import os


def get_topological_sort_from_bpmn(filename):
    # TODO: Change logic
    bpmn_graph = nx.Graph()

    if os.path.basename(filename) == 'A.1.0.bpmn':
        edge_list = [('Start Event', 'Task 1'),
                     ('Task 1', 'Task 2'),
                     ('Task 2', 'Task 3'),
                     ('Task 3', 'End Event')]
    elif os.path.basename(filename) == 'A.2.0.bpmn':
        edge_list = [('Start Event', 'Task 1'),
                     ('Task 1', 'Task 2'),
                     ('Task 1', 'Task 3'),
                     ('Task 1', 'Task 4'),
                     ('Task 2', 'End Event'),
                     ('Task 3', 'End Event'),
                     ('Task 4', 'End Event')]
    elif os.path.basename(filename) == 'A.2.1.bpmn':
        edge_list = [('Start Event', 'Task 1'),
                     ('Task 1', 'Task 2'),
                     ('Task 1', 'Task 3'),
                     ('Task 1', 'Task 4'),
                     ('Task 2', 'End Event'),
                     ('Task 3', 'End Event'),
                     ('Task 4', 'End Event'),
                     ('Task 2',  'Task 3'),
                     ('Task 4', 'Task 3')]
    elif os.path.basename(filename) == 'A.3.0.bpmn':
        edge_list = []
    elif os.path.basename(filename) == 'A.4.0.bpmn':
        edge_list = []
    elif os.path.basename(filename) == 'A.4.1.bpmn':
        edge_list = []
    else:
        raise ValueError('Unknown BPMN File.')

    bpmn_graph.add_edges_from(edge_list)
    sorted_nodes = nx.topological_sort(bpmn_graph)
    all_sorted_nodes = []
    for node in sorted_nodes:
        all_sorted_nodes.append(str(node) + ':resource')
        all_sorted_nodes.append(str(node) + ':process_instances')

    return all_sorted_nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data to SCM')
    parser.add_argument('bpmn_filename')
    parser.add_argument('mxml_filename')
    args = parser.parse_args()

    groupby_frequency = 'H'

    aggregate_df = get_aggregate_df(args.mxml_filename, groupby_frequency)
    graph = get_topological_sort_from_bpmn(args.bpmn_filename)

    # call structure learning
