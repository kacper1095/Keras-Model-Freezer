import argparse

import tensorflow as tf


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph


def load_model(graph_path, input_node_names, output_node_names):
    input_names = input_node_names.split(',')
    output_names = output_node_names.split(',')

    config = tf.ConfigProto(
        allow_soft_placement=True,
        device_count={'GPU': 0}
    )
    config.gpu_options.allow_growth = True
    graph = load_graph(graph_path)
    sess = tf.Session(config=config, graph=graph)

    inputs = [graph.get_tensor_by_name('import/{}:0'.format(input_name)) for input_name in input_names]
    outputs = [graph.get_tensor_by_name('import/{}:0'.format(output_name)) for output_name in output_names]

    print(inputs)
    print(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Scripts to test whether model loads or not")
    parser.add_argument("-g", "--graph", help="Input graph file path")
    parser.add_argument("-o", "--output_node_names", help="Output node names, comma separated")
    parser.add_argument("-i", "--input_node_names", help="Input node names, comma separated")
    args = parser.parse_args()

    load_model(args.graph, args.input_node_names, args.output_node_names)
