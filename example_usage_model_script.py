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


class ExampleModel(object):

    def __init__(self, graph_path):
        self._graph_path = graph_path

    def get_model(self):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            device_count={'GPU': 0}
        )
        config.gpu_options.allow_growth = True
        graph = load_graph(self._graph_path)
        sess = tf.Session(config=config, graph=graph)

        input_images = graph.get_tensor_by_name('import/input_images:0')
        some_sigmoid_output = graph.get_tensor_by_name('import/feature_fusion/Conv_7/Sigmoid:0')
        some_concat_output_if_necessary = graph.get_tensor_by_name('import/feature_fusion/concat_3:0')

        # ... using ops as normally with session
