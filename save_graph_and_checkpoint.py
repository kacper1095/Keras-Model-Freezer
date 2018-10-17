import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras.backend as K

try:
    K.set_image_dim_ordering("tf")
except Exception:
    K.set_image_dim_ordering("channels_last")

import argparse
import tensorflow as tf
from keras.models import load_model, model_from_json


def convert(weights, output, json_file=''):
    if not os.path.exists(output):
        os.makedirs(output)

    if json_file is not None and len(json_file) > 0:
        with open(json_file) as f:
            model = model_from_json(f.read())
        model.load_weights(weights)
    else:
        model = load_model(weights)

    session = K.get_session()
    saver = tf.train.Saver()
    graph = session.graph

    op_names = [op.name for op in graph.get_operations()]

    saver.save(session, os.path.join(output, "model.ckpt"))
    tf.train.write_graph(graph, output, 'model.pbtxt')
    with open(os.path.join(output, "op_names.txt"), "w") as f:
        f.write('\n'.join(op_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for saving checkpoint and graph for further tensorflow optimizization")
    parser.add_argument("-w", "--weights", help="Path to *.h5 file with model weights. It can be used as only parameter"
                                                "to convert model if architecture is saved there")
    parser.add_argument("-j", "--json", default="", help="Path to json file with architecture. Not required")
    parser.add_argument("-o", "--output", help="Output folder")
    args = parser.parse_args()

    convert(args.weights, args.output, args.json)
