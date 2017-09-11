"""
Minimum packages versions:
    - keras >= 1.2.2

For additional optimizations call for scripts:
    - python ./tensorflow/tensorflow/python/tools/optimize_for_inference.py \
        --input=retrained_graph.pb \
        --output=optimized_graph.pb \
        --frozen_graph=True \
        --input_names=Mul \
        --output_names=final_result
    - python ./tensorflow/tensorflow/tools/quantization/quantize_graph.py \
        --input=optimized_graph.pb \
        --output=rounded_graph.pb \
        --output_node_names=final_result \
        --mode=weights_rounded
"""

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import model_from_json
from google.protobuf import text_format
from tensorflow.python.framework import graph_util

import argparse
import tensorflow as tf
import keras.backend as K

K.set_image_dim_ordering('tf')
OUTPUT_PATH = os.path.join('output')


def save_checkpoint_and_return_model(json_path, weights_path, ckpt_path, proto_txt):
    sess = K.get_session()
    K.set_learning_phase(0)
    with open(json_path) as f:
        model = model_from_json(f.read())
    if weights_path != '' and weights_path is not None:
        model.load_weights(weights_path)
    else:
        sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, ckpt_path)
    sess.close()
    tf.train.write_graph(sess.graph_def, os.path.dirname(proto_txt), os.path.basename(proto_txt))
    return model


def get_output_nodes_names(model):
    return [layer.name.split(':')[0] for layer in model.outputs]


def freeze_graph_helper(input_graph, input_saver, input_binary, input_checkpoint,
                        output_node_names, restore_op_name, filename_tensor_name,
                        output_graph, clear_devices, initializer_nodes):
    """Converts all variables in a graph and checkpoint into constants."""

    if not tf.gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1

    if input_saver and not tf.gfile.Exists(input_saver):
        print("Input saver file '" + input_saver + "' does not exist!")
        return -1

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if not tf.train.checkpoint_exists(input_checkpoint):
        print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
        return -1

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    input_graph_def = tf.GraphDef()
    mode = "rb" if input_binary else "r"
    with tf.gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read().decode("utf-8"), input_graph_def)
    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ""
    _ = tf.import_graph_def(input_graph_def, name="")

    with tf.Session() as sess:
        if input_saver:
            with tf.gfile.FastGFile(input_saver, mode) as f:
                saver_def = tf.train.SaverDef()
                if input_binary:
                    saver_def.ParseFromString(f.read())
                else:
                    text_format.Merge(f.read(), saver_def)
                saver = tf.train.Saver(saver_def=saver_def)
                saver.restore(sess, input_checkpoint)
        else:
            sess.run([restore_op_name], {filename_tensor_name: input_checkpoint})
            if initializer_nodes:
                sess.run(initializer_nodes)

        whitelist_nodes = []
        illegal_names = ['keras_learning_phase', 'is_training']
        for node in input_graph_def.node:
            inputs = node.input
            if any([name in node.name for name in illegal_names]) or \
                any([il_name in inp_name for il_name in illegal_names for inp_name in inputs]):
                print(node.name)
                continue
            whitelist_nodes.append(node.name)

        print(len(whitelist_nodes))
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_node_names.split(","),
            variable_names_whitelist=whitelist_nodes)

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


def freeze_graph(output_nodes_names, input_graph, input_checkpoint, output_path):
    freeze_graph_helper(input_graph=input_graph,
                        input_checkpoint=input_checkpoint,
                        output_graph=output_path,
                        output_node_names=','.join(output_nodes_names),
                        clear_devices=True,
                        input_saver='',
                        input_binary='',
                        initializer_nodes='',
                        filename_tensor_name='save/Const:0',
                        restore_op_name='save/restore_all'
                        )


def main(args):
    print('Loading model and saving to tensorflow files')
    model = save_checkpoint_and_return_model(args.structure,
                                             args.weights,
                                             args.saver,
                                             args.graph_txt)
    nodes = get_output_nodes_names(model)
    print('Output nodes: ' + str(nodes))
    print('Freezing graph')
    freeze_graph(nodes, args.graph_txt, args.saver, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script converting tensorflow weights in keras to graph definition'
                                                 'as protobuffer.')
    parser.add_argument('--structure', default='output/structure.json',
                        help='Path to structure of model in json format saved by model_to_json method in keras')
    parser.add_argument('--weights', default='',
                        help='Path to weights of model in h5 format saved by save_weights/save_model methods in keras')
    parser.add_argument('--saver', default='output/model.ckpt',
                        help='Path of saver files to hold for further processing')
    parser.add_argument('--graph_txt', default='output/model.pbtxt',
                        help='Path of intermidiate prototxt graph definition to save')
    parser.add_argument('--output', default='output/graph.pb', help='Path of output file after freeze')
    args = parser.parse_args()
    main(args)
