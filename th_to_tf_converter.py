"""
Sample usage: python th_to_tf_converter.py -j sample_data/architecture.json -w sample_data/weights.h5
"""


import os
import h5py
import argparse
import json
import shutil
import numpy as np
import pdb

from pprint import pprint

OUTPUT = os.path.join('output')

first_dense = True
nb_last_conv = 0


def convert_weight_model_files(weights_file_name='', json_file_name='', clear=False):
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    if weights_file_name != '' and weights_file_name is not None:
        print('converting h5')
        shutil.copyfile(os.path.join(weights_file_name), os.path.join(OUTPUT, os.path.basename(weights_file_name)))
        h5_file = h5py.File(os.path.join(OUTPUT, os.path.basename(weights_file_name)), mode='r+')
        if clear:
            del h5_file['optimizer_weights']
        change_dims_in_weights(h5_file)
        h5_file.close()
    if json_file_name != '' and json_file_name is not None:
        print('converting json')
        with open(os.path.join(json_file_name)) as f:
            json_file = json.load(f)
        change_dims_in_structure(json_file)
        save_file_json(json_file, json_file_name)
    print('done')


def change_dims_in_weights(h5_file):
    global nb_last_conv
    global first_dense

    pdb.set_trace()
    layers = h5_file['model_weights']

    if 'training_config' in h5_file.attrs.keys():
        del h5_file.attrs['training_config']

    if 'model_config' in h5_file.attrs.keys():
        h5_file.attrs['model_config'] = json.dumps(change_dims_in_structure(json.loads(h5_file.attrs['model_config']), debug_print=True))

    nb_of_dense = count_denses(h5_file)
    if nb_of_dense == 1:
        first_dense = False

    for layer_key in layers.keys():
        transform_weights(layers, layer_key, '_W')
        transform_weights(layers, layer_key, '_alphas')


def transform_weights(layers, layer_key, suffix):
    global first_dense
    global nb_last_conv

    layer_weights_key = layer_key + suffix
    if layer_weights_key in layers[layer_key].keys():
        weights = layers[layer_key][layer_weights_key][:]
        if len(weights.shape) == 4:
            weights = weights.transpose((2, 3, 1, 0))
        if len(weights.shape) == 3:
            weights = weights.transpose((1, 2, 0))
            print(weights.shape)
        elif len(weights.shape) == 2 and first_dense:
            nb_rows_dense_layer = weights.shape[0]
            weights = shuffle_rows(weights, nb_last_conv, nb_rows_dense_layer)
            first_dense = False
        nb_last_conv = weights.shape[-1]
        del layers[layer_key][layer_weights_key]
        layers[layer_key].create_dataset(layer_weights_key, data=weights)


def shuffle_rows(original_w, nb_last_conv, nb_rows_dense):
    """
        Note :
        This algorithm to shuffle dense layer rows was provided by Kent Sommers (@kentsommer)
        in a gist : https://gist.github.com/kentsommer/e872f65926f1a607b94c2b464a63d0d3
        """
    converted_w = np.zeros(original_w.shape)
    count = 0
    for index in range(original_w.shape[0]):
        if (index % nb_last_conv) == 0 and index != 0:
            count += 1
        new_index = ((index % nb_last_conv) * nb_rows_dense) + count
        print("index from " + str(index) + " -> " + str(new_index))
        converted_w[index] = original_w[new_index]

    return converted_w


def count_denses(h5_file):
    result = 0
    layers = h5_file['model_weights']
    for layer_key in layers.keys():
        layer_weights_key = layer_key + '_W'
        if layer_weights_key in layers[layer_key].keys():
            weights = layers[layer_key][layer_weights_key][:]
            if len(weights.shape) == 2:
                result += 1
    return result


def change_dims_in_structure(json_file, depth=1, debug_print=False):
    if json_file is None or json_file == {} or json_file == []:
        return
    print depth
    layers = json_file['config']['layers']
    for layer in layers:
        if layer['class_name'] == 'InputLayer':
            dim_ordering = layer['config']['batch_input_shape']
            change_dims_in_place(dim_ordering)
            layer['config']['batch_input_shape'] = dim_ordering
        if 'axis' in layer['config'].keys() and layer['config']['axis'] == 1:
            layer['config']['axis'] = 3
        if 'output_shape' in layer['config'].keys():
            output_dims = layer['config']['output_shape']
            if output_dims is not None:
                change_dims_in_place(output_dims)
        if 'concat_axis' in layer['config'].keys():
            layer['config']['concat_axis'] = 3
        if 'dim_ordering' in layer['config'].keys():
            layer['config']['dim_ordering'] = 'tf'
    if debug_print:
        pprint(json_file)
    return json_file


def change_dims_in_place(dim_ordering):
    dim_ordering[1:3], dim_ordering[3] = dim_ordering[2:4], dim_ordering[1]


def save_file_json(json_file, json_file_name):
    with open(os.path.join(OUTPUT, os.path.basename(json_file_name)), 'w') as f:
        json.dump(json_file, f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Put files in data to convert (both model in json and weights). '
                                             'Script transforms weights of th backend to tf backend')
    ap.add_argument('-w', '--weights', default='', help='File name of weights (include h5 extension)')
    ap.add_argument('-j', '--json', default='', help='File name of structure (include json extension)')
    ap.add_argument('-c', '--clear', help='Clear keras compiled data, like optimizer and loss function', action='store_true')
    args = ap.parse_args()
    convert_weight_model_files(args.weights, args.json, args.clear)
