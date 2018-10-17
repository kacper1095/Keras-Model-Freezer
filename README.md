# Keras model freezer

## Complete exemplar usage
I provide complete command sequence to create static, readonly tensorflow graph from model created in [keras](https://github.com/keras-team/keras) framework.

Following commands to those things:
1) Permute convolutional kernel dimensions from `channels_first` to `channels_last` (meaning is explained in [keras docs](https://keras.io/)) and shuffle dense layers weights accordingly.
2) Create `ckpt` files that are usable in raw tensorflow. Additionaly, all op names are dumped (they will be used in later commands). `pbtxt` graph from session is also dumped.
3) Freeze model using `ckpt` and `pbtxt` files to create static, readonly graph. *Input node names* and *output node names* are provided in `op_names.txt` file*.
4) Optimize model for inference ( :) ). This means, few ops are fused to be executed in one step, dropouts are deleted etc. 

```
python th_to_tf_converter.py -w <path_to_weights.h5> -j <not_required_json_architecture.json> -c -o <output_directory_path>
python save_graph_and_checkpoint.py -w <path_to_weights.h5> -j <not_required_json_architecture.json> -o <output_directory_path>
python freeze_graph.py --input_checkpoint <input_model_checkpoint_path.ckpt> --output_graph <output_graph_path.pb> --output_node_names <comma_separated_output_nodes_names> --input_graph <input_graph_path.pbtxt>
python optimize_for_inference.py --input <input_graph_path.pb> --output <output_graph_path.pb> --frozen_graph=True --output_names <comma_separated_output_nodes_names> --input_names <comma_separated_input_nodes_names>
```

*You are required to focus and choose wisely those names. Default name for input tensor is `input_1` for single input network and `Sigmoid`, `Softmax`... for output nodes. It can be different in your case (especially if you customized layer names or something like this). More info on op names are below.

## Guidelines to find node names:
1) Input nodes:
    1. They usually are first in `op_names.txt` file. If you customized layer names, they can be different than `input_1`.
    
2) Output nodes:
    1. They usually have name provided by activation function name with capital letter (ex. `Sigmoid`)
    2. If modified name, they can be found before following sequence of names in `op_names.txt`:
    ```
    ...
    Sigmoid
    Placeholder
    Assign
    Placeholder_1
    Assign_1
    Placeholder_2
    Assign_2
    Placeholder_3
    Assign_3
    Placeholder_4
    Assign_4
    Placeholder_5
    Assign_5
    Placeholder_6
    Assign_6
    ...
    ```
    In this case, `Sigmoid` is our output node name 
3) Naming nodes:
- if you finally found your nodes' names, then to get nodes themselves by name you have to call method on `tf.Graph` object `get_tensor_by_name(<name>)` where `<name>` should be written as `import/<found_node_name>:0`