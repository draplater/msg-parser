# Bidirectional Dynamic RNN
from tensorlayer.layers import *


class BiDynamicRNNLayerM(Layer):
    """
    The :class:`BiDynamicRNNLayer` class is a RNN layer, you can implement vanilla RNN,
    LSTM and GRU with it.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN cell as follow (Note TF1.0+ and TF1.0- are different).
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/>`_
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : an int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    sequence_length : a tensor, array or None.
        The sequence length of each row of input data, see ``Advanced Ops for Dynamic RNN``.
            - If None, it uses ``retrieve_seq_length_op`` to compute the sequence_length, i.e. when the features of padding (on right hand side) are all zeros.
            - If using word embedding, you may need to compute the sequence_length from the ID array (the integer features before word embedding) by using ``retrieve_seq_length_op2`` or ``retrieve_seq_length_op``.
            - You can also input an numpy array.
            - More details about TensorFlow dynamic_rnn in `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`_.
    fw_initial_state : None or forward RNN State
        If None, initial_state is zero_state.
    bw_initial_state : None or backward RNN State
        If None, initial_state is zero_state.
    dropout : `tuple` of `float`: (input_keep_prob, output_keep_prob).
        The input and output keep probability.
    n_layer : an int, default is 1.
        The number of RNN layers.
    return_last : boolean
        If True, return the last output, "Sequence input and single output"\n
        If False, return all outputs, "Synced sequence input and output"\n
        In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolean
        - When return_last = False
        - If True, return 2D Tensor [n_example, 2 * n_hidden], for stacking DenseLayer or computing cost after it.
        - If False, return 3D Tensor [n_example/n_steps(max), n_steps(max), 2 * n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Attributes
    -----------------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, 2 * n_hidden)

    fw(bw)_final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    fw(bw)_initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    sequence_length : a tensor or array, shape = [batch_size]
        The sequence lengths computed by Advanced Opt or the given sequence lengths.

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`ReshapeLayer`.


    References
    ----------
    - `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`_
    - `bidirectional_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/bidirectional_rnn.ipynb>`_
    """
    def __init__(
            self,
            layer = None,
            cell_fn = None,#tf.nn.rnn_cell.LSTMCell,
            cell_init_args = {'state_is_tuple':True},
            n_hidden = 256,
            initializer = tf.random_uniform_initializer(-0.1, 0.1),
            sequence_length = None,
            fw_initial_state = None,
            bw_initial_state = None,
            dropout = None,
            n_layer = 1,
            return_last = False,
            return_seq_2d = False,
            dynamic_rnn_init_args={},
            name = 'bi_dyrnn_layer',
    ):
        Layer.__init__(self, name=name)
        if cell_fn is None:
            raise Exception("Please put in cell_fn")
        if 'GRU' in cell_fn.__name__:
            try:
                cell_init_args.pop('state_is_tuple')
            except:
                pass
        self.inputs = layer.outputs

        print("  [TL] BiDynamicRNNLayer %s: n_hidden:%d in_dim:%d in_shape:%s cell_fn:%s dropout:%s n_layer:%d" %
              (self.name, n_hidden, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout, n_layer))

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("       batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("       non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        with tf.variable_scope(name, initializer=initializer) as vs:
            # Creats the cell function
            # cell_instance_fn=lambda: cell_fn(num_units=n_hidden, **cell_init_args) # HanSheng
            rnn_creator = lambda: cell_fn(num_units=n_hidden, **cell_init_args)

            # Apply dropout
            if dropout:
                if type(dropout) in [tuple, list]:
                    in_keep_prob = dropout[0]
                    out_keep_prob = dropout[1]
                elif isinstance(dropout, float):
                    in_keep_prob, out_keep_prob = dropout, dropout
                else:
                    raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                    "float)")
                try:
                    DropoutWrapper_fn = tf.contrib.rnn.DropoutWrapper
                except:
                    DropoutWrapper_fn = tf.nn.rnn_cell.DropoutWrapper

                    # cell_instance_fn1=cell_instance_fn            # HanSheng
                    # cell_instance_fn=lambda: DropoutWrapper_fn(
                    #                     cell_instance_fn1(),
                    #                     input_keep_prob=in_keep_prob,
                    #                     output_keep_prob=out_keep_prob)
                cell_creator = lambda: DropoutWrapper_fn(rnn_creator(),
                                                         input_keep_prob=in_keep_prob,
                                                         output_keep_prob=1.0)  # out_keep_prob)
            else:
                cell_creator = rnn_creator
            self.fw_cell = cell_creator()
            self.bw_cell = cell_creator()
            # Apply multiple layers
            if n_layer > 1:
                try:
                    MultiRNNCell_fn = tf.contrib.rnn.MultiRNNCell
                except:
                    MultiRNNCell_fn = tf.nn.rnn_cell.MultiRNNCell

                # cell_instance_fn2=cell_instance_fn            # HanSheng
                # cell_instance_fn=lambda: MultiRNNCell_fn([cell_instance_fn2() for _ in range(n_layer)])
                self.fw_cell = MultiRNNCell_fn([cell_creator() for _ in range(n_layer)])
                self.bw_cell = MultiRNNCell_fn([cell_creator() for _ in range(n_layer)])

            if dropout:
                self.fw_cell = DropoutWrapper_fn(self.fw_cell,
                                                 input_keep_prob=1.0, output_keep_prob=out_keep_prob)
                self.bw_cell = DropoutWrapper_fn(self.bw_cell,
                                                 input_keep_prob=1.0, output_keep_prob=out_keep_prob)

            # self.fw_cell=cell_instance_fn()
            # self.bw_cell=cell_instance_fn()
            # Initial state of RNN
            if fw_initial_state is None:
                self.fw_initial_state = self.fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.fw_initial_state = fw_initial_state
            if bw_initial_state is None:
                self.bw_initial_state = self.bw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.bw_initial_state = bw_initial_state
            # Computes sequence_length
            if sequence_length is None:
                try: ## TF1.0
                    sequence_length = retrieve_seq_length_op(
                        self.inputs if isinstance(self.inputs, tf.Tensor) else tf.stack(self.inputs))
                except: ## TF0.12
                    sequence_length = retrieve_seq_length_op(
                        self.inputs if isinstance(self.inputs, tf.Tensor) else tf.pack(self.inputs))

            outputs, (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.fw_cell,
                cell_bw=self.bw_cell,
                inputs=self.inputs,
                sequence_length=sequence_length,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state,
                **dynamic_rnn_init_args
            )
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            print("     n_params : %d" % (len(rnn_variables)))
            # Manage the outputs
            try: # TF1.0
                outputs = tf.concat(outputs, 2)
            except: # TF0.12
                outputs = tf.concat(2, outputs)
            if return_last:
                # [batch_size, 2 * n_hidden]
                self.outputs = advanced_indexing_op(outputs, sequence_length)
            else:
                # [batch_size, n_step(max), 2 * n_hidden]
                if return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, 2 * n_hidden]
                    try: # TF1.0
                        self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, 2 * n_hidden])
                    except: # TF0.12
                        self.outputs = tf.reshape(tf.concat(1, outputs), [-1, 2 * n_hidden])
                else:
                    # <akara>:
                    # 3D Tensor [batch_size, n_steps(max), 2 * n_hidden]
                    self.outputs = outputs
                    pass

        # Final state
        self.fw_final_states = states_fw
        self.bw_final_states = states_bw

        self.sequence_length = sequence_length

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )

