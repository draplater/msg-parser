import dynet as dn
import nn
from logger import logger


class TriLinear(nn.DynetSaveable):
    def __init__(self, model, input_dim, output_dim):
        super(TriLinear, self).__init__(model)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.w1 = self.add_parameters((output_dim, input_dim))
        self.w2 = self.add_parameters((output_dim, input_dim))
        self.w3 = self.add_parameters((output_dim, input_dim))
        self.bias = self.add_parameters(output_dim)

    def __call__(self, input_1, input_2, input_3):
        return self.w1.expr() * input_1 + self.w2.expr() * input_2 + self.w3.expr() * input_3 + self.bias

    def restore_components(self, components):
        self.w1, self.w2, self.w3, self.bias = components


class ExpressionTransform(object):
    def __init__(self, expr, dims):
        self.expr = expr
        self.dims = dims
        self.dims_shift = self.dims[1:] + (1,)

    def __getitem__(self, item):
        assert len(item) == len(self.dims)
        total_idx = 0
        for dim, idx in zip(self.dims_shift, item):
            total_idx += idx
            total_idx *= dim
        return self.expr[0][int(total_idx)]


class EdgeSiblingEvaluation(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        pass

    def __init__(self, model, options):
        super(EdgeSiblingEvaluation, self).__init__(model)
        self.options = options
        self.activation = nn.activations[options.activation]

        self.ldims = options.lstm_dims

        self.trilinear_layer = TriLinear(self, self.ldims * 2, options.bilinear_dim)

        dense_dims = [options.bilinear_dim] + options.mlp_dims + [1]
        # don't use bias in last transform
        use_bias = [True] * (len(dense_dims) - 2) + [False]

        self.dense_layer = nn.DenseLayers(self, dense_dims, self.activation, use_bias)

    def restore_components(self, components):
        self.trilinear_layer = components.pop(0)
        self.dense_layer = components.pop(0)
        assert not components

    def get_complete_scores(self, lstm_output, use_special=True):
        length = len(lstm_output)

        v1s = [self.trilinear_layer.w1.expr() * i for i in lstm_output]
        v2s = [self.trilinear_layer.w2.expr() * i for i in lstm_output]
        v3s = [self.trilinear_layer.w3.expr() * i for i in lstm_output]

        input_1 = dn.concatenate_cols([v1s[i // (length * length)] for i in range(length * length * length)])
        input_2 = dn.concatenate_cols([v2s[i // length] for i in range(length * length)] * length)
        input_3 = dn.concatenate_cols([v3s[i] for i in range(length)] * length * length)
        biases = dn.concatenate_cols([self.trilinear_layer.bias.expr() for i in range(length * length * length)])

        exprs_3 = self.dense_layer(self.activation(input_1 + input_2 + input_3 + biases))
        scores_3 = exprs_3.npvalue().reshape(length, length, length)

        if use_special:
            input2_1 = dn.concatenate_cols([v1s[i // length] for i in range(length * length)])
            input2_3 = dn.concatenate_cols([v3s[i] for i in range(length)] * length)
            biases2 = dn.concatenate_cols([self.trilinear_layer.bias.expr() for _ in range(length * length)])
            exprs_2 = self.dense_layer(self.activation(input2_1 + input2_3 + biases2))
            scores_2 = exprs_2.npvalue().reshape(length, length)

            return ExpressionTransform(exprs_2, (length, length)), scores_2,\
                   ExpressionTransform(exprs_3, (length, length, length)), scores_3
        else:
            return None, None, \
                   ExpressionTransform(exprs_3, (length, length, length)), scores_3
