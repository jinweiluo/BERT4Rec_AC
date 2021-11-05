#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import paddle
import paddle.nn as nn


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.,
                 param_initializer=None,
                 name='multi_head_att'):
        super(MultiHeadAttention, self).__init__()
        self.q_linear = nn.Linear(in_features=d_model,
                                  out_features=d_key * n_head,
                                  weight_attr=paddle.ParamAttr(
                                      name=name + '_query_fc.w_0',
                                      initializer=param_initializer),
                                  bias_attr=name + '_query_fc.b_0')
        self.k_linear = nn.Linear(in_features=d_model,
                                  out_features=d_key * n_head,
                                  weight_attr=paddle.ParamAttr(
                                      name=name + '_key_fc.w_0',
                                      initializer=param_initializer),
                                  bias_attr=name + '_key_fc.b_0')
        self.v_linear = nn.Linear(in_features=d_model,
                                  out_features=d_value * n_head,
                                  weight_attr=paddle.ParamAttr(
                                      name=name + '_value_fc.w_0',
                                      initializer=param_initializer),
                                  bias_attr=name + '_value_fc.b_0')

        self.out_linear = nn.Linear(in_features=d_key * n_head,
                                    out_features=d_model,
                                    weight_attr=paddle.ParamAttr(
                                        name=name + '_output_fc.w_0',
                                        initializer=param_initializer),
                                    bias_attr=name + '_output_fc.b_0')
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def forward(self, queries, keys, values, attn_bias):
        keys = queries if keys is None else keys
        values = keys if values is None else values

        q = self.q_linear(queries)
        k = self.k_linear(keys)
        v = self.v_linear(values)

        hidden_size = q.shape[-1]

        q = paddle.reshape(
            x=q, shape=[0, 0, self.n_head, hidden_size // self.n_head])
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])  # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        k = paddle.reshape(
            x=k, shape=[0, 0, self.n_head, hidden_size // self.n_head])
        k = paddle.transpose(x=k, perm=[0, 2, 1, 3])  # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        v = paddle.reshape(
            x=v, shape=[0, 0, self.n_head, hidden_size // self.n_head])
        v = paddle.transpose(x=v, perm=[0, 2, 1, 3])  # [batch_size, n_head, max_sequence_len, hidden_size_per_head]

        # scale dot product attention
        attention_scores = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.multiply(attention_scores,
                                  paddle.to_tensor(1.0 / math.sqrt(float(self.d_key)), dtype='float32')
                                  )

        if attn_bias is not None:
            product += attn_bias
        weights = nn.functional.softmax(product)
        if self.dropout_rate:
            weights = nn.functional.dropout(
                weights,
                p=self.dropout_rate,
                mode="upscale_in_train",
                training=self.training)
        out = paddle.matmul(weights, v)
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        out = self.out_linear(out)
        return out


class NormalizeLayer(nn.Layer):
    def __init__(self,
                 norm_shape=768,
                 name=''):
        super(NormalizeLayer, self).__init__()
        self.name = name
        self.LayerNormal = nn.LayerNorm(norm_shape,
                                        epsilon=1e-05,
                                        weight_attr=paddle.ParamAttr(
                                            name=self.name + '_layer_norm_scale',
                                            initializer=nn.initializer.Constant(1.)),
                                        bias_attr=paddle.ParamAttr(
                                            name=self.name + '_layer_norm_bias',
                                            initializer=nn.initializer.Constant(0.))
                                        )

    def forward(self, out):
        out_dtype = out.dtype
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float32")
        out = self.LayerNormal(out)
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float16")
        return out


class NormalizeDropLayer(nn.Layer):
    def __init__(self,
                 dropout_rate=0.,
                 norm_shape=768,
                 name=''):
        super(NormalizeDropLayer, self).__init__()
        self.name = name
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate, mode="upscale_in_train")
        self.LayerNormal = nn.LayerNorm(norm_shape,
                                        epsilon=1e-05,
                                        weight_attr=paddle.ParamAttr(
                                            name=self.name + '_layer_norm_scale',
                                            initializer=nn.initializer.Constant(1.)),
                                        bias_attr=paddle.ParamAttr(
                                            name=self.name + '_layer_norm_bias',
                                            initializer=nn.initializer.Constant(0.))
                                        )

    def forward(self, out):
        out_dtype = out.dtype
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float32")
        out = self.LayerNormal(out)
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float16")
        if self.dropout_rate:
            out = self.dropout(out)
        return out


class DropResidualNormalizeLayer(nn.Layer):
    def __init__(self,
                 dropout_rate=0.,
                 norm_shape=768,
                 name=''):
        super(DropResidualNormalizeLayer, self).__init__()
        self.name = name
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate, mode="upscale_in_train")
        self.LayerNormal = nn.LayerNorm(norm_shape,
                                        epsilon=1e-05,
                                        weight_attr=paddle.ParamAttr(
                                            name=self.name + '_layer_norm_scale',
                                            initializer=nn.initializer.Constant(1.)),
                                        bias_attr=paddle.ParamAttr(
                                            name=self.name + '_layer_norm_bias',
                                            initializer=nn.initializer.Constant(0.))
                                        )

    def forward(self, out, prev_out=None):
        if self.dropout_rate:
            out = self.dropout(out)
        if prev_out is not None:
            out = out + prev_out
        out_dtype = out.dtype
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float32")
        out = self.LayerNormal(out)
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float16")
        return out


class FFN(nn.Layer):
    def __init__(self,
                 d_inner_hid,
                 d_hid,
                 hidden_act,
                 param_initializer=None,
                 name='ffn'):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(in_features=d_hid,
                             out_features=d_inner_hid,
                             weight_attr=paddle.ParamAttr(
                                 name=name + '_fc_0.w_0',
                                 initializer=param_initializer),
                             bias_attr=name + '_fc_0.b_0')
        self.hidden_act = hidden_act
        self.fc2 = nn.Linear(in_features=d_inner_hid,
                             out_features=d_hid,
                             weight_attr=paddle.ParamAttr(
                                 name=name + '_fc_1.w_0',
                                 initializer=param_initializer),
                             bias_attr=name + '_fc_1.b_0')

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.hidden_act(hidden)
        out = self.fc2(hidden)
        return out


class EncoderLayer(nn.Layer):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 attention_dropout,
                 hidden_act,
                 param_initializer=None,
                 name=''):
        super(EncoderLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout,
            param_initializer=param_initializer,
            name=name + '_multi_head_att')

        self.drop_residual_normalize_layer_1 = DropResidualNormalizeLayer(attention_dropout, norm_shape=d_model,
                                                                          name=name + '_post_att')

        self.positionwise_feed_layer = FFN(d_inner_hid,
                                           d_model,
                                           hidden_act,
                                           param_initializer,
                                           name=name + '_ffn')
        self.drop_residual_normalize_layer_2 = DropResidualNormalizeLayer(attention_dropout, norm_shape=d_model,
                                                                          name=name + '_post_ffn')

    def forward(self, enc_input, attn_bias):
        multi_output = self.multi_head_attn(queries=enc_input, keys=None, values=None, attn_bias=attn_bias)
        attn_output = self.drop_residual_normalize_layer_1(prev_out=enc_input, out=multi_output)
        ffd_output = self.positionwise_feed_layer(attn_output)
        out = self.drop_residual_normalize_layer_2(prev_out=attn_output, out=ffd_output)

        return out


class Encoder(nn.Layer):
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 attention_dropout,
                 hidden_act,
                 param_initializer=None,
                 name=''):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.LayerList([EncoderLayer(
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            attention_dropout,
            hidden_act,
            param_initializer,
            name + '_layer_' + str(i))
            for i in range(n_layer)])

    def forward(self, enc_input, attn_bias):
        enc_output = None
        for enc in self.encoder_layer:
            enc_output = enc(enc_input, attn_bias)
            enc_input = enc_output
        return enc_output
