"""
NeurodataLab LLC 04.04.2018
Created by Andrey Belyaev
"""
import mxnet as mx

Activation = mx.sym.Activation
BatchNorm = mx.sym.BatchNorm
BidirectionalCell = mx.rnn.BidirectionalCell
BlockGrad = mx.sym.BlockGrad
Concat = mx.sym.Concat
Conv = mx.sym.Convolution
Dropout = mx.sym.Dropout
DropoutCell = mx.rnn.DropoutCell
FC = mx.sym.FullyConnected
Flatten = mx.sym.Flatten
LinearRegression = mx.sym.LinearRegressionOutput
LogisticRegression = mx.sym.LogisticRegressionOutput
LSTMCell = mx.rnn.LSTMCell
MAERegression = mx.sym.MAERegressionOutput
Pool = mx.sym.Pooling
Reshape = mx.sym.Reshape
RNNCell = mx.rnn.RNNCell
SequentialRNNCell = mx.rnn.SequentialRNNCell
SoftmaxOutput = mx.sym.SoftmaxOutput
Transpose = mx.sym.transpose
Variable = mx.sym.Variable
ZoneoutCell = mx.rnn.ZoneoutCell

Relu = lambda d: Activation(data=d, act_type='relu')
Sigmoid = lambda d: Activation(data=d, act_type='sigmoid')
Tanh = lambda d: Activation(data=d, act_type='tanh')


class ImagesModel:
    def __init__(self, batch_size, num_samples, input_data, image_shape,
                 pretrained_path, pretrained_epoch, pretrained_input_name, pretrained_out_name,
                 conv3d_pretrained_path=None, conv3d_pretrained_epoch=None,
                 conv3d_pretrained_input_name=None, conv3d_pretrained_out_name=None,):
        self.batch_size, self.num_samples, self.image_shape = batch_size, num_samples, image_shape
        self.data_var = Variable(input_data) if isinstance(input_data, str) else input_data
        self.sym, self.arg_params, self.aux_params = mx.model.load_checkpoint(pretrained_path, pretrained_epoch)
        self.sym = self.sym.get_internals()[pretrained_out_name + '_output']
        self.pretrained_input_name = pretrained_input_name

        if conv3d_pretrained_path is not None:
            self.sym_3d, arg_params, aux_params = mx.model.load_checkpoint(conv3d_pretrained_path,
                                                                           conv3d_pretrained_epoch)
            self.arg_params.update(arg_params)
            self.aux_params.update(aux_params)
            # self.sym_3d = self.sym_3d.get_internals()[conv3d_pretrained_out_name + '_output']
            # self.pretrained_input_name_3d = conv3d_pretrained_input_name
        else:
            self.sym_3d = None

    @property
    def features_resnet(self):
        reshaped_data = Reshape(self.data_var, (self.batch_size * self.num_samples,) + self.image_shape)
        features = self.sym(**{self.pretrained_input_name: reshaped_data})

        features_reshaped = Reshape(features, (self.batch_size, self.num_samples, -1))

        return features_reshaped, Flatten(features)

    @property
    def features_3d(self):
        data = Reshape(self.data_var, (self.batch_size, self.image_shape[0], self.num_samples) + self.image_shape[1:])
        # if self.sym_3d is None:
        data = BatchNorm(data)

        conv0_1 = Conv(data, kernel=(3, 3, 3), num_filter=64, name='3d_conv_0')
        pool0 = Pool(conv0_1, kernel=(1, 2, 2), pool_type='max', name='3d_conv_pool0')

        conv1_1 = Conv(pool0, kernel=(3, 3, 3), num_filter=64, name='3d_conv_1_1')
        relu1_1 = Relu(conv1_1)
        # conv1_2 = Conv(relu1_1, kernel=(3, 3, 3), num_filter=32, name='3d_conv_1_2')
        # relu1_2 = Relu(conv1_2)
        pool1 = Pool(relu1_1, kernel=(2, 2, 2), stride=(2, 2, 2), pool_type='max', name='3d_conv_pool1')
        dropout1 = Dropout(pool1, p=.2)
        dropout1 = BatchNorm(dropout1)

        conv2_1 = Conv(dropout1, kernel=(3, 3, 3), num_filter=128, name='3d_conv_2_1')
        relu2_1 = Relu(conv2_1)
        conv2_2 = Conv(relu2_1, kernel=(3, 3, 3), num_filter=128, name='3d_conv_2_2')
        relu2_2 = Relu(conv2_2)
        pool2 = Pool(relu2_2, kernel=(2, 2, 2), stride=(2, 2, 2), pool_type='max', name='3d_conv_pool2')
        dropout2 = Dropout(pool2, p=.2)
        dropout2 = BatchNorm(dropout2)

        conv3_1 = Conv(dropout2, kernel=(3, 3, 3), num_filter=256, name='3d_conv_3_1')
        relu3_1 = Relu(conv3_1)
        conv3_2 = Conv(relu3_1, kernel=(3, 3, 3), num_filter=256, name='3d_conv_3_2')
        relu3_2 = Relu(conv3_2)
        pool3 = Pool(relu3_2, kernel=(2, 2, 2), stride=(2, 2, 2), pool_type='max', name='3d_conv_pool3')
        dropout3 = Dropout(pool3, p=.2)
        dropout3 = BatchNorm(dropout3)

        # conv4 = Conv(dropout3, kernel=(1, 1, 1), num_filter=256, name='3d_conv_4')
        # relu4 = Relu(conv4)
        # pool4 = Pool(dropout2, kernel=(1, 1, 1), pool_type='avg', global_pool=True, name='3d_conv_avg_pool')

        features = dropout3

        # else:
        #     features = self.sym_3d(**{self.pretrained_input_name_3d: data})

        return FC(features, num_hidden=256)


class SpectersModel:
    def __init__(self, batch_size, num_samples, input_data, spec_shape,
                 pretrained_path=None, pretrained_epoch=None, pretrained_input_name=None, pretrained_out_name=None
                 ):
        self.batch_size, self.num_samples, self.spec_shape = batch_size, num_samples, spec_shape
        self.data_var = Variable(input_data) if isinstance(input_data, str) else input_data
        self.arg_params, self.aux_params = {}, {}
        if pretrained_path is not None:
            self.sym, arg_params, aux_params = mx.model.load_checkpoint(pretrained_path, pretrained_epoch)
            self.arg_params.update(arg_params)
            self.aux_params.update(aux_params)
            self.sym = self.sym.get_internals()[pretrained_out_name + '_output']
            self.pretrained_input_name = pretrained_input_name
        else:
            self.sym = None

    @staticmethod
    def conv_block(prev, block_info, block_num=0):
        name = 'block%d' % block_num
        p = prev
        for n, (kernel_size, num_filter) in enumerate(block_info):
            p = Conv(data=p, kernel=kernel_size, num_filter=num_filter, name=name + '_conv%d' % n)
            p = Activation(data=p, act_type='relu', name=name + '_relu%d' % n)
        return p

    @property
    def features(self):
        if self.sym is None:
            return self.features_old
        reshaped_data = Reshape(self.data_var, (self.batch_size * self.num_samples,) + self.spec_shape)
        features = self.sym(**{self.pretrained_input_name: reshaped_data})
        # features = FC(features, num_hidden=128)
        features_reshaped = Reshape(features, (self.batch_size, self.num_samples, -1))

        return features_reshaped, Flatten(features)

    @property
    def features_old(self):
        reshaped_data = Reshape(self.data_var, (self.batch_size * self.num_samples,) + self.spec_shape)

        bn1 = BatchNorm(reshaped_data)
        block1 = self.conv_block(bn1, [((3, 3), 32)] * 2, block_num=1)
        pool1 = Pool(block1, pool_type='max', kernel=(2, 2), stride=(2, 2), name='spec_pool1')
        dropout1 = Dropout(pool1, p=.2)

        bn2 = BatchNorm(dropout1)
        block2 = self.conv_block(bn2, [((3, 3), 64)] * 2, block_num=2)
        pool2 = Pool(block2, pool_type='max', kernel=(2, 2), stride=(2, 2), name='spec_pool2')
        dropout2 = Dropout(pool2, p=.2)

        bn3 = BatchNorm(dropout2)
        block3 = self.conv_block(bn3, [((3, 3), 128)] * 2, block_num=3)
        pool3 = Pool(block3, pool_type='max', kernel=(2, 2), stride=(2, 2), name='spec_pool3')
        dropout3 = Dropout(pool3, p=.2)

        features = Flatten(dropout3)
        features_reshaped = Reshape(features, (self.batch_size, self.num_samples, -1))

        return features_reshaped, features


class BodyModel:
    def __init__(self, batch_size, num_samples, input_data):
        self.batch_size, self.num_samples = batch_size, num_samples
        self.data_var = Variable(input_data) if isinstance(input_data, str) else input_data
        self.arg_params, self.aux_params = {}, {}

    @property
    def features(self):
        data_reshaped = Reshape(self.data_var, (self.batch_size * self.num_samples, -1))

        bn = BatchNorm(data_reshaped)
        features = FC(bn, num_hidden=128)
        features_reshaped = Reshape(features, (self.batch_size, self.num_samples, -1))

        return features_reshaped, features


class MajorModel:
    def __init__(self, batch_size, data_names, num_images, num_bodies, num_specters,
                 image_shape, specter_shape, label_names, label_shape,
                 image_pretrained_path, image_pretrained_epoch, image_pretrained_input_name, image_pretrained_out_name,
                 specter_pretrained_path=None, specter_pretrained_epoch=None,
                 specter_pretrained_input_name=None, specter_pretrained_out_name=None,
                 conv3d_pretrained_path=None, conv3d_pretrained_epoch=None,
                 conv3d_pretrained_input_name=None, conv3d_pretrained_out_name=None,
                 context=(mx.gpu(0), mx.gpu(1)), norm_alpha=None):
        self.ctx = context if isinstance(context, (list, tuple)) else [context]
        self.batch_size = batch_size / len(self.ctx)

        assert num_images is not None and num_bodies is not None and num_specters is not None
        assert sorted(data_names) == sorted(('images', 'body', 'specters'))
        self.data_names = data_names
        self.num_images, self.num_bodies, self.num_specters = num_images, num_bodies, num_specters
        self.image_shape, self.specter_shape = image_shape, specter_shape
        self.label_names, self.label_shape = label_names, label_shape

        self.image_var = Variable('images')
        self.body_var = Variable('body')
        self.specter_var = Variable('specters')
        self.label_var = Variable(self.label_names[0])
        label_var_sum = mx.sym.expand_dims(mx.sym.sum(self.label_var, axis=1), axis=-1)
        self.normal_label_var = mx.sym.broadcast_div(self.label_var, label_var_sum)

        self.image_model = ImagesModel(self.batch_size, self.num_images, self.image_var, self.image_shape,
                                       image_pretrained_path, image_pretrained_epoch, image_pretrained_input_name,
                                       image_pretrained_out_name, conv3d_pretrained_path, conv3d_pretrained_epoch,
                                       conv3d_pretrained_input_name, conv3d_pretrained_out_name)
        self.body_model = BodyModel(self.batch_size, self.num_bodies, self.body_var)
        self.specter_model = SpectersModel(self.batch_size, self.num_specters, self.specter_var, self.specter_shape,
                                           specter_pretrained_path, specter_pretrained_epoch,
                                           specter_pretrained_input_name, specter_pretrained_out_name)

        self.arr_label = Variable(self.label_names[1])
        self.val_label = Variable(self.label_names[2])

        self.arg_params, self.aux_params = {}, {}
        for m in (self.image_model, self.body_model, self.specter_model):
            self.arg_params.update(m.arg_params)
            self.aux_params.update(m.aux_params)

        self.normal_alpha = norm_alpha

    @property
    def train(self):
        return self.get_model(True)

    def features_loss(self, features, num_samples, feature_name, grad_scale=1.):
        fc = FC(features, num_hidden=self.label_shape, name='middle_%s_end_fc' % feature_name)
        label = mx.sym.expand_dims(self.normal_label_var, axis=1)
        label = mx.sym.repeat(label, num_samples, axis=1)
        label = Reshape(label, (self.batch_size * num_samples, -1))
        return SoftmaxOutput(fc, label=label, grad_scale=grad_scale, name='middle_%s_loss' % feature_name)

    def lstm_model(self, features, num_samples, feature_name, num_lstm=2, num_hidden=128):
        stack = SequentialRNNCell()
        for i in range(num_lstm):
            lstm1 = LSTMCell(num_hidden=num_hidden, prefix='lstm_%s_l_l_%d' % (feature_name, i))
            lstm2 = LSTMCell(num_hidden=num_hidden, prefix='lstm_%s_r_l_%d' % (feature_name, i))
            bi_cell = BidirectionalCell(ZoneoutCell(lstm1, zoneout_states=0.2, zoneout_outputs=0.2),
                                        ZoneoutCell(lstm2, zoneout_states=0.2, zoneout_outputs=0.2))
            stack.add(bi_cell)
            stack.add(DropoutCell(0.2, prefix='dropout_%s_l_%d' % (feature_name, i)))
        stack.reset()

        outputs, states = stack.unroll(num_samples, inputs=features, merge_outputs=True)
        return outputs

    def cls_logistic_regression(self, features, grad_scale=1.):
        grad_scale = grad_scale if isinstance(grad_scale, (tuple, list)) else [grad_scale] * self.label_shape
        losses = []
        for n in range(self.label_shape):
            fc = FC(features, num_hidden=1, name='log_reg_%d_cls_end_fc' % n)
            label = mx.sym.slice_axis(self.label_var, begin=n, end=n + 1, axis=1)
            losses.append(LogisticRegression(fc, label=label, name='log_reg_%d_cls_loss' % n, grad_scale=grad_scale[n]))
        return losses

    def anno_reg_linear_regression(self, features, grad_scale=1.):
        grad_scale = grad_scale if isinstance(grad_scale, (tuple, list)) else [grad_scale] * self.label_shape
        losses = []
        for n in range(self.label_shape):
            fc = FC(features, num_hidden=1, name='anno_reg_%d_cls_end_fc' % n)
            label = mx.sym.slice_axis(self.reg_label_var, begin=n, end=n + 1, axis=1)
            losses.append(LinearRegression(fc, label=label, name='anno_reg_%d_cls_loss' % n, grad_scale=grad_scale[n]))
        return losses

    def cls_softmax(self, feat, grad_scale=1., name='softmax'):
        fc = FC(feat, num_hidden=self.label_shape, name='%s_end_fc' % name)
        return SoftmaxOutput(fc, label=self.normal_label_var, name='%s_out_loss' % name, grad_scale=grad_scale)

    def arr_val_loss(self, feat, grad_scale=(1., 1.)):
        fc_arr = FC(feat, num_hidden=1, name='arr_end_fc')
        arr_loss = LinearRegression(fc_arr, label=self.arr_label, name='arr_out_loss', grad_scale=grad_scale[0])

        fc_val = FC(feat, num_hidden=1, name='val_end_fc')
        val_loss = LinearRegression(fc_val, label=self.val_label, name='val_out_loss', grad_scale=grad_scale[1])

        return arr_loss, val_loss

    def rnn_model(self, features, num_samples, feature_name, num_rnn=2, num_hidden=128):
        stack = SequentialRNNCell()
        for i in range(num_rnn):
            lstm1 = RNNCell(num_hidden=num_hidden, prefix='rnn_%s_l_l_%d' % (feature_name, i))
            lstm2 = RNNCell(num_hidden=num_hidden, prefix='rnn_%s_r_l_%d' % (feature_name, i))
            bi_cell = BidirectionalCell(ZoneoutCell(lstm1, zoneout_states=0.2, zoneout_outputs=0.2),
                                        ZoneoutCell(lstm2, zoneout_states=0.2, zoneout_outputs=0.2))
            stack.add(bi_cell)
            stack.add(DropoutCell(0.2, prefix='dropout_%s_l_%d' % (feature_name, i)))
        stack.reset()

        outputs, states = stack.unroll(num_samples, inputs=features, merge_outputs=True)
        return outputs

    def get_model(self, comp=True):
        im_ft_rs, im_ft = self.image_model.features_resnet  # (BS, N, F1), (BS * N, F1), F=128
        # im_3d_ft = self.image_model.features_3d
        b_ft_rs, b_ft = self.body_model.features  # (BS, N, F2), (BS * N, F2), F=64
        sp_ft_rs, sp_ft = self.specter_model.features  # (BS, N, F3), (BS * N, F3), F=128

        # im_middle_loss = self.features_loss(im_ft, self.num_images, 'images', grad_scale=1. / self.num_images)
        # im_3d_middle_loss = self.cls_softmax(im_3d_ft, grad_scale=1., name='3d_softmax')
        # b_middle_loss = self.features_loss(b_ft, self.num_bodies, 'body', gra0.1)
        # sp_middle_loss = self.features_loss(sp_ft, self.num_specters, 'specters', grad_scale=1. / self.num_specters)

        # im_lstm_features = self.lstm_model(im_ft_rs, self.num_images, 'images', num_lstm=3, num_hidden=256)
        im_lstm_features = self.rnn_model(im_ft_rs, self.num_images, 'images', num_rnn=2, num_hidden=128)
        # b_lstm_features = self.lstm_model(b_ft_rs, self.num_bodies, 'body', num_lstm=2, num_hidden=128)
        b_lstm_features = self.rnn_model(b_ft_rs, self.num_bodies, 'body', num_rnn=2, num_hidden=128)
        # sp_lstm_features = self.lstm_model(sp_ft_rs, self.num_specters, 'specters', num_lstm=3, num_hidden=128)
        sp_lstm_features = self.rnn_model(sp_ft_rs, self.num_specters, 'specters', num_rnn=3, num_hidden=128)

        im_mean_features = mx.sym.mean(im_lstm_features, axis=1)
        b_mean_features = mx.sym.mean(b_lstm_features, axis=1)
        sp_mean_features = mx.sym.mean(sp_lstm_features, axis=1)

        im_features = im_mean_features
        b_features = b_mean_features
        sp_features = sp_mean_features

        # im_conv_features = Conv(im_lstm_features, kernel=(3, ), stride=(2, ), num_filter=128, pad=(1, ))
        # im_conv_features = Relu(im_conv_features)
        # b_conv_features = Conv(b_lstm_features, kernel=(3, ), stride=(1, ), num_filter=128, pad=(1, ))
        # b_conv_features = Relu(b_conv_features)
        # sp_conv_features = Conv(sp_lstm_features, kernel=(3, ), stride=(2, ), num_filter=128, pad=(1, ))
        # sp_conv_features = Relu(sp_conv_features)
        #
        # im_features = Conv(im_conv_features, kernel=(1, ), num_filter=32)
        # im_features = Flatten(Pool(im_features, kernel=(1, ), pool_type='max', global_pool=True))
        # b_features = Conv(b_conv_features, kernel=(1, ), num_filter=32)
        # b_features = Flatten(Pool(b_features, kernel=(1,), pool_type='max', global_pool=True))
        # sp_features = Conv(sp_conv_features, kernel=(1, ), num_filter=32)
        # sp_features = Flatten(Pool(sp_features, kernel=(1,), pool_type='max', global_pool=True))

        im_dp_features = Dropout(im_features, p=0.2)
        b_dp_features = Dropout(b_features, p=0.2)
        sp_dp_features = Dropout(sp_features, p=0.2)

        c = Concat(im_dp_features, b_dp_features, sp_dp_features)
        feat = FC(c, num_hidden=64)

        # im_fc = FC(im_dp_features, num_hidden=64)
        # im_3d_fc = FC(im_3d_ft, num_hidden=64)
        # b_fc = FC(b_dp_features, num_hidden=64)
        # sp_fc = FC(sp_dp_features, num_hidden=64)

        # c = Concat(im_fc, im_3d_fc, b_fc, sp_fc)
        # feat = FC(c, num_hidden=64)
        # feat = im_3d_fc

        if self.normal_alpha is not None:
            norm_feat = mx.sym.L2Normalization(feat, mode='instance', name='feat_norm', eps=.001)
            scale_feat = norm_feat * self.normal_alpha
            norm_losses = [LinearRegression(feat, label=scale_feat, name='norm_loss', grad_scale=0.02)]
        else:
            norm_losses = []

        end_softmax_loss = self.cls_softmax(feat, grad_scale=1.)
        log_reg_losses = self.cls_logistic_regression(feat, grad_scale=0.5)
        end_arr_loss, end_val_loss = self.arr_val_loss(feat, grad_scale=(1., 1.))

        out_sym = mx.sym.Group([end_softmax_loss]
                               + [end_arr_loss, end_val_loss]
                               + log_reg_losses
                               # + [im_middle_loss, im_3d_middle_loss, sp_middle_loss]
                               # + [mx.sym.MakeLoss(self.reg_label_var)]
                               # + [mx.sym.MakeLoss(self.reg_label_var)]
                               # + anno_reg_losses
                               # + [mx.sym.MakeLoss(f, grad_scale=1e-11) for f in (im_fc, b_fc, sp_fc)]
                               # + [mx.sym.MakeLoss(im_3d_fc)]
                               + norm_losses)

        if comp:
            return (mx.mod.Module(out_sym, data_names=self.data_names, label_names=self.label_names, context=self.ctx),
                    self.arg_params, self.aux_params)


if __name__ == '__main__':
    model = MajorModel(8, ('images', 'body', 'specters'), 30, 30, 10, (3, 48, 48), (3, 40, 40), 'emotions', 7,
                       '/home/mopkobka/NDL-Projects/mxnetprojects/FacialEmotionRecognition/mxNets/face_emotions_l2softmax_8norm',
                       115, 'data', 'feat_emo')

