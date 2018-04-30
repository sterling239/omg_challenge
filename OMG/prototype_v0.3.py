"""
NeurodataLab LLC 04.04.2018
Created by Andrey Belyaev
"""
import logging
import mxnet as mx
import os.path as osp
from common_utils import load_config
from data_iterators import MajorDataIterator
from metrics import *
from models import MajorModel

logging.getLogger().setLevel(logging.DEBUG)
config = load_config()


def train_v1(data_path, name='', labels=None, pretrained=None):
    data_names = ('images', 'body', 'specters')
    label_names = ('emotions', 'arr', 'val')

    print 'Initialize train iter'
    train_iter = MajorDataIterator(osp.join(data_path, 'train'), num_images, num_bodies, num_specters, shuffle,
                                   data_names, label_names, batch_size, num_train_batches, labels,
                                   label_weights=lbls_weights, anno_reg=True, for_train=True)

    print 'Initialize test iter'
    test_iter = MajorDataIterator(osp.join(data_path, 'test'), num_images, num_bodies, num_specters, False,
                                  data_names, label_names, batch_size, num_test_batches, labels,
                                  anno_reg=True, for_train=False)

    print train_iter.provide_data
    print train_iter.provide_label
    print 'Train batches: %d, test batches: %d' % (num_train_batches, num_test_batches)

    print 'Initialize model'
    m = MajorModel(batch_size, data_names, num_images, num_bodies, num_specters, (3, 112, 112), (3, 40, 40),
                   label_names, num_cls,
                   # '/home/mopkobka/NDL-Projects/mxnetprojects/FacialEmotionRecognition/mxNets/face_emotions_48x48',
                   # 0, 'data', 'feat_emo',
                   # '/home/mopkobka/NDL-Projects/mxnetprojects/FacialEmotionRecognition/mxNets/face_emotions_48x48_tree_0',
                   # 0, '_tree_0_data', 'feat__tree_0_emo',
                   '/home/mopkobka/NDL-Projects/mxnetprojects/FacialEmotionRecognition/affect_net/mxNets/face_emotions_affect_net_val_ar_112x112_tree_0',
                   0, '_tree_0_data', 'feat__tree_0_emo',
                   # '/home/mopkobka/NDL-Projects/PretrainedNets/resnet-50_tree_0',
                   # 0, '_tree_0_data', '_tree_0_pool1',
                   '/home/mopkobka/NDL-Projects/PretrainedNets/biometry_256_x_norm_eqhist_nag',
                   0, 'data', 'fc_2',
                   # '/home/mopkobka/NDL-Projects/emotion-recognition-prototype/andrew/v5_3d/mxNets/video_actions_3d',
                   # 18, 'reshape0', '3d_conv_pool3',
                   context=[mx.gpu(0), mx.gpu(1)], norm_alpha=norm_alpha)

    model, arg_params, aux_params = m.train

    if pretrained is not None:
        _, arg_params, aux_params = mx.model.load_checkpoint(pretrained[0], pretrained[1])
    for params in (arg_params, aux_params):
        for k in params.keys():
            if pretrained is not None:
                if 'end_fc' in k:
                    params.pop(k)
            if 'softmax' in k or 'fullyconnected' in k:
                try:
                    params.pop(k)
                except KeyError:
                    pass

    metrics = []
    metrics += [AveragePrecision(num_output=0, freq=5000, name='end_avg_prec')]
    metrics += [F1Metric(num_output=0, name='f1')]
    metrics += [RMSEMetric(num_output=1, name='arr_rmse')]
    metrics += [RMSEMetric(num_output=2, name='val_rmse')]
    # metrics += [AveragePrecision(num_output=1, freq=5000, name='im_avg_prec')]
    # metrics += [AveragePrecision(num_output=2, freq=5000, name='im3d_avg_prec')]
    # metrics += [RegressionAverageAccuracy(outputs=range(3, 3 + num_cls))]
    # metrics += [RegressionAveragePrecision(outputs=range(3, 3 + num_cls))]
    # metrics += [AnnoRegressionMae(label_num=-1, num_outputs=range(4 + num_cls, 4 + 2 * num_cls))]
    # metrics += [RegressionAccuracy(num_output=4 + i, label_num=i) for i in range(num_cls)]
    if norm_alpha is not None:
        metrics += [FeatNormMetric()]
    # metrics += [PrintMetric(num_batches=num_cls * num_repeats, frequent=1)]

    lr_scheduler = mx.lr_scheduler.FactorScheduler(int(num_train_batches * 25), factor=0.3, stop_factor_lr=1e-5)
    optimizer_params = {
        'lr_scheduler': lr_scheduler,
        'learning_rate': 0.002,
        'wd': 1e-5,
        'momentum': 0.95
    }

    model.fit(train_data=train_iter, eval_data=test_iter, eval_metric=mx.metric.CompositeEvalMetric(metrics),
              batch_end_callback=[mx.callback.Speedometer(batch_size, frequent=50, auto_reset=False)],
              epoch_end_callback=[mx.callback.do_time_checkpoint('big_net_3d_new' + name)],
              initializer=mx.init.Xavier('uniform'),
              optimizer='nag', optimizer_params=optimizer_params,
              arg_params=arg_params, aux_params=aux_params, allow_missing=True,
              num_epoch=1000)


if __name__ == '__main__':
    # 0: 7416 1: 1053 2: 1014
    # 3: 1933 4: 1905 5: 7266
    # bucket_num = 1
    # lbls = (0, 1, 2, 3, 4)
    # lbls_weights = (1, 1, 1, 1, 1)
    # num_repeats = 2
    # num_cls = len(lbls) if lbls is not None else 6
    # batch_size = num_cls * num_repeats
    # num_images, num_bodies, num_specters = 40, 40, 10
    # shuffle = True
    # norm_alpha = None
    # num_train_batches, num_test_batches = 900 // num_repeats, 1e8 // num_repeats
    # preprocessed_data_name = 'new_preprocessed_data_labels8-10-11-12-21_multi_new-ims-1(112, 112)_new-specs9'
    # train_v1(osp.join(config.DATA.preprocessed_path, preprocessed_data_name),
    #          name='_21_03__lbs_' + reduce(lambda a, b: a + str(b) + '-', lbls, '')[:-1] +
    #          '_images-%d_bodies-%d_specters-%d_bucket-%d' % (num_images, num_bodies, num_specters, bucket_num),
    #          labels=lbls, pretrained=None)
             # ('/SSD2/mxNets/big_net_3d_new_OMG__lbs_0-1-2-3-4-5_images-40_bodies-40_specters-9/27-04-2018_17-57-34/model', 32)
             # ('mxNets/big_net_3d_OMG__lbs_0-1-2-3-4-5_images-40_bodies-40_specters-10', 35)
             # ('mxNets/big_net_21_03__lbs_0-1-2-3-4_images-40_bodies-40_specters-10', 13)
             # ('goodNets/big_net_OMG__lbs_0-1-2-3-4-5_images-60_bodies-60_specters-12', 20))

    # 0: 535 1: 238 2: 100
    # 3: 1311 4: 1634 5: 619
    lbls = (0, 1, 2, 3, 4, 5, 6)
    lbls_weights = (1, 1, 1, 1, 1, 1, 1)
    num_repeats = 2
    num_cls = len(lbls) if lbls is not None else 6
    # batch_size = num_cls * num_repeats
    batch_size = 12
    num_images, num_bodies, num_specters = 40, 40, 9
    shuffle = True
    norm_alpha = None
    num_train_batches, num_test_batches = 300 // num_repeats, 1e8 // num_repeats
    # preprocessed_data_name = 'new_preprocessed_data_labels0-1-2-3-4-5-6_multi_new-ims-1(48, 48)_new-specs10'
    train_v1('/SSD2/DATA/VERSION_OMG/new_preprocessed_data_labels0-1-2-3-4-5-6_multi_new-ims-1(112, 112)_new-specs9',
             # osp.join('/SSD2/DATA/VERSION_OMG', preprocessed_data_name),
             name='_OMG__lbs_' + reduce(lambda a, b: a + str(b) + '-', lbls, '')[:-1] +
                  '_images-%d_bodies-%d_specters-%d' % (num_images, num_bodies, num_specters),
             labels=lbls, pretrained=('/SSD2/mxNets/big_net_3d_new_21_03__lbs_0-1-2-3-4_images-40_bodies-40_specters-10_bucket-1/28-04-2018_17-03-47/model', 30))
