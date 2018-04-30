"""
NeurodataLab LLC 04.04.2018
Created by Andrey Belyaev
"""
import mxnet as mx
from data_aggregators import MajorDataAggregator


class MajorDataIterator(mx.io.DataIter):
    def __init__(self, data_path, num_images=None, num_bodies=None, num_specters=None,  shuffle=True,
                 data_names=('images', 'body', 'specters'), label_names=('emotions',),
                 batch_size=16, num_batches=100, labels=None, label_weights=None, anno_reg=False, for_train=True):
        mx.io.DataIter.__init__(self, batch_size=batch_size)

        self.data_path = data_path

        assert (num_images is not None) == ('images' in data_names)
        assert (num_specters is not None) == ('specters' in data_names)
        assert (num_bodies is not None) == ('body' in data_names)
        self.num_images, self.num_specters, self.num_bodies = num_images, num_specters, num_bodies
        self.shuffle = shuffle

        self.data_names, self.label_names = data_names, label_names
        self.num_batches = num_batches
        self.labels, self.label_weights = labels, label_weights
        self.anno_reg, self.for_train = anno_reg, for_train

        self.cur_batch = 0
        self.data_aggregator = MajorDataAggregator(self.data_path, self.num_images, self.num_bodies,
                                                   self.num_specters, self.shuffle, self.labels,
                                                   self.label_weights, self.anno_reg, self.for_train)

        # Process single data element
        single_data, single_label = self.data_aggregator.next()
        self.data_shapes = [(self.batch_size,) + s_data.shape for s_data in single_data]
        self.label_shapes = [(self.batch_size,) + s_label.shape for s_label in single_label]

        self.provide_data = zip(self.data_names, self.data_shapes)
        self.provide_label = zip(self.label_names, self.label_shapes)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_batch = 0

    def next(self):
        self.cur_batch += 1
        if self.cur_batch < self.num_batches:
            data, labels = [[] for _ in self.data_names], [[] for _ in self.label_names]
            for _ in range(self.batch_size):
                d, l = self.data_aggregator.next()
                for new_data, cur_data in zip(d, data):
                    cur_data.append(new_data)
                for new_label, cur_label in zip(l, labels):
                    cur_label.append(new_label)
            return mx.io.DataBatch(data=list(map(mx.nd.array, data)), label=list(map(mx.nd.array, labels)))
        else:
            raise StopIteration


if __name__ == '__main__':
    from time import time
    data_iter = MajorDataIterator('/media/mopkobka/1Tb SSD/DATA/EmotionMinerCorpus/VERSION_21_03/preprocessed_data/preprocessed_data_labels8-10-11-12-21_multi_neutral_augment_ims-1_specs8/test',
                                  30, 30, 10)

    start = time()
    for _ in range(99):
        print data_iter.next()
    print time() - start
