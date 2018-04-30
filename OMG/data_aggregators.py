"""
NeurodataLab LLC 04.04.2018
Created by Andrey Belyaev
"""
import numpy as np
import os
import os.path as osp

# thresholds = np.array([0.95, 0.4, 0.5, 0.6, 0.7])  # 21_03, 1 bucket
# thresholds = np.array((0.79, 0.79, 0.59, 0.4, 0.65))  # 21_03, 2 bucket
# label_nums = [0, 1, 2, 3, 4]

# 21_03, 1 bucket
# label_nums = [8, 10, 11, 12, 21]
# thresholds = np.array([0.9, 0.45, 0.5, 0.6, 0.6])
# use_val_arr = False

# 21_03, 2 bucket
# label_nums = [2, 3, 9, 13, 14]
# thresholds = np.array((0.79, 0.79, 0.59, 0.4, 0.65))
# thresholds = np.array([1, 1, 0.79, 0.79, 1, 1, 1, 1, 1, 0.59, 1, 1, 1, 0.5, 0.65, 1, 1, 1, 1, 1, 1, 1])

# OMG
label_nums = [0, 1, 2, 3, 4, 5, 6]
thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
use_val_arr = True

# label_nums = [8, 10, 11, 12, 21]
# thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # OMG


class FaceImagesAndBodyDataAggregator:
    def __init__(self, data_paths, num_samples, shuffle=True, use_images=True, use_bodies=True):
        self.data_paths = data_paths
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.use_images, self.use_bodies = use_images, use_bodies

        assert not (use_images ^ use_bodies)

        self.data_paths = np.array(self.data_paths, dtype=str)[np.random.permutation(len(self.data_paths))]
        self.cur = 0

    def reset(self, idxs=None):
        if idxs is None:
            idxs = np.random.permutation(len(self.data_paths))
        self.data_paths = self.data_paths[idxs]
        self.cur = 0

    def next(self):
        if not self.use_images and not self.use_bodies:
            return [None, None]

        if self.cur >= len(self.data_paths):
            raise UserWarning('reset')

        cur_dir = self.data_paths[self.cur]
        images = np.load(osp.join(cur_dir, 'images.npy'))
        bodies = np.load(osp.join(cur_dir, 'body.npy'))

        if len(images) < 1 or len(bodies) < 1:
            self.cur += 1
            raise UserWarning('bad image body')

        if self.shuffle:
            idxs = sorted(np.random.permutation(min(len(images), len(bodies)))[:self.num_samples].tolist())
            images = images[idxs]
            bodies = bodies[idxs]
        else:
            start = np.random.randint(0, max(1, min(len(images), len(bodies)) - self.num_samples))
            images = images[start: start + self.num_samples]
            bodies = bodies[start: start + self.num_samples]

        if len(images) < self.num_samples or len(bodies) < self.num_samples:
            idxs = np.linspace(0, min(len(images), len(bodies)) - 1, self.num_samples).astype(int).tolist()
            images = images[idxs]
            bodies = bodies[idxs]

        if np.random.random() > 0.5:
            images = images[:, :, ::-1, :]
        self.cur += 1

        # import cv2
        # for image in images:
        #     cv2.imshow('q', image)
        #     if cv2.waitKey(0) & 0xff == ord('q'):
        #         exit(0)

        label_reg = np.load(osp.join(cur_dir, 'full_label.npy'))
        if any(map(np.isnan, label_reg)):
            raise UserWarning('bad image body')
        label = (label_reg[label_nums] > thresholds).astype(int)
        label = np.concatenate([label, [1 if label.sum() < 1 else 0]])
        arr, val = label_reg[-2:] if use_val_arr else (0, 0)
        return [images.transpose((0, 3, 1, 2)), bodies.reshape((self.num_samples, -1))], \
               [label, label_reg[label_nums], arr, val]


class SpectersDataAggregator:
    def __init__(self, data_paths, num_specters):
        self.data_paths = data_paths
        self.num_specters = num_specters

        self.data_paths = np.array(self.data_paths, dtype=str)[np.random.permutation(len(self.data_paths))]
        self.cur = 0

    def reset(self, idxs=None):
        if idxs is None:
            idxs = np.random.permutation(len(self.data_paths))
        self.data_paths = self.data_paths[idxs]
        self.cur = 0

    def next(self):
        if self.num_specters is None:
            return [None]

        if self.cur >= len(self.data_paths):
            raise UserWarning('reset')

        cur_dir = self.data_paths[self.cur]
        specters = np.load(osp.join(cur_dir, 'specters.npy'))

        if len(specters) >= self.num_specters:
            start = np.random.randint(0, len(specters) - self.num_specters + 1)
            specters = specters[start: start + self.num_specters]
        else:
            specters = specters[np.linspace(0, len(specters) - 1, self.num_specters).astype(int).tolist()]

        self.cur += 1
        specters = specters.transpose((0, 3, 1, 2))
        return [specters], None
        # return [np.random.random(specters.shape)], None


class LabelDataAggregator:
    def __init__(self, data_path, num_images, num_bodies, num_specters, shuffle=True, for_train=True):
        self.data_path = data_path
        self.all_data_paths = map(lambda p: osp.join(self.data_path, p), os.listdir(self.data_path))
        self.for_train = for_train

        self.aggregators = [
            FaceImagesAndBodyDataAggregator(self.all_data_paths, num_images, shuffle,
                                            use_images=num_images is not None, use_bodies=num_bodies is not None),
            SpectersDataAggregator(self.all_data_paths, num_specters)
        ]

    def reset(self):
        idxs = np.random.permutation(len(self.all_data_paths))
        for aggregator in self.aggregators:
            aggregator.reset(idxs)

    def next(self):
        cur_data = []
        cur_label = None
        cur_anno_label = None
        cur_arr, cur_val = None, None
        try:
            for aggregator in self.aggregators:
                d, l = aggregator.next()
                cur_data.extend(d)
                if l is not None:
                    cur_label, cur_anno_label, cur_arr, cur_val = l
        except UserWarning as e:
            msg = str(e)
            if msg == 'reset':
                if self.for_train:
                    self.reset()
                else:
                    raise StopIteration
            elif 'bad' in msg:
                if 'image' in msg or 'body' in msg:
                    self.aggregators[1].cur = self.aggregators[0].cur
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            return self.next()

        return filter(lambda d: d is not None, cur_data), cur_label, cur_anno_label, cur_arr, cur_val


class MajorDataAggregator:
    def __init__(self, data_path, num_images, num_bodies, num_specters, shuffle=True, labels=None,
                 labels_weights=None, anno_reg=False, for_train=True):
        self.data_path = data_path
        self.num_images = num_images
        self.num_specters = num_specters
        self.num_bodies = num_bodies
        self.shuffle = shuffle
        self.for_train = for_train
        self.anno_reg = anno_reg

        data_paths = filter(lambda p: labels is None or int(p.split('_')[-1]) in labels, os.listdir(self.data_path))

        self.data = {int(l.split('_')[-1]): LabelDataAggregator(osp.join(self.data_path, l), self.num_images,
                                                                self.num_bodies, self.num_specters,
                                                                self.shuffle, self.for_train)
                     for l in data_paths}
        self.label_nums = sorted(self.data.keys())
        self.labels = map(lambda l: np.array([1 if i == l else 0 for i in range(len(self.data))]),
                          range(len(self.label_nums)))
        self.cur_label_num = 0

        print 'Labels count:'
        for key, agg in self.data.items():
            print '%d: #%d' % (key, len(agg.all_data_paths))
        print

        if labels_weights is not None:
            self.label_nums = reduce(lambda a, b: a + b, [[l] * i for l, i in zip(self.label_nums, labels_weights)], [])
            self.labels = {k: v for k, v in zip(sorted(list(set(self.label_nums))), self.labels)}

    def next(self):
        if self.for_train:
            lbl_num = self.label_nums[self.cur_label_num]
            cur_data, cur_label, anno_reg_label, arr, val = self.data[lbl_num].next()
            # cur_label = self.labels[lbl_num]
            self.cur_label_num = (self.cur_label_num + 1) % len(self.label_nums)
            return [cur_data,
                    [cur_label[sorted(np.unique(self.label_nums).tolist())]] +
                    [np.array([arr]), np.array([val])]]
        else:
            return self.next_val()

    def next_val(self):
        if self.cur_label_num >= len(self.labels):
            for d in self.data.values():
                d.reset()
            self.cur_label_num = 0
            raise StopIteration
        cur_data_iter = self.data[self.label_nums[self.cur_label_num]]
        try:
            cur_data, cur_label, anno_reg_label, arr, val = cur_data_iter.next()
            # cur_label = self.labels[self.cur_label_num]
            return [cur_data,
                    [cur_label[self.label_nums]] +
                    [np.array([arr]), np.array([val])]]
        except StopIteration:
            self.cur_label_num = self.cur_label_num + 1
            return self.next_val()
