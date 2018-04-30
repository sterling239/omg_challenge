"""
NeurodataLab LLC 03.04.2018
Created by Andrey Belyaev
"""
import cv2
import dlib
import json
import numpy as np
import os
import os.path as osp
from common_utils import load_config
from PIL import Image, ImageDraw
# from scipy.ndimage import convolve
from shutil import copyfile
from skimage.exposure import equalize_hist
from tqdm import tqdm

config = load_config()
conv_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=float)
conv_kernel = conv_kernel / conv_kernel.sum()

predictor = dlib.shape_predictor(
    '/home/mopkobka/NDL-Projects/autonomous_fulltracker/FaceTracker/data/shape_predictor.dat')


def dlib_to_np(shape):
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=int)


def process_specter(specter_path, specters_samples_count):
    def prepare_mini_spec(spec):
        data = np.log(1 + spec * 10000) / 15.
        if data.max() < 1e-5:
            return np.repeat(data[..., np.newaxis], 3, axis=-1)
        try:
            data = data
            data_norm = (data - data.min()) / (data.max() - data.min())
            data_hist = equalize_hist(data)
            return np.concatenate([data[..., np.newaxis],
                                   data_norm[..., np.newaxis],
                                   data_hist[..., np.newaxis]], axis=-1)
        except:
            return np.repeat(data[..., np.newaxis], 3, axis=-1)
        # data = np.log(spec + 1e-8)
        #
        # try:
        #     data[data < -5] = -5
        #     data += np.random.rand(*data.shape) * 1e-8
        #     data = (data - data.min()) / (data.max() - data.min())
        #     data_hist = equalize_hist(data, nbins=128)
        #     data_conv = convolve(data, conv_kernel)
        #     data_conv = data_conv / data_conv.max()
        #
        #     return np.concatenate([data[..., np.newaxis],
        #                            data_hist[..., np.newaxis],
        #                            data_conv[..., np.newaxis]], axis=-1)
        # except UserWarning:
        #     return np.repeat(data[..., np.newaxis], 3, axis=-1)

    specter = np.load(specter_path)
    spec_step = min(config.DATA.features.spectrogram.step, specter.shape[0] // (specters_samples_count + 1) - 1)
    if spec_step < 1:
        raise UserWarning
    mini_specs = np.array([specter[s: s + config.DATA.features.spectrogram.target_shape[0], :40] for s
                           in range(0, specter.shape[0] - config.DATA.features.spectrogram.target_shape[0], spec_step)])
    mini_specs = map(prepare_mini_spec, mini_specs)
    return np.array(mini_specs)


def process_image(image_path):
    image = cv2.imread(image_path)

    face_rect = dlib.rectangle(0, 0, image.shape[0] - 1, image.shape[1] - 1)
    shape = dlib_to_np(predictor(image, face_rect))
    face_oval = Image.new("L", image.shape[:2][::-1], 0)
    idxs = list(range(17)) + [26, 24, 19, 17]
    ImageDraw.Draw(face_oval).polygon(zip(shape[idxs, 0], shape[idxs, 1]), outline=1, fill=1)
    mask_oval = np.repeat(np.array(face_oval, dtype=float)[..., np.newaxis], 3, axis=-1)
    image[np.logical_not(mask_oval)] = 0

    X, Y, _ = np.where(mask_oval)
    image = image[X.min(): X.max(), Y.min(): Y.max()]

    h, w = image.shape[:2]
    if w > h:
        diff_2 = (w - h) // 2
        image = cv2.copyMakeBorder(image, diff_2, diff_2, 0, 0, cv2.BORDER_CONSTANT, 0)
    elif h > w:
        diff_2 = (h - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, diff_2, diff_2, cv2.BORDER_CONSTANT, 0)

    image = cv2.resize(image, tuple(config.DATA.features.f_img.target_shape))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.max() > 1:
        image = image.astype(float) / 255.
    return image


def process_body(body_path):
    with open(body_path, 'r') as f:
        body_info = {k: v for k, v in json.load(f).items() if v is not None}
    if len(body_info) == 0:
        return None
    body_keys = sorted(body_info.keys(), key=lambda k: int(k))
    body_info[body_keys[0]] = [kp[:2] if kp is not None else [0, 0] for kp in body_info[body_keys[0]]]
    prev_body = body_info[body_keys[0]]
    for n, key in enumerate(body_keys[1:]):
        prev_body = [prev if cur is None else cur[:2] for prev, cur in zip(prev_body, body_info[key])]
        body_info[key] = prev_body

    body_info = {int(k): v for k, v in body_info.items()}
    for key in range(min(body_info.keys()), max(body_info.keys())):
        if key not in body_info:
            body_info[key] = body_info[key - 1]

    return np.array([body_info[k] for k in sorted(body_info.keys())])


def compose_data(data_keys, data_info, out_path, images_samples_count, specters_samples_count, augment_data=False):
    if not osp.exists(out_path):
        os.mkdir(out_path)
    for label_num, keys_per_label in enumerate(data_keys):
        label_out_path = osp.join(out_path, 'label_%d' % label_num)
        if not osp.exists(label_out_path):
            os.mkdir(label_out_path)
        cur_num = 0
        for key in tqdm(keys_per_label):
            try:
            # if True:
                cur_path = osp.join(config.DATA.data_path, key)
                images = np.array([process_image(osp.join(cur_path, 'f_imgs', p))
                                   for p in sorted(os.listdir(osp.join(cur_path, 'f_imgs')),
                                                   key=lambda p: int(p.split('.')[0]))])
                specters = process_specter(osp.join(cur_path, 'spectrogram.npy'), specters_samples_count)
                body = process_body(osp.join(cur_path, 'body.json'))
                cur_out_path = osp.join(label_out_path, str(cur_num))
                if not osp.exists(cur_out_path):
                    os.mkdir(cur_out_path)

                np.save(osp.join(cur_out_path, 'images.npy'), images)
                np.save(osp.join(cur_out_path, 'specters.npy'), specters)
                copyfile(osp.join(cur_path, 'spectrogram.npy'), osp.join(cur_out_path, 'spectrogram.npy'))
                np.save(osp.join(cur_out_path, 'body.npy'), body)
                np.save(osp.join(cur_out_path, 'full_label.npy'), np.asarray(data_info[key]))

                cur_num += 1

                if augment_data and len(images) > 0:
                    augment_out_path = osp.join(label_out_path, str(cur_num))
                    if not osp.exists(augment_out_path):
                        os.mkdir(augment_out_path)
                    np.save(osp.join(augment_out_path, 'images.npy'), images[:, :, ::-1, :])
                    copyfile(osp.join(cur_out_path, 'specters.npy'), osp.join(augment_out_path, 'specters.npy'))
                    copyfile(osp.join(cur_path, 'spectrogram.npy'), osp.join(augment_out_path, 'spectrogram.npy'))
                    copyfile(osp.join(cur_out_path, 'body.npy'), osp.join(augment_out_path, 'body.npy'))
                    copyfile(osp.join(cur_out_path, 'full_label.npy'), osp.join(augment_out_path, 'full_label.npy'))

                    cur_num += 1

            except:
                continue


def preprocess_data(train_info_path, test_info_path, out_path, labels=(0, 1), thresholds=(0.5, 0.5),
                    single_labels=False, with_neutral=False, augment_data=False,
                    images_samples_count=60, specters_samples_count=12):
    labels = list(labels)
    if not osp.exists(out_path):
        os.mkdir(out_path)

    with open(train_info_path, 'r') as f:
        train_info = {k: np.array(v) for k, v in json.load(f).items() if not any(map(np.isnan, v))}

    with open(test_info_path, 'r') as f:
        test_info = {k: np.array(v) for k, v in json.load(f).items() if not any(map(np.isnan, v))}

    train_data_per_label, test_data_per_label = [], []
    for label_num, th in zip(range(len(labels)), thresholds):
        for data_per_label, info in zip((train_data_per_label, test_data_per_label), (train_info, test_info)):
            data = []
            for key, val in info.items():
                val = val[labels]
                if val[label_num] > th:
                    if single_labels:
                        if sum([int(val[n] > thresholds[n]) for n in range(len(labels)) if n != label_num]) > 0:
                            continue
                    data.append(key)
            data_per_label.append(data)

    if with_neutral:
        for data_per_label, info in zip((train_data_per_label, test_data_per_label), (train_info, test_info)):
            data = []
            for key, val in info.items():
                # if len(np.where(val > thresholds)[0]) == 0:
                if val[labels].sum() <= 0.1:
                # print np.any(val[2:] > 0.4)
                # if val[1] >= 0.|8 and val[3] > 0.8 and not np.any(val[labels] > 0.5):
                    data.append(key)
            data = [data[idx] for idx in np.random.permutation(len(data))[:max(map(len, data_per_label))]]
            data_per_label.append(data)

    print 'Train count %d, test count %d' % (len(train_info), len(test_info))
    print map(lambda d: len(d) * (1 + int(augment_data)), train_data_per_label)
    print map(lambda d: len(d) * (1 + int(augment_data)), test_data_per_label)

    with open(osp.join(out_path, 'train.json'), 'w') as f:
        d = {i: list(map(lambda p: osp.join(config.DATA.data_path, p), v)) for i, v in enumerate(train_data_per_label)}
        json.dump(d, f, indent=2)
    with open(osp.join(out_path, 'test.json'), 'w') as f:
        d = {i: list(map(lambda p: osp.join(config.DATA.data_path, p), v)) for i, v in enumerate(test_data_per_label)}
        json.dump(d, f, indent=2)
    # exit(0)

    compose_data(train_data_per_label, train_info, osp.join(out_path, 'train'),
                 images_samples_count, specters_samples_count, augment_data)
    compose_data(test_data_per_label, test_info, osp.join(out_path, 'test'),
                 images_samples_count, specters_samples_count, augment_data)


if __name__ == '__main__':
    # lbls = (0, 1, 2, 3, 4, 5, 6)
    # ths = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    # single_lbl = False
    # neutral = False
    # augment = False
    # num_images = -1
    # num_specs = 9

    lbls = (8, 10, 11, 12, 21)
    ths = (0.9, 0.45, 0.5, 0.6, 0.6)
    # lbls = (2, 3, 9, 13, 14)
    # ths = (0.79, 0.79, 0.59, 0.4, 0.65)
    # lbls = range(22)
    # ths = np.ones(22) * 0.39
    single_lbl = False
    neutral = False
    augment = False
    num_images = -1
    num_specs = 9
    name = 'new_preprocessed_data_labels%s_%s%s%s_new-ims%d%s_new-specs%d' % (
        reduce(lambda a, b: a + str(b) + '-', lbls, '')[:-1],
        'single' if single_lbl else 'multi', '_neutral' if neutral else '', '_augment' if augment else '',
        num_images, str(tuple(config.DATA.features.f_img.target_shape)), num_specs
    )

    preprocess_data(config.DATA.train_info,
                    config.DATA.test_info,
                    osp.join(config.DATA.preprocessed_path, name),
                    labels=lbls, thresholds=ths, single_labels=single_lbl, with_neutral=neutral, augment_data=augment,
                    images_samples_count=num_images, specters_samples_count=num_specs)
