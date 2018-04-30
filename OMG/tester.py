"""
NeurodataLab LLC 12.04.2018
Created by Andrey Belyaev
"""
import cv2
import dlib
import json
import mxnet as mx
import numpy as np
import os
import os.path as osp
from common_utils import load_config, write_wave
from PIL import Image, ImageDraw
from scipy.io.wavfile import read as read_wave
from scipy.ndimage import convolve
from shutil import rmtree
from skimage.exposure import equalize_hist
from sklearn.metrics import confusion_matrix
from subprocess import call
from tqdm import tqdm

dev_null = open('/dev/null', 'w')
config = load_config()
conv_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=float)
conv_kernel = conv_kernel / conv_kernel.sum()
thresholds = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
# thresholds = np.array([0.75] * 5)
emotion_buckets = ('Happiness', 'Disgust', 'Surprise', 'Anger', 'Sadness')
# emotion_sizes = (0.3, 0.4, 0.4, 0.5, 0.4)

predictor = dlib.shape_predictor(
    '/home/mopkobka/NDL-Projects/autonomous_fulltracker/FaceTracker/data/shape_predictor.dat')


def dlib_to_np(shape):
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=int)


class EmotionRecognitionTester:
    def __init__(self, data_path, pretrained_models, out_layers, data_names, data_shapes, label_names, label_shapes,
                 batch_size=2, init_nets=True, in_fragment_step=4, context=(mx.gpu(1), mx.gpu(0))):
        self.data_path = data_path
        self.batch_size, self.ctx = batch_size, context
        self.out_layers = list(out_layers)
        self.in_fragment_step = in_fragment_step
        self.models = []
        self.data_info = {name: shape for name, shape in zip(data_names, data_shapes)}
        if init_nets:
            for path, epoch in pretrained_models:
                sym, arg_params, aux_params = mx.model.load_checkpoint(path, epoch)
                model = mx.mod.Module(sym, data_names=data_names, label_names=label_names, context=self.ctx)
                model.bind(data_shapes=zip(data_names, data_shapes), label_shapes=zip(label_names, label_shapes),
                           for_training=False)
                model.init_params(arg_params=arg_params, aux_params=aux_params)
                self.models.append(model)

            # zero forward
            zero_batch = [mx.nd.zeros(shape) for shape in data_shapes]
            for model in self.models:
                model.forward(mx.io.DataBatch(data=zero_batch), is_train=False)
        else:
            self.models = pretrained_models

    def process_batch(self, im_batch, b_batch, s_batch, results, end_sample=None):
        for m, model in enumerate(self.models):
            model.forward(mx.io.DataBatch(data=map(mx.nd.array, (im_batch, b_batch, s_batch))), is_train=False)
            preds = model.get_outputs()
            for n, layer in enumerate(self.out_layers):
                results[n][m].extend(preds[layer].asnumpy()[:end_sample].tolist())

    @staticmethod
    def get_images(images_path):
        images = []
        for im_path in sorted(os.listdir(images_path), key=lambda p: int(p.split('.')[0])):
            image = cv2.imread(osp.join(images_path, im_path))
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

            # cv2.imshow('1', image)
            # cv2.waitKey(0)

            image = cv2.resize(image, tuple(config.DATA.features.f_img.target_shape))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.max() > 1:
                image = image.astype(float) / 255.
            images.append(image)
        return np.array(images)

    @staticmethod
    def get_bodies(body_path):
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

    @staticmethod
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

    def get_specters(self, specter_path, specters_samples_count):

        specter = np.load(specter_path)
        spec_step = min(config.DATA.features_resnet.spectrogram.step, specter.shape[0] // (specters_samples_count + 1) - 1)
        if spec_step < 1:
            raise UserWarning
        mini_specs = np.array([specter[s: s + config.DATA.features_resnet.spectrogram.target_shape[0], :40] for s
                               in range(0, specter.shape[0] - config.DATA.features_resnet.spectrogram.target_shape[0],
                                        spec_step)])
        mini_specs = map(self.prepare_mini_spec, mini_specs)
        return np.array(mini_specs)

    def process_fragment(self, video_name, fragment_num):
        fragment_path = osp.join(self.data_path, video_name, 'fragment_%d' % fragment_num)
        fragment_files = os.listdir(fragment_path)
        assert 'spectrogram.npy' in fragment_files
        assert 'body.json' in fragment_files
        assert 'f_imgs' in fragment_files

        images = self.get_images(osp.join(fragment_path, 'f_imgs'))
        if len(images) < self.data_info['images'][1]:
            if len(images) >= self.data_info['images'][1] // 2:
                images = images[np.linspace(0, len(images) - 1, self.data_info['images'][1]).astype(int)]
            else:
                return []
        bodies = self.get_bodies(osp.join(fragment_path, 'body.json'))
        if len(bodies) == 0:
            return []
        if len(bodies) > len(images):
            bodies = bodies[:len(images)]
        elif len(bodies) < len(images):
            bodies = bodies[np.linspace(0, len(bodies) - 1, len(images)).astype(int)]

        specter = np.load(osp.join(fragment_path, 'spectrogram.npy'))
        spec_lag = float(specter.shape[0]) / images.shape[0] - 1

        im_batch, b_batch, s_batch = [], [], []
        results = [[[] for __ in self.models] for _ in self.out_layers]
        for im_start in range(0, len(images) - self.data_info['images'][1] + 1, self.in_fragment_step):
            images_to_batch = images[im_start: im_start + self.data_info['images'][1]]
            bodies_to_batch = bodies[im_start: im_start + self.data_info['body'][1]]

            specters_to_batch = []
            for i in range(self.data_info['specters'][1]):
                spec_start = int(im_start * spec_lag + i * config.DATA.features.spectrogram.step)
                mini_spec = specter[spec_start: spec_start + config.DATA.features.spectrogram.target_shape[0], :40]
                if len(specters_to_batch) > 0 and len(mini_spec) == 0:
                    specters_to_batch.append(specters_to_batch[-1])
                elif len(specters_to_batch) > 0 and mini_spec.shape != specters_to_batch[-1].shape[:2]:
                    mini_spec = cv2.resize(mini_spec, tuple(specters_to_batch[-1].shape[:2]))
                    specters_to_batch.append(self.prepare_mini_spec(mini_spec))
                else:
                    specters_to_batch.append(self.prepare_mini_spec(mini_spec))

            im_batch.append(images_to_batch.transpose((0, 3, 1, 2)))
            b_batch.append(bodies_to_batch.reshape((-1, 36)))
            s_batch.append(np.array(specters_to_batch).transpose((0, 3, 1, 2)))

            if len(im_batch) == len(b_batch) == len(s_batch) == self.batch_size:
                self.process_batch(im_batch, b_batch, s_batch, results)
                im_batch, b_batch, s_batch = [], [], []

        if len(im_batch) > 0:
            end_sample = len(im_batch)
            for _ in range(len(im_batch), self.batch_size):
                im_batch.append(np.zeros_like(im_batch[0]))
                b_batch.append(np.zeros_like(b_batch[0]))
                s_batch.append(np.zeros_like(s_batch[0]))
            self.process_batch(im_batch, b_batch, s_batch, results, end_sample)

        return results

    def process_video(self, video_name, out_dir, with_tqdm=True):
        if not osp.exists(out_dir):
            os.mkdir(out_dir)
        cycle_processor = tqdm if with_tqdm else (lambda x: x)

        video_path = osp.join(self.data_path, video_name)
        for fragment in cycle_processor(os.listdir(video_path)):
            out_frag_path = osp.join(out_dir, '%s.json' % fragment)
            if osp.exists(out_frag_path):
                continue
            try:
                frag_results = self.process_fragment(video_name, int(fragment.split('_')[-1]))
                with open(out_frag_path, 'w') as f:
                    json.dump(frag_results, f, indent=2)
            except AssertionError as e:
                print video_name, fragment, e
                continue

    def process_data(self, out_dir, with_tqdm=True):
        if not osp.exists(out_dir):
            os.mkdir(out_dir)
        cycle_processor = tqdm if with_tqdm else (lambda x: x)

        for video_name in cycle_processor(os.listdir(self.data_path)):
            self.process_video(video_name, osp.join(out_dir, video_name), with_tqdm ^ True)

    def calculate_prediction(self, fragment_prediction):
        with open(fragment_prediction, 'r') as f:
            all_predictions = json.load(f)
        if len(all_predictions) == 0:  #  or len(all_predictions[0]) != len(self.models):
            return None
        predictions = [[None for __ in self.out_layers] for _ in self.models]
        for layer_num, layer in enumerate(all_predictions):
            for model_num, model in enumerate(layer):
                predictions[model_num][layer_num] = np.mean(all_predictions[layer_num][model_num], axis=0)

        mean_layer_predictions = [np.mean([predictions[m][layer_num] for m in range(len(self.models))], axis=0).tolist()
                                  for layer_num in range(len(self.out_layers))]

        return mean_layer_predictions

    def calculate_metrics(self, labels_path, results_path, label_classes, videos_to_calculate_info):
        with open(videos_to_calculate_info, 'r') as f:
            videos_to_calculate = list(set([key.split('/')[0] for key in json.load(f).keys()]))
        preprocessed_videos = filter(lambda p: p in videos_to_calculate, os.listdir(results_path))
        with open(labels_path, 'r') as f:
            labels = {k: v for k, v in json.load(f).items() if k in preprocessed_videos}

        total_sm_labels, total_lr_labels = [], []
        total_sm_preds, total_lr_preds = [], []
        total_anno_reg_labels, total_anno_reg_preds = [], []
        for video_name in tqdm(preprocessed_videos):
            for preprocessed_fragment_path in os.listdir(osp.join(results_path, video_name)):
                prediction = self.calculate_prediction(osp.join(results_path, video_name, preprocessed_fragment_path))
                frag_num = preprocessed_fragment_path.split('_')[-1].split('.')[0]
                if prediction is None or video_name not in labels or frag_num not in labels[video_name]:
                    continue
                label = labels[video_name][frag_num]
                label = np.array(label)[label_classes]
                if any(map(np.isnan, label)):
                    continue
                sm_pred = np.argsort(prediction[0])[::-1]
                lr_pred = np.array(prediction[1:6]).reshape(-1).argsort()[::-1]
                anno_reg_pred = np.array(prediction[6:]).reshape(-1)
                # print label, sm_pred, lr_pred
                good_cls = np.where(label >= thresholds)[0]
                if len(good_cls) > 1:
                    print label, good_cls, sm_pred, lr_pred
                if len(good_cls) == 0:
                    continue
                # good_cls = [np.argsort(label)[::-1][0]]

                sm_good_cls = sm_pred[:len(good_cls)]
                sm_good = set(good_cls) & set(sm_good_cls)
                total_sm_labels += list(sm_good) + list(set(good_cls) ^ sm_good)
                total_sm_preds += list(sm_good) + list(set(sm_good_cls) ^ sm_good)

                lr_good_cls = lr_pred[:len(good_cls)]
                lr_good = set(good_cls) & set(lr_good_cls)
                total_lr_labels += list(lr_good) + list(set(good_cls) ^ lr_good)
                total_lr_preds += list(lr_good) + list(set(lr_good_cls) ^ lr_good)

                anno_reg_good_cls = np.where(anno_reg_pred >= (thresholds - 0.1))[0]
                anno_reg_good = set(good_cls) & set(anno_reg_good_cls)
                total_anno_reg_labels += list(anno_reg_good)
                total_anno_reg_preds += list(anno_reg_good)

        print 'Total count', len(total_sm_preds)
        print

        sm_cm = confusion_matrix(total_sm_labels, total_sm_preds).astype(float)
        sm_cm = sm_cm / sm_cm.sum(axis=1)[:, np.newaxis]

        lr_cm = confusion_matrix(total_lr_labels, total_lr_preds).astype(float)
        lr_cm = lr_cm / lr_cm.sum(axis=1)[:, np.newaxis]

        anno_cm = confusion_matrix(total_anno_reg_labels, total_anno_reg_preds).astype(float)
        anno_cm = anno_cm / anno_cm.sum(axis=1)[:, np.newaxis]

        print 'Softmax', sm_cm.diagonal().mean()
        print sm_cm
        print
        print 'Regression', lr_cm.diagonal().mean()
        print lr_cm
        print
        print 'Anno regression', anno_cm.diagonal().mean()
        print anno_cm
        print

    # Assert 1 - softmax, 2..6 - regression, 3 models
    @staticmethod
    def get_from_results(results, frag_num):
        batch_num = frag_num // 4
        softmax_score = np.mean([results[0][model_num][batch_num] for model_num in (0, 1, 2)], axis=0)
        regression_scores = [np.mean([results[cls][model_num][batch_num] for model_num in (0, 1, 2)], axis=0)
                             for cls in range(1, 6)]
        models_results = [results[0][model_num][batch_num] + [results[cls][model_num][batch_num][0]
                                                               for cls in range(1, 6)]
                          for model_num in (0, 1, 2)]
        return softmax_score, regression_scores, models_results

    def visualize_fragment(self, frag_info, video_imgs_path, frag_out_path):
        images = []
        video_imgs_path = sorted(video_imgs_path, key=lambda p: int(p.split('_')[-1].split('.')[0]))[40:]
        for n, img_path in enumerate(video_imgs_path):
            image = cv2.imread(img_path)
            image = cv2.copyMakeBorder(image, 0, 50, 0, 300, cv2.BORDER_CONSTANT, value=0)

            sm_score, lr_scores, models_results = self.get_from_results(frag_info, n)

            # max_sm_score_num = np.argmax(sm_score)
            # cv2.putText(image, 'Multi class', (600, 35), cv2.FONT_ITALIC, 0.7, (255, 255, 255))
            # for score_num, (score, cls_name, cls_size) in enumerate(zip(sm_score, emotion_buckets, emotion_sizes)):
            #     color = (0, 255, 0) if score_num == max_sm_score_num else (255, 255, 255)
            #     cv2.putText(image, cls_name, (600 + score_num * 60, 50), cv2.FONT_ITALIC, cls_size, color)
            #     cv2.putText(image, '%1.3f' % score, (600 + score_num * 60, 65), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
            #
            # cv2.putText(image, 'Single class regression', (600, 120), cv2.FONT_ITALIC, 0.7, (255, 255, 255))
            # for score_num, (score, cls_name, cls_size) in enumerate(zip(lr_scores, emotion_buckets, emotion_sizes)):
            #     color = (0, 255, 0) if score > 0.4 else (255, 255, 255)
            #     cv2.putText(image, cls_name, (600 + score_num * 60, 135), cv2.FONT_ITALIC, cls_size, color)
            #     cv2.putText(image, '%1.3f' % score, (600 + score_num * 60, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

            score_args = np.argsort(sm_score)[::-1]
            sm_score = sm_score[score_args]
            e_buckets = np.array(emotion_buckets)[score_args]

            for n, (cls_name, score) in enumerate(zip(e_buckets, sm_score)):
                color = [122, 122, 122]
                color[1] += int(score * 250)
                cv2.putText(image, cls_name, (600, 50 + 45 * n), cv2.FONT_ITALIC, 1, color)
                cv2.putText(image, '%.3f' % score, (800, 50 + 45 * n), cv2.FONT_HERSHEY_COMPLEX, 1, color)

            for model_num, model_result in enumerate(models_results):
                for n_res, res in enumerate(model_result):
                    cv2.putText(image, '%.3f' % res, (n_res * 60, 350 + model_num * 15), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (255, 255, 255))

            # cv2.imshow('image', image)
            # if cv2.waitKey(0) & 0xff == ord('q'):
            #     exit(0)
            images.append(image.copy())

        for n, image in enumerate(images):
            cv2.imwrite(osp.join(frag_out_path, '%04d.jpg' % n), image)
        # print frag_out_path + '.mp4'
        # video = cv2.VideoWriter(frag_out_path + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 25.0, images[0].shape[:2][::-1])
        # for image in images:
        #     # cv2.imshow('image', image)
        #     # if cv2.waitKey(0) & 0xff == ord('q'):
        #     #     exit(0)
        #     print image.shape
        #     video.write(image)
        # video.release()

        return images

    def visualize_video(self, data_path, results_path, video_name, out_path):
        print video_name
        if not osp.exists(out_path):
            os.mkdir(out_path)
        video_out_path = osp.join(out_path, video_name)
        if not osp.exists(video_out_path):
            os.mkdir(video_out_path)

        video_imgs_path = osp.join(data_path, video_name, 'images')
        video_key_corr_path = osp.join(data_path, video_name, 'FEATURES/key_corresponding.json')
        video_results_path = osp.join(results_path, video_name)
        fragments_info_path = osp.join(data_path, video_name, 'FEATURES/fragments_info.json')
        main_wav_path = osp.join(data_path, video_name, '%s.wav' % video_name)

        assert osp.exists(video_imgs_path)
        assert osp.exists(video_key_corr_path)
        assert osp.exists(video_results_path)
        assert osp.exists(fragments_info_path)
        assert osp.exists(main_wav_path)

        with open(video_key_corr_path, 'r') as f:
            video_key_corr = json.load(f)

        with open(fragments_info_path, 'r') as f:
            fps, _ = json.load(f)

        fs, wav = read_wave(main_wav_path)
        if len(wav.shape) > 1:
            wav = wav[:, 0]

        full_video_path = osp.join(video_out_path, 'full')
        if not osp.exists(full_video_path):
            os.mkdir(full_video_path)
        last_image_id = 0
        full_audio = []

        for frag_info_path in tqdm(sorted(os.listdir(video_results_path), key=lambda k: int(k.split('_')[-1].split('.')[0]))):
            with open(osp.join(video_results_path, frag_info_path), 'r') as f:
                frag_info = json.load(f)
            frag_num = int(frag_info_path.split('_')[-1].split('.')[0])
            frag_out_path = osp.join(video_out_path, 'fragment_%d' % frag_num)
            if not osp.exists(frag_out_path):
                os.mkdir(frag_out_path)
            fragment_faces = os.listdir(osp.join(self.data_path, video_name, 'fragment_%d' % frag_num, 'f_imgs'))
            fragment_imgs = [video_key_corr[k] for k in map(lambda k: k.split('.')[0], fragment_faces)]
            if len(fragment_imgs) < 40 + fps:
                continue
            images = self.visualize_fragment(frag_info, map(lambda p: osp.join(video_imgs_path, p), fragment_imgs),
                                             frag_out_path)

            img_nums = sorted(map(lambda ff: float(ff.split('.')[0]), fragment_faces))[40:]
            sec_nums = map(lambda n: float(fs * n) / fps, img_nums)
            secs = sorted(reduce(lambda a, b: a + b, [list(range(int(sec_num), int(sec_num + fs / fps)))
                                               for sec_num in sec_nums], []))
            # wav_start, wav_end = fs * min(img_nums) / fps, fs * max(img_nums) / fps
            fragment_audio = wav[secs]
            write_wave(frag_out_path + '.wav', fragment_audio, fs)
            full_audio = full_audio + fragment_audio.tolist()
            call(["ffmpeg", '-r', str(int(fps + 0.5)), '-i', osp.join(frag_out_path, '%04d.jpg'),
                  '-i', frag_out_path + '.wav', '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p', '-crf', '23',
                  '-y', osp.join(video_out_path, 'fragment_%d.mp4' % frag_num)])#, stderr=dev_null, stdout=dev_null)
            # rmtree(frag_out_path)
            # os.remove(frag_out_path + '.wav')
            for image in images:
                cv2.imwrite(osp.join(full_video_path, '%04d.jpg' % last_image_id), image)
                last_image_id += 1
        # write_wave(full_video_path + '.wav', np.array(full_audio), fs)
        # call(["ffmpeg", '-r', str(fps), '-i', osp.join(full_video_path, '%04d.jpg'),
        #       '-i', full_video_path + '.wav', '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p', '-crf', '23',
        #       '-y', osp.join(video_out_path, 'full.mp4')], stderr=dev_null, stdout=dev_null)

if __name__ == '__main__':
    # nets_path = '/home/mopkobka/NDL-Projects/emotion-recognition-prototype/andrew/v3/mxNets'
    net1 = '/media/mopkobka/512Gb SSD2/mxNets/big_net_3d_new_OMG__lbs_0-1-2-3-4-5-6_images-40_bodies-40_specters-9/29-04-2018_10-51-37/model', 33
    # net2 = '/media/mopkobka/512Gb SSD2/mxNets/big_net_3d_new_OMG__lbs_0-1-2-3-4-5-6_images-40_bodies-40_specters-9/28-04-2018_20-10-35/model', 34
    emt = EmotionRecognitionTester(
        '/media/mopkobka/512Gb SSD2/DATA/V2/VERSION_OMG_TEST_v2',
        # '/SSD/DATA/EmotionMinerCorpus/VERSION_21_03/data_merged',
        [
            net1,
            # net2,
            # (osp.join(nets_path, 'big_net_21_03__lbs_0-1-2-3-4_images-40_bodies-40_specters-10'), 9),
            # (osp.join(nets_path, 'big_net_21_03__lbs_0-1-2-3-4_images-40_bodies-40_specters-10'), 13),
            # (osp.join(nets_path, 'big_net_21_03__lbs_0-1-2-3-4_images-40_bodies-40_specters-10'), 18),
            # (osp.join(nets_path, 'big_net_OMG__lbs_0-1-2-3-4-5_images-40_bodies-40_specters-10'), 25),
            # (osp.join(nets_path, 'big_net_OMG__lbs_0-1-2-3-4-5_images-40_bodies-40_specters-10'), 19),
            # (osp.join(nets_path, 'big_net_OMG__lbs_0-1-2-3-4-5_images-40_bodies-40_specters-10'), 13),
            # (osp.join(nets_path, 'big_net_21_03__lbs_0-1-2-3-4_images-40_bodies-40_specters-10'), 4),
            # (osp.join(nets_path, 'big_net_21_03__lbs_0-1-2-3-4_images-40_bodies-40_specters-10'), 5),
            # (osp.join(nets_path, 'big_net_21_03__lbs_0-1-2-3-4_images-40_bodies-40_specters-10'), 6),
            # (osp.join(nets_path, 'big_net_21_03__lbs_0-1-2-3-4-5_images-60_bodies-60_specters-12'), 11) # from tmpNets
        ],
        (0, 3, 4, 5, 6, 7, 8, 9, 1, 2),
        ('images', 'body', 'specters'),
        ((12, 40, 3, 112, 112), (12, 40, 36), (12, 9, 3, 40, 40)),
        ('emotions', 'arr', 'val'),
        ((12, 7), (12, 1), (12, 1)),
        12,
        init_nets=True
    )

    emt.process_data('/media/mopkobka/512Gb SSD2/DATA/V2/VERSION_OMG_TEST_v2_results', with_tqdm=True)
    # emt.calculate_prediction('/SSD/DATA/EmotionMinerCorpus/VERSION_21_03/results/aolbuild81/fragment_0.json')
    # emt.calculate_metrics(osp.join(config.DATA.home, 'all_labels.json'),
    #                       osp.join(config.DATA.home, 'results'),
    #                       [8, 10, 11, 12, 21], config.DATA.test_info)

    # for video_name in os.listdir('/SSD/DATA/EmotionMinerCorpus/_TMP_DIR/tmp'):
    #     if video_name not in ('ted1172', '_Ellen191', 'jobint150', '_10_Questions14'):
    #         continue
    #     try:
    #         emt.visualize_video('/SSD/DATA/EmotionMinerCorpus/_TMP_DIR/tmp', osp.join(config.DATA.home, 'merged_results'),
    #                             video_name, osp.join(config.DATA.home, 'visualize_merged_results'))
    #     except AssertionError as e:
    #         print e, video_name
    # emt.visualize_video('/SSD/DATA/EmotionMinerCorpus/_TMP_DIR/tmp', osp.join(config.DATA.home, 'results'),
    #                     '_10_Questions12', osp.join(config.DATA.home, 'visualize_results'))
    # emt.visualize_video('/SSD/DATA/EmotionMinerCorpus/_TMP_DIR/tmp', osp.join(config.DATA.home, 'results'),
    #                     '_10_Questions13', osp.join(config.DATA.home, 'visualize_results'))
    # emt.visualize_video('/SSD/DATA/EmotionMinerCorpus/_TMP_DIR/tmp', osp.join(config.DATA.home, 'results'),
    #                     '_10_Questions14', osp.join(config.DATA.home, 'visualize_results'))
    # emt.visualize_video('/SSD/DATA/EmotionMinerCorpus/_TMP_DIR/tmp', osp.join(config.DATA.home, 'results'),
    #                     '_10_Questions15', osp.join(config.DATA.home, 'visualize_results'))
