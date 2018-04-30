"""
NeurodataLab LLC 04.04.2018
Created by Andrey Belyaev
"""
import mxnet as mx
import numpy as np
from sklearn.metrics import log_loss, confusion_matrix, f1_score


class AveragePrecision(mx.metric.EvalMetric):
    def __init__(self, num_output=0, freq=1, name='average_precision',
                 output_names=None, label_names=None):
        super(AveragePrecision, self).__init__(name, output_names=output_names, label_names=label_names)
        self.name = name
        self.num_output = num_output
        self.cm = None
        self.cur = 0.
        self.freq = freq
        self.count = 0
        self.iter_preds, self.iter_labels = [], []

    def update(self, labels, preds):
        labels = labels[0].asnumpy()
        preds = preds[self.num_output].asnumpy()
        if labels.shape[0] != preds.shape[0]:
            labels = np.repeat(labels, preds.shape[0] // labels.shape[0], axis=0)

        preds_cls = preds.argsort(axis=1)[:, ::-1]
        for l, p in zip(labels, preds_cls):
            classes = set(np.where(l == 1)[0].tolist())
            pred_classes = set(p[:len(classes)].tolist())
            good = classes & pred_classes
            self.iter_labels += list(good) + list(good ^ classes)
            self.iter_preds += list(good) + list(good ^ pred_classes)

        # labels_cls = np.argmax(labels, axis=1)
        # preds_cls = np.argmax(preds, axis=1)

        cm = confusion_matrix(self.iter_labels, self.iter_preds).astype(float)
        if np.any(map(np.isnan, cm[-1])) or cm.sum(axis=1).min() < 1e-10:
            return

        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        if np.any(map(np.isnan, cm[-1])):
            cm = cm[:-1, :-1]
        loss = cm.diagonal().mean()
        self.sum_metric = loss
        self.num_inst = 1
        self.cur += 1

        if self.cm is None:
            self.cm = cm

        if self.cur % self.freq == 0 and self.cur > 0:
            cm = confusion_matrix(self.iter_labels, self.iter_preds).astype(float)
            cm = cm / cm.sum(axis=1)[:, np.newaxis]
            print cm

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0
        self.cm = None
        self.cur = 0
        self.count = 0
        try:
            if len(self.iter_preds) > 2 and len(self.iter_labels) > 2:
                cm = confusion_matrix(self.iter_labels, self.iter_preds).astype(float)
                cm = cm / cm.sum(axis=1)[:, np.newaxis]
                if np.any(map(np.isnan, cm[-1])):
                    cm = cm[:-1, :-1]
                loss = cm.diagonal().mean()
                print '%s: %f' % (self.name, loss)
                print cm
                print
        except AttributeError:
            pass
        self.iter_preds, self.iter_labels = [], []


class F1Metric(mx.metric.EvalMetric):
    def __init__(self, num_output=0, name='f1',
                 output_names=None, label_names=None):
        super(F1Metric, self).__init__(name, output_names=output_names, label_names=label_names)
        self.name = name
        self.num_output = num_output
        self.cm = None
        self.cur = 0.
        self.count = 0
        self.iter_preds, self.iter_labels = [], []

    def update(self, labels, preds):
        labels = labels[0].asnumpy()
        preds = preds[self.num_output].asnumpy()
        if labels.shape[0] != preds.shape[0]:
            labels = np.repeat(labels, preds.shape[0] // labels.shape[0], axis=0)

        preds_cls = preds.argsort(axis=1)[:, ::-1]
        for l, p in zip(labels, preds_cls):
            classes = set(np.where(l == 1)[0].tolist())
            pred_classes = set(p[:len(classes)].tolist())
            good = classes & pred_classes
            self.iter_labels += list(good) + list(good ^ classes)
            self.iter_preds += list(good) + list(good ^ pred_classes)

        # labels_cls = np.argmax(labels, axis=1)
        # preds_cls = np.argmax(preds, axis=1)

        loss = f1_score(self.iter_labels, self.iter_preds, average='macro')
        self.sum_metric = loss
        self.num_inst = 1
        self.cur += 1

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0
        self.cm = None
        self.cur = 0
        self.count = 0
        self.iter_preds, self.iter_labels = [], []


class SVMAveragePrecision(mx.metric.EvalMetric):
    def __init__(self, num_output=0, freq=1, name='average_precision',
                 output_names=None, label_names=None):
        super(SVMAveragePrecision, self).__init__(name, output_names=output_names, label_names=label_names)
        self.name = name
        self.num_output = num_output
        self.cm = None
        self.cur = 0.
        self.freq = freq
        self.count = 0
        self.iter_preds, self.iter_labels = [], []

    def update(self, labels, preds):
        labels = labels[0].asnumpy()
        preds = preds[self.num_output].asnumpy()

        for l, p in zip(labels, preds):
            classes = set(np.where(l == 1)[0].tolist())
            pred_classes = set(p[:len(classes)].tolist())
            good = classes & pred_classes
            self.iter_labels += list(good) + list(good ^ classes)
            self.iter_preds += list(good) + list(good ^ pred_classes)

        # labels_cls = np.argmax(labels, axis=1)
        # preds_cls = np.argmax(preds, axis=1)

        cm = confusion_matrix(self.iter_labels, self.iter_preds).astype(float)
        if np.any(np.isnan(cm)) or cm.sum(axis=1).min() < 1e-10:
            return

        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        loss = cm.diagonal().mean()
        self.sum_metric += loss
        self.num_inst += 1
        self.cur += 1

        if self.cm is None:
            self.cm = cm

        if self.cur % self.freq == 0 and self.cur > 0:
            cm = confusion_matrix(self.iter_labels, self.iter_preds).astype(float)
            cm = cm / cm.sum(axis=1)[:, np.newaxis]
            print cm

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0
        self.cm = None
        self.cur = 0
        self.count = 0
        try:
            if len(self.iter_preds) > 2 and len(self.iter_labels) > 2:
                cm = confusion_matrix(self.iter_labels, self.iter_preds).astype(float)
                cm = cm / cm.sum(axis=1)[:, np.newaxis]
                loss = cm.diagonal().mean()
                print '%s: %f' % (self.name, loss)
                print cm
                print
        except AttributeError:
            pass
        self.iter_preds, self.iter_labels = [], []


class RegressionAverageAccuracy(mx.metric.EvalMetric):
    def __init__(self, name='reg_avg_accuracy', outputs=None, output_names=None, label_names=None):
        super(RegressionAverageAccuracy, self).__init__(name, output_names=output_names, label_names=label_names)

        self.outputs = outputs

    def update(self, labels, preds):
        if self.outputs is None:
            self.outputs = list(range(len(preds)))
        preds = [preds[i] for i in self.outputs]
        labels = labels[0].asnumpy().astype(int)

        loss = 0.
        for n in range(labels.shape[1]):
            p = preds[n].asnumpy()
            p = p[np.where(labels[:, n] == 1)[0]]
            p = np.floor(p + 0.5).flatten().astype(int) == 1
            loss += p.astype(int).mean() if len(p) > 0 else 0

        self.sum_metric += loss / labels.shape[1]
        self.num_inst += 1


class RegressionAveragePrecision(mx.metric.EvalMetric):
    def __init__(self, name='reg_avg_precision', outputs=None, output_names=None, label_names=None):
        super(RegressionAveragePrecision, self).__init__(name, output_names=output_names, label_names=label_names)

        self.outputs = outputs

    def update(self, labels, preds):
        if self.outputs is None:
            self.outputs = list(range(len(preds)))
        preds = [preds[i] for i in self.outputs]
        labels = labels[0].asnumpy().astype(int)
        preds_np = np.array(map(lambda p: p.asnumpy(), preds)).reshape(labels.shape)
        preds_np = np.floor(preds_np + 0.5).astype(int)

        tps, fps = [], []
        for n in range(labels.shape[1]):
            tp = np.logical_and(preds_np[:, n] == 1, labels[:, n] == 1).sum()
            fp = np.logical_and(preds_np[:, n] == 1, labels[:, n] == 0).sum()
            tps.append(tp)
            fps.append(fp)
        avg_precision = sum(map(lambda tp, fp: (float(tp) / (tp + fp)) if tp + fp > 0 else 0., tps, fps))

        # self.sum_metric += avg_precision / float(labels.shape[1])
        self.sum_metric += float(avg_precision) / labels.shape[1]
        self.num_inst += 1


class RegressionAccuracy(mx.metric.EvalMetric):
    def __init__(self, num_output=0, label_num=0, name='reg_acc',
                 output_names=None, label_names=None):
        super(RegressionAccuracy, self).__init__(name + '_%d' % num_output,
                                                 output_names=output_names, label_names=label_names)
        self.num_output = num_output
        self.label_num = label_num

    def update(self, labels, preds):
        labels_cls = labels[0].asnumpy()[:, self.label_num].astype(int)
        preds = preds[self.num_output].asnumpy()
        preds = preds[np.where(labels_cls == 1)[0]]
        preds = np.floor(preds + 0.5).flatten().astype(int) == 1
        self.sum_metric += preds.astype(int).mean() if len(preds) > 0 else 0
        self.num_inst += 1


class PrintMetric(mx.metric.EvalMetric):
    def __init__(self, name='print', frequent=1, num_batches=2,
                 output_names=None, label_names=None):
        super(PrintMetric, self).__init__(name, output_names=output_names, label_names=label_names)
        self.frequent = frequent
        self.cur = 0
        self.num_batches = num_batches

    def update(self, labels, preds):
        if self.cur % self.frequent == 0:
            labels = labels[0].asnumpy()[:self.num_batches].argmax(axis=1).tolist()
            metric_str = 'Labels: %s\n' % str(labels)

            for n_out, pred in enumerate(preds):
                pred = pred.asnumpy()
                metric_str += 'out #%d: %s; ' % (n_out, pred[:self.num_batches].argmax(axis=1).tolist())
                # metric_str += 'out #%d: %s; ' % (n_out, pred[:self.num_batches].tolist())
            print metric_str

        self.cur += 1
        self.sum_metric += 0
        self.num_inst += 1


class FeatNormMetric(mx.metric.EvalMetric):
    def __init__(self, num_output=-1, name='feat_norm',
                 output_names=None, label_names=None):
        super(FeatNormMetric, self).__init__(name, output_names=output_names, label_names=label_names)
        self.num_output = num_output

    def update(self, _, preds):
        loss = np.linalg.norm(preds[self.num_output].asnumpy(), axis=1).mean()
        self.sum_metric += loss
        self.num_inst += 1


class AnnoRegressionMae(mx.metric.EvalMetric):
    def __init__(self, num_outputs, label_num=1, name='anno_mae', output_names=None, label_names=None):
        super(AnnoRegressionMae, self).__init__(name, output_names=output_names, label_names=label_names)
        self.num_outputs = num_outputs
        self.label_num = label_num

    def update(self, labels, preds):
        loss = 0.
        labels = labels[self.label_num].asnumpy()
        for i, n in enumerate(self.num_outputs):
            label = labels[:, i].reshape((-1, 1))
            pred = preds[n].asnumpy()
            # loss += np.sqrt(((label - pred) ** 2).sum(axis=1)).mean()
            loss += np.abs(label - pred).sum(axis=1).mean()
            loss += 0

        self.sum_metric += loss / len(self.num_outputs)
        self.num_inst += 1


class RMSEMetric(mx.metric.EvalMetric):
    def __init__(self, num_output=0, name='rmse',
                 output_names=None, label_names=None):
        super(RMSEMetric, self).__init__(name, output_names=output_names, label_names=label_names)
        self.num_output = num_output

    def update(self, labels, preds):
        labels, preds = labels[self.num_output].asnumpy(), preds[self.num_output].asnumpy()
        loss = np.sqrt(((labels - preds) ** 2).mean())
        self.sum_metric += loss
        self.num_inst += 1
