"""
NeurodataLab LLC 29.04.2018
Created by Andrey Belyaev
"""
import csv
import json
import numpy as np
import os
import os.path as osp
from tqdm import tqdm


def make_submission(results_dir, out_file):
    ret = {}
    for video_name in tqdm(os.listdir(results_dir)):
        video_path = osp.join(results_dir, video_name)
        video_info = {}
        for fragment_path in map(lambda p: osp.join(video_path, p), os.listdir(video_path)):
            with open(fragment_path, 'r') as f:
                frag_predictions = json.load(f)
                if len(frag_predictions) == 0:
                    continue
                frag_predictions = frag_predictions[0]
                predictions = [np.array(model_result).mean(axis=0) for model_result in frag_predictions]
                prediction = np.array(predictions).mean(axis=0).argmax()
                frag_info = {
                    'cls': prediction,
                    'arr': 0.0,
                    'val': 0.0
                }
                video_info[int(fragment_path.split('_')[-1].split('.')[0])] = frag_info
        ret[video_name] = video_info

    writer = csv.writer(open(out_file, 'w'))
    writer.writerow(['name', 'utterance', 'arousal', 'valence', 'EmotionMaxVote'])
    for video_name, video_res in ret.items():
        for frag_num, frag_info in video_res.items():
            writer.writerow([video_name, 'utterance_%d.mp4' % (frag_num + 1),
                             frag_info['arr'], frag_info['val'], frag_info['cls']])


def rename_submissions(submission_path, corr_path, out_file):
    sub_reader = csv.reader(open(submission_path, 'r'))
    sub_rows = [line for line in sub_reader]
    corr_reader = csv.reader(open(corr_path, 'r'), delimiter=';')
    corr = {l[1]: l[0] for l in corr_reader if len(l) > 1}
    out_rows = [sub_rows[0]]
    for row in sub_rows:
        if row[0] in corr:
            out_rows.append([corr[row[0]]] + row[1:])
    writer = csv.writer(open(out_file, 'w'), delimiter=',')
    for r in out_rows:
        writer.writerow(r)


def create_final_submission(submission_path, names_path, out_path):
    sub_reader = csv.reader(open(submission_path, 'r'), delimiter=',')
    sub_rows = [l for l in sub_reader]
    sub_info = {}
    for link, utter, ar, val, cls in sub_rows[1:]:
        if link.split('?')[-1] not in sub_info:
            sub_info[link.split('?')[-1]] = {}
        sub_info[link.split('?')[-1]][utter] = ar, val, cls

    names_reader = csv.reader(open(names_path, 'r'), delimiter=',')
    names_reader.next()
    out_writer = csv.writer(open(out_path, 'w'), delimiter=',')
    out_writer.writerow(['link', 'start', 'end', 'video', 'utterance', 'arousal', 'valence', 'EmotionMaxVote'])
    for link, start, end, video, utter in names_reader:
        if link.split('?')[-1] in sub_info:
            if utter in sub_info[link.split('?')[-1]]:
                u = sub_info[link.split('?')[-1]][utter]
                out_writer.writerow([link, start, end, video, utter, u[0], u[1], u[2]])


if __name__ == '__main__':
    make_submission('/SSD2/DATA/VERSION_OMG_TEST_results',
                    '/SSD2/DATA/OMG_TEST_RESULTS_tmp_full_name.csv')
    rename_submissions('/SSD2/DATA/OMG_TEST_RESULTS_tmp_full_name.csv',
                       '/SSD2/DATA/OMG_name_corr.csv',
                       '/SSD2/DATA/OMG_TEST_RESULTS_tmp_renamed.csv')
    create_final_submission('/SSD2/DATA/OMG_TEST_RESULTS_tmp_renamed.csv',
                            '/SSD2/DATA/omg_TestVideos_WithoutLabels.csv',
                            '/SSD2/DATA/OMG_TEST_RESULTS_final.csv')
