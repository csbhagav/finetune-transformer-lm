import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from datasets import _rocstories, _abd_rocstories


def rocstories(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _rocstories(
        os.path.join(data_dir, 'test.csv'))
    test_accuracy = accuracy_score(labels, preds) * 100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ROCStories Valid Accuracy: %.2f' % (valid_accuracy))
    print('ROCStories Test Accuracy:  %.2f' % (test_accuracy))


def abd_nli(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _abd_rocstories(os.path.join(data_dir, 'test.tsv'))
    test_accuracy = accuracy_score(labels, preds) * 100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ABD NLI Valid Accuracy: %.2f' % (valid_accuracy))
    print('ABD NLI Test Accuracy:  %.2f' % (test_accuracy))

    test_f1 = f1_score(labels, preds) * 100.
    print('ABD NLI Test F1 Score:  %.2f' % (test_f1))


def analyze_task_pair(data_dir_1, data_dir_2, outfile):
    pred_path_1 = os.path.join(data_dir_1, "ROCStories.tsv")
    pred_path_2 = os.path.join(data_dir_2, "ROCStories.tsv")

    preds_1 = pd.read_csv(pred_path_1, delimiter='\t')['prediction'].values.tolist()
    preds_2 = pd.read_csv(pred_path_2, delimiter='\t')['prediction'].values.tolist()

    ctx_1, y_1, z_1, labels_1 = _abd_rocstories(os.path.join(data_dir_1, 'test.tsv'))
    ctx_2, y_2, z_2, labels_2 = _abd_rocstories(os.path.join(data_dir_2, 'test.tsv'))

    joined_results = []
    for idx, (p1, p2, l1, l2) in enumerate(zip(preds_1, preds_2, labels_1, labels_2)):
        joined_results.append({
            'ctx_1': ctx_1[idx],
            'y1': y_1[idx],
            'z1': z_1[idx],
            'labels_1': l1,
            'preds_1': p1,
            'correct_1': l1 == p1,
            'ctx_2': ctx_2[idx],
            'y2': y_2[idx],
            'z2': z_2[idx],
            'labels_2': l2,
            'preds_2': p2,
            'correct_2': l2 == p2
        })
    with open(outfile, "w") as f:
        for r in joined_results:
            f.write(json.dumps(r))
            f.write("\n")
    f.close()
