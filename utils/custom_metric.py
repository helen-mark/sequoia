"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

def calc_metric(_predictions: list, _ground_truth: list, _thrs: float):
    assert len(_predictions) == len(_ground_truth)

    print(_predictions)
    print(_ground_truth)

    fn, fp, tn, tp = 0, 0, 0, 0
    for n, p in enumerate(_predictions):
        gt = _ground_truth[n]
        pred = int(p > _thrs)

        if pred == 1:
            if gt == 1:
                tp += 1
            else:
                fp += 1
        elif pred == 0:
            if gt == 0:
                tn += 1
            else:
                fn += 1

    if tp + fp == 0:
        print("Can't calculate precision: tp + fp = 0!")
        return

    if tp + fn == 0:
        print("Can't calculate recall: tp + fp = 0!")
        return

    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Recall: {tp/(tp+fn)}, Precision: {tp/(tp+fp)}")