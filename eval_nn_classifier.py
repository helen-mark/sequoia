"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

from tensorflow import keras as K

from utils.dataset import create_dataset
from utils.custom_metric import calc_metric


def test_model(_model: K.Model, _dataset_path: str, _batch_size: 32):
    optimizer = K.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', K.metrics.Recall(), K.metrics.Precision()])
    train_ds, val_ds, val_ds_2, train_dataframe, val_dataframe = create_dataset(_dataset_path, _batch_size)  # using same random state as during training. So getting the same train/val sampling

    model.evaluate(val_ds)

    predictions = []
    ground_truth = []

    # calculate metrics with manual choice of threshold:
    for sample in train_ds:
        ground_truth.append(int(sample[1]))
        x_sample = sample[0]
        # no = x_sample.pop('no')
        prediction = model.predict(x_sample, verbose=False)
        predictions.append(prediction[0][0])

    calc_metric(predictions, ground_truth, 0.2)


if __name__ == '__main__':
    dataset_path = 'data/dataset-nn-small.csv'
    batch_size = 1
    model = (K.models.load_model('best_keras_model'))

    test_model(model, dataset_path, batch_size)