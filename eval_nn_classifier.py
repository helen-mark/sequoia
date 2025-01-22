import tensorflow as tf
from tensorflow import keras as K
from train_nn_classifier import create_dataset, prepare_all_features, dataframe_to_dataset


def test_model(_model: K.Model, _dataset_path: str, _batch_size: 32):
    optimizer = K.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', K.metrics.Recall(), K.metrics.Precision()])
    train_ds, val_ds, val_ds_2, train_dataframe, val_dataframe = create_dataset(_dataset_path, 1)
    result = {}
    for sample in val_ds:
        sample = sample[0]
        no = sample.pop('no')
        prediction = model.predict(sample, verbose=False)
        result[int(no)] = prediction[0][0]

    for key in sorted(list(result.keys())):
        print("letf" if result[key] > 0.5 else "works")


if __name__ == '__main__':
    dataset_path = '/home/elena/ATTRITION/sequoia/data/dataset-nn-small-numerated.csv'
    batch_size = 32
    model = (K.models.load_model('../model.tf'))

    test_model(model, dataset_path, batch_size)