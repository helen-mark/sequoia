import tensorflow as tf
from tensorflow import keras as K
from train_nn_classifier import create_dataset, prepare_all_features, dataframe_to_dataset


def test_model(_model: K.Model, _dataset_path: str, _batch_size: 32):
    optimizer = K.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', K.metrics.Recall(), K.metrics.Precision()])
    train_ds, val_ds, val_ds_2, train_dataframe, val_dataframe = create_dataset(_dataset_path, _batch_size)  # using same random state as during training. So getting the same train/val sampling

    result = {}
    model.evaluate(val_ds)

    # calculate metrics with manual choice of threshold:
    fn = 0
    fp = 0
    tn = 0
    tp = 0

    for sample in train_ds:
        ground_truth = int(sample[1])
        x_sample = sample[0]
        no = x_sample.pop('no')
        prediction = model.predict(x_sample, verbose=False)
        #result[int(no)] = prediction[0][0]
        pred = int(prediction[0][0] > 0.2)

        if pred == 1:
            if ground_truth == 1:
                tp += 1
            else:
                fp += 1
        elif pred == 0:
            if ground_truth == 0:
                tn += 1
            else:
                fn += 1

    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Recall: {tp/(tp+fn)}, Precision: {tp/(tp+fp)}")
    #for key in sorted(list(result.keys())):
    #    print("left" if result[key] > 0.5 else "works")


if __name__ == '__main__':
    dataset_path = 'data/october_works.csv'
    batch_size = 1
    model = (K.models.load_model('best_keras_model'))

    test_model(model, dataset_path, batch_size)