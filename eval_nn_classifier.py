import tensorflow as tf
from tensorflow import keras as K

def test_model():
    model = K.models.load_model('model.tf')
