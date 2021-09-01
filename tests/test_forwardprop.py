from lib import *

def test_create():
    model = Model(input_shape=(5,))

    assert model.input_shape == (5,)

def test_get_error():
    model = Model(input_shape=(2,))
    model.add(Layer(2, 2))
    model.add(Layer(2, 2))

    