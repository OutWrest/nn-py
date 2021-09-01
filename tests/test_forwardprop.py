<<<<<<< HEAD
from lib import *

def test_create():
    model = Model(input_shape=(5,))

    assert model.input_shape == (5,)

def test_get_error():
    model = Model(input_shape=(2,))
    model.add(Layer(2, 2))
    model.add(Layer(2, 2))

=======
from lib import *

def test_create():
    model = Model(input_shape=(5,))

    assert model.input_shape == (5,)

def test_get_error():
    model = Model(input_shape=(2,))
    model.add(Layer(2, 2))
    model.add(Layer(2, 2))

>>>>>>> 7c2025e996510f7628466cb167b95d61d5ef2043
    