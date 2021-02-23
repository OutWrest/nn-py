from model import Model
from layer import Layer

model = Model()
model.addLayer(10, input_shape=2)
model.addLayer(10)

print(model.layers[0].neurons)

print(model.predict([1.0, 1.0]))