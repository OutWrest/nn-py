# Neural Network Py

This repository contains a neural network model library. It is inteded to learn machine learning implementations without the use of any outside machine learning libraries. It is really slow and zero optimizations. The math behind backprop is very simple and biases across layers remain unchanged. 

## Usage

```python
# Import the model and layers
from lib import *

# Only supports 1d input for now, flatten the input before passing it to the model
model = Model(input_shape=(2,)) 
# Similar to the usage of a Keras model, you can add layers to the model
model.add(Layer(nodes=2, prev_nodes=2))
model.add(Layer(nodes=2, prev_nodes=2))

print(model.forwardpropagate([.05, .10]))
print(f"Total Error Before Training: {example.get_error([.05, .10], [.01, .99])}")

# Model only takes a single data point at a time, so you can use a batch of data in a loops. Uses online learning.
model.train([.05, .10], [.01, .99])

print(f"Total Error After Training: {example.get_error([.05, .10], [.01, .99])}")
```
