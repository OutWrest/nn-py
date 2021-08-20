from layer import Layer
from activation import sigmoid, d_sigmoid

class Model:
    def __init__(self, input_shape: tuple) -> None:
        """
        Initializes a new instance of the Model class.
        :param input_shape: The shape of the input data.
        :return: None
        """
        self.layers = []
        self.input_shape = input_shape
        self.output_shape = None

    def add(self, layer: Layer) -> None:
        """
        Adds a new layer to the model.
        :param layer: The layer to add.
        :return: None
        """
        self.layers.append(layer)

    def propagate(self, input_data: list) -> list:
        """
        Propagates the input data through the model.
        :param input_data: The input data.
        :return: The output data through the model.
        """

        if len(self.layers) == 0:
            raise ValueError("Model has no layers.")
        
        if len(input_data) != self.input_shape[0]:
            raise ValueError("Input data has invalid shape.")

        output_data = input_data
        for layer in self.layers:
            output_data = layer.propagate(output_data, sigmoid)
        return output_data

if __name__ == "__main__":
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    example = Model(input_shape=(2,))

    example.add(Layer(2, 2))
    example.add(Layer(2, 2))

    example.layers[0].weights = [
        [.15, .20], [.25, .30]
    ]
    example.layers[0].bias = .35

    example.layers[1].weights = [
        [.40, .45], [.50, .55]
    ]
    example.layers[1].bias = .60

    print(example.propagate([.05, .10]))