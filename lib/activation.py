import math

def sigmoid(x: float) -> float:
    """
    Compute the sigmoid of x
    :param x:
    :return:
    """
    return 1.0 / (1.0 + math.exp(-x))

def d_sigmoid(x: float) -> float:
    """
    Calculates the derivative of the sigmoid function for a given input that has already been passed through sigmoid.
    :param x: The input.
    :return: The derivative.
    """
    return x * (1 - x)