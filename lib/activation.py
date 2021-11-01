import math

def sigmoid(x: float) -> float:
    """
    Compute the sigmoid of x.
    :param x: The input.
    :return: The sigmoid of x.
    """
    return 1.0 / (1.0 + math.exp(-x))

def d_sigmoid(x: float) -> float:
    """
    Calculates the derivative of the sigmoid function for a given input that has already been passed through sigmoid.
    :param x: The input.
    :return: The derivative.
    """
    return x * (1 - x)

def relu(x: float) -> float:
    """
    Compute the ReLu of x.
    :param x: The input.
    :return: The ReLu of x.
    """
    return max(x, 0)

def d_relu(x: float) -> float:
    """
    Compute the derivative of the ReLu function for a given input that has already been passed throuh ReLu.
    :param x: The input.
    :return: The derivative.
    """
    return float(x > 0)