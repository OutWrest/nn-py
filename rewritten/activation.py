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
    Compute the sigmoid derivative of x
    :param x:
    :return:
    """
    return sigmoid(x) * (1.0 - sigmoid(x))