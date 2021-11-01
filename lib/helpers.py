import math

from typing import List

def reshape_list(data: List[List[float]]) -> List[float]:
    """
    Reshapes a x by y list of lists of floats into a y by x list of lists of floats.
    :param data: The list of lists.
    :return: The reshaped list.
    """
    ret = []
    for i in range(len(data[0])):
        ret.append([])
        for j in range(len(data)):
            ret[i].append(data[j][i])
    return ret

def dot(v1: List[float], v2: List[float]) -> List[float]:
    """
    Calculates the dot product of v1 and v2.
    :param v1: The first vector.
    :param v2: The second vector.
    :return: The dot product of v1 and v2.
    """
    if len(v1) != len(v2):
        raise ValueError("Error in processing the dot product. The lengths of two vectors do not match.")

    return sum(x * y for x, y in zip(v1, v2))

def t_error(expected: List[float], actual: List[float]) -> float:
    """
    Calculates the total error between the expected and actual values.
    :param expected: The expected values.
    :param actual: The actual values.
    :return: The error.
    """
    if len(expected) != len(actual):
        raise ValueError("Expected and actual data have different lengths.")

    return sum([0.5 * math.pow(expected[i] - actual[i], 2) for i in range(len(expected))])
    
def l_error(expected: float, actual: float) -> float:
    """
    Calculates the error between the expected and actual values.
    :param expected: The expected value.
    :param actual: The actual value.
    :return: The error.
    """
    return 0.5 * math.pow(expected - actual, 2)