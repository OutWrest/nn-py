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