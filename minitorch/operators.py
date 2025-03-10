"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def relu_back(x: float, y: float) -> float:
    """Compute the gradient of the ReLU function.

    Args:
    ----
        x (float): The input value.
        y (float): The upstream gradient.

    Returns:
    -------
        float: The gradient of the ReLU function at x, multiplied by y.

    """
    return (1 if x > 0 else 0) * y


def inv_back(x: float, y: float) -> float:
    """Compute the gradient of the inverse function.

    Args:
    ----
        x (float): The input value.
        y (float): The upstream gradient.

    Returns:
    -------
        float: The gradient of the inverse function at x, multiplied by y.

    """
    return -1 * inv(x * x)


def log_back(x: float, y: float) -> float:
    """Compute the gradient of the logarithm function.

    Args:
    ----
        x (float): The input value.
        y (float): The upstream gradient.

    Returns:
    -------
        float: The gradient of the logarithm function at x, multiplied by y.

    """
    return y * inv(x)


def inv(x: float) -> float:
    """Compute the inverse of a number.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The inverse of x.

    """
    return 1 / x


def log(x: float) -> float:
    """Compute the natural logarithm of a number.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x)


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The ReLU of x, which is max(x, 0).

    """
    return max(x, 0)


def exp(x: float) -> float:
    """Compute the exponential of a number.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    The sigmoid function is defined as:
    - For x >= 0: f(x) = 1.0 / (1.0 + exp(-x))
    - For x < 0: f(x) = exp(x) / (1 + exp(x))

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The sigmoid of x, a value between 0 and 1.

    """
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1 + exp(x))


def is_close(a: float, b: float) -> bool:
    """Check if two floating-point numbers are close to each other within a small tolerance.

    The function checks if the absolute difference between the two numbers is less than 1e-2.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if the absolute difference between a and b is less than 1e-2, False otherwise.

    """
    return -1e-2 < a - b < 1e-2
    return -1e-2 < a - b < 1e-2


def eq(a: float, b: float) -> bool:
    """Check if two floating-point numbers are equal.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if the numbers are equal, False otherwise.

    """
    return a == b


def lt(a: float, b: float) -> bool:
    """Check if the first floating-point number is less than the second.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if the first number is less than the second, False otherwise.

    """
    return a < b


def max(a: float, b: float) -> float:
    """Compute the maximum of two floating-point numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The larger of the two numbers, a or b.

    """
    return a if a > b else b


def neg(a: float) -> float:
    """Compute the negation of a number.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The negation of a, which is -a.

    """
    return -a


def id(a: float) -> float:
    """Compute the identity function.

    Args:
    ----
        a (float): The input value.

    Returns:
    -------
        float: The input value unchanged.

    """
    return a


def mul(a: float, b: float) -> float:
    """Compute the product of two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The product of a and b.

    """
    return a * b


def add(a: float, b: float):
    """Compute the sum of two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The sum of a and b.

    """
    return a + b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map():
    pass


def zipWith():
    pass


def reduce():
    pass


def addLists():
    pass


def negList():
    pass


def sum():
    pass


def prod():
    pass
