import numpy as np

# Reasonable parameters for Gaussian-log utility setting
params = {
    "a0": 100,         # Intended action (mean of output distribution)
    "sigma": 20,       # Standard deviation of output
    "sigma2": 20**2,   # Variance of output
    "w0": 50,          # Agent's initial wealth
    "lambda": 100,       # Lagrange multiplier on IR constraint
    "mu": 400,           # Lagrange multiplier on IC constraint
    "y_min": -100,     # Minimum output for plotting
    "y_max": 200       # Maximum output for plotting
}


def delta_g(y, lambda_, mu, a0, sigma2, w0):
    """
    Compute Delta g(y | lambda) for log utility and Gaussian output.

    Parameters:
    y : float or np.ndarray
        Output value(s)
    lambda_ : float
        Lagrange multiplier on the IR constraint
    mu : float
        Lagrange multiplier on the LIC constraint
    a0 : float
        Intended action
    sigma2 : float
        Variance of the Gaussian output
    w0 : float
        Agent's initial wealth

    Returns:
    float or np.ndarray
        Value(s) of Delta g(y | lambda)
    """
    # Compute the score
    S = (y - a0) / sigma2
    z = lambda_ + mu * S

    # Compute g(z) = log(max(z, w0))
    g_z = np.log(np.maximum(z, w0))
    g_lambda = np.log(np.maximum(lambda_, w0))

    # Compute Delta g
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = np.where(S != 0, (g_z - g_lambda) / (mu * S), 0.0)

    return delta





def delta_g_exponential(y, lambda_, mu, a0, w0):
    """
    Compute Delta g(y | lambda) for log utility and Exponential output.

    Parameters:
    y : float or np.ndarray
        Output value(s)
    lambda_ : float
        Lagrange multiplier on the IR constraint
    mu : float
        Lagrange multiplier on the LIC constraint
    a0 : float
        Intended action (mean of exponential distribution)
    w0 : float
        Agent's initial wealth

    Returns:
    float or np.ndarray
        Value(s) of Delta g(y | lambda)
    """
    # Score function for exponential distribution
    S = (y - a0) / (a0 ** 2)
    z = lambda_ + mu * S

    # Evaluate log of max to handle limited liability
    g_z = np.log(np.maximum(z, w0))
    g_lambda = np.log(np.maximum(lambda_, w0))

    # Compute Delta g
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = np.where(S != 0, (g_z - g_lambda) / (mu * S), 0.0)

    return delta
