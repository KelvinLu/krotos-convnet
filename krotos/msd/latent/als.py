import numpy as np

def confidence_transform(R, param_alpha, param_epsilon):
    C = R.copy()
    C.data = param_alpha * np.log(1 + param_epsilon * C.data)
    return C
