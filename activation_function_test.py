import numpy as np


netInput = 100
result = float(1.0) / (float(1.0) + np.exp(-netInput))
print("result: ", result)