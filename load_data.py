# todo : This class will support various data preprocessing
# todo : Rename method being different from module name

import numpy as np

def load_data(path):
    print(path)
    return np.load(path)