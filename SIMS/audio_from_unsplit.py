import pickle
import os
import numpy as np


with open("D:\Search\MSA\MOSEI/unaligned_unsplit_data.pkl", 'rb') as file:
    data = pickle.load(file)
    feature_A = data['audio']

    save_path = os.path.join("D:\Search\MSA\MOSEI\\AudioFeature", 'AudioFeature.npz')
    np.savez(save_path,feature_A=feature_A)