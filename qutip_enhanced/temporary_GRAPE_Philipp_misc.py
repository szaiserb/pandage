import numpy as np
import datetime

SDx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype='complex_')/2.
SDy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype='complex_')/2.
SDz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype='complex_')/2.
STx = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype='complex_')/np.sqrt(2)
STy = np.array([[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]], dtype='complex_')/np.sqrt(2)
STz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype='complex_')

def xy2aphi(xy):
    norm = np.array([np.linalg.norm(xy, axis=1)]).transpose()
    phi = np.arctan2(xy[:, 1:2], xy[:,0:1])
    return np.concatenate([norm, phi], axis=1)

def aphi2xy(aphi):
    return np.array([aphi[:, 0] * np.cos(aphi[:, 1]),
                     aphi[:, 0] * np.sin(aphi[:, 1])]).T