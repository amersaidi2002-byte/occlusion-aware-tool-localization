import numpy as np
from toolloc.experiment import confidence


def test_sharper_peaks_raise_confidence():
    flat = np.zeros((1, 3, 64, 64), dtype=np.float32)
    sharp = flat.copy()
    sharp[:, 1:, 20, 30] = 10
    assert confidence(sharp)[0] > confidence(flat)[0]
