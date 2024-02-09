import numpy as np
import pytest
import sys
sys.path.append('..')
from plap.core.audio_info import AudioInfo

@pytest.mark.parametrize("file, expected_signal_shape, expected_sample_rate",
                         [
                             ('data/001_guit_solo.wav', (3924900,), 44100),
                             ('data/redhot.wav', (661500,), 22050),
                         ])

def test_init(file, expected_signal_shape, expected_sample_rate):

    x = AudioInfo(file)

    assert isinstance(x.signal, np.ndarray)
    assert x.signal.shape == expected_signal_shape
    assert x.sample_rate == expected_sample_rate
    assert x.blocks == []
