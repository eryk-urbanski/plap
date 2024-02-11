import numpy as np
import pytest
import sys

sys.path.append("..")
from plap.core.audio_info import AudioInfo
from plap.core.preprocessing import Preprocessing


@pytest.mark.parametrize(
    "file, block_size, overlap, expected_nblocks",
    [
        ("data/001_guit_solo.wav", 512, 0, 7665),
        ("data/001_guit_solo.wav", 256, 50, 30662),
    ],
)
def test_framing(file, block_size, overlap, expected_nblocks):

    x = AudioInfo(file)
    Preprocessing.framing(x, block_size, overlap)

    assert all(
        isinstance(arr, np.ndarray) and arr.size == block_size for arr in x.blocks
    )
    assert len(x.blocks) == expected_nblocks


@pytest.mark.parametrize(
    "file, block_size, overlap", [("data/001_guit_solo.wav", 512, 0)]
)
@pytest.mark.parametrize("window_type", ["hann", "hamming", "bartlett", "blackman"])
def test_windowing(file, block_size, overlap, window_type):

    x = AudioInfo(file)
    Preprocessing.framing(x, block_size, overlap)
    Preprocessing.windowing(x, window_type)

    assert all(isinstance(arr, np.ndarray) for arr in x.windowed_blocks)


@pytest.mark.parametrize(
    "file, block_size, overlap, window_type",
    [
        ("data/001_guit_solo.wav", 512, 0, "hann"),
    ],
)
def test_fft(file, block_size, overlap, window_type):

    x = AudioInfo(file)
    Preprocessing.framing(x, block_size, overlap)
    Preprocessing.windowing(x, window_type)
    Preprocessing.fft(x)

    assert all(isinstance(arr, np.ndarray) for arr in x.dft_blocks)
    assert len(x.dft_blocks) == len(x.windowed_blocks)
