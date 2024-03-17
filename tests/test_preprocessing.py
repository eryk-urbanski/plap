import numpy as np
import pytest
import sys

sys.path.append("..")
from plap.core.audio_info import AudioInfo
from plap.core.preprocessing import Preprocessing


@pytest.mark.parametrize(
    "file, block_size, overlap, expected_nblocks",
    [
        (
            "data/001_guit_solo.wav",
            512,
            0,
            np.floor(3924900/512) + 1),
        (
            "data/redhot.wav",
            512,
            0,
            np.floor(661500/512) + 1,
        ),
    ],
)
def test_framing(file, block_size, overlap, expected_nblocks):

    x = AudioInfo(file)
    blocks = Preprocessing.framing(x, block_size, overlap)

    assert blocks.shape == (expected_nblocks, block_size)


@pytest.mark.parametrize(
    "file, block_size, overlap", [("data/001_guit_solo.wav", 512, 0)]
)
@pytest.mark.parametrize("window_type", ["hann", "hamming", "bartlett", "blackman"])
def test_windowing(file, block_size, overlap, window_type):

    x = AudioInfo(file)
    blocks = Preprocessing.framing(x, block_size, overlap)
    windowed_blocks = Preprocessing.windowing(blocks=blocks, window_type=window_type)
    assert windowed_blocks.shape == blocks.shape


@pytest.mark.parametrize(
    "file, block_size, overlap, window_type",
    [
        ("data/001_guit_solo.wav", 512, 0, "hann"),
    ],
)
def test_fft(file, block_size, overlap, window_type):

    x = AudioInfo(file)
    blocks = Preprocessing.framing(x, block_size, overlap)
    windowed_blocks = Preprocessing.windowing(blocks=blocks, window_type=window_type)
    dft_blocks = Preprocessing.fft(windowed_blocks=windowed_blocks)
    assert dft_blocks.shape == windowed_blocks.shape
