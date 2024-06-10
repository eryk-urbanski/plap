import numpy as np
import pytest
import sys

sys.path.append("..")
from plap.core.preprocessor import Preprocessor


@pytest.mark.parametrize(
    "path, \
    block_size, \
    overlap, \
    window_type, \
    preemphasis_coeff, \
    expected_sr, \
    expected_nblocks, \
    expected_spectrumsize",
    [
        ("data/redhot.wav", 512, 50, "hann", 0.68, 22050, 2583, 512),
    ],
)
def test_preprocess(
    path,
    block_size,
    overlap,
    window_type,
    preemphasis_coeff,
    expected_sr,
    expected_nblocks,
    expected_spectrumsize,
):
    preprocessor = Preprocessor(preemphasis_coeff, block_size, overlap, window_type)
    _, sample_rate, windowed_blocks, dft_blocks = preprocessor.preprocess(path)
    assert sample_rate == expected_sr
    assert windowed_blocks.shape[0] == expected_nblocks
    assert dft_blocks.shape[1] == expected_spectrumsize
