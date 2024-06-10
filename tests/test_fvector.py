import pytest
import sys

sys.path.append("..")
from plap.parameterization.fvector import FeatureVector

@pytest.mark.parametrize(
    "feature",
    [
        ("rms"),
        ("zcr")
    ],
)
def test_fvector(
    feature: str
):
    try:
        fvector = FeatureVector(feature)
    except Exception as exc:
        assert False, f"'FeatureVector' raised an exception {exc}"


fvector = FeatureVector("rms")
print(len(fvector.features))
print(fvector.features.keys())