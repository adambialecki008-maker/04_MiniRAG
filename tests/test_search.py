from minirag.search import cosine_similarity
import pytest


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1, 0], [1, 0], 1.0),
        ([1, 0], [0, 1], 0.0),
        ([1, 0], [-1, 0], -1.0),
        ([1, 1], [1, 0], 0.70710678),
    ],
)
def test_cosine_similiarity(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)
