# Standard Library
from pathlib import Path

# Third Party
import pytest

# First Party
from metadamage.utils import extract_name


# @pytest.mark.slow
# @pytest.mark.skip


def test_extracting_name_from_string():
    # given
    string = "./data/input/data_ancient.txt"
    # when
    result = extract_name(string)
    # then
    assert result == "data_ancient"


# @pytest.mark.skip
def test_extracting_name_from_path():
    # given
    path = Path("./data/input/data_ancient.txt")
    # when
    result = extract_name(path)
    # then
    assert result == "data_ancient"
