# Standard Library
from pathlib import Path

# Third Party
import pytest
from typer.testing import CliRunner

# First Party
from metadamage.cli import app
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


def test_cli_bad_file():
    # given
    runner = CliRunner()
    # when
    result = runner.invoke(app, ["file_which_does_not_exist.txt"])
    # then
    assert result.exit_code == 1
    assert isinstance(result.exception, Exception)


def test_cli_bad_files():
    # given
    runner = CliRunner()
    # when
    result = runner.invoke(
        app,
        [
            "file_which_does_not_exist.txt",
            "another_file_which_does_not_exist.txt",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, Exception)


def test_cli_version():
    # given
    runner = CliRunner()
    # when
    result = runner.invoke(app, ["--version"])
    # then
    assert result.exit_code == 0
    assert "version" in result.stdout


# result.exc_info
# result.exception
# result.exit_code
# result.output
# result.output_bytes
# result.runner