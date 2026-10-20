from typing import Any

import pytest

from text2cypher import Config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attempts_w_filtered": -1}, "attempts_w_filtered"),
        ({"total_attempts": -1, "attempts_w_filtered": 0}, "total_attempts"),
        ({"attempts_w_filtered": 3, "total_attempts": 2}, "at least"),
    ],
)
def test_config_rejects_impossible_limits(kwargs: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


def test_config_defaults():
    assert Config() == Config(attempts_w_filtered=3, total_attempts=5)


def test_execution_is_off_by_default_with_a_timeout():
    config = Config()

    assert config.execute is False
    assert config.execute_timeout_ms == 10_000


@pytest.mark.parametrize("timeout", [0, -5])
def test_config_rejects_a_timeout_that_is_not_positive(timeout: int):
    with pytest.raises(ValueError, match="execute_timeout_ms"):
        Config(execute_timeout_ms=timeout)


def test_config_allows_no_timeout():
    assert Config(execute_timeout_ms=None).execute_timeout_ms is None


def test_the_schema_is_read_on_every_run_by_default():
    assert Config().schema_refresh_seconds is None


def test_a_schema_refresh_time_cannot_be_negative():
    assert Config(schema_refresh_seconds=0).schema_refresh_seconds == 0
    with pytest.raises(ValueError, match="schema_refresh_seconds"):
        Config(schema_refresh_seconds=-1)


def test_types_are_checked_and_performance_is_not_by_default():
    config = Config()

    assert (config.check_types, config.check_performance) == (True, False)


def test_values_are_written_in_the_query_by_default():
    assert Config().use_parameters is False
