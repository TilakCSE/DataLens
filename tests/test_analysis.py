import json

import pandas as pd

from datalens.analysis import profile_dataframe


def test_profiles_clean_dataframe_without_modifying_it():
    frame = pd.DataFrame(
        {
            "amount": [10, 20, 30],
            "group": ["east", "west", "east"],
            "active": [True, False, True],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )
    before = frame.copy(deep=True)

    profiles, duplicate_count, issues = profile_dataframe(frame)

    pd.testing.assert_frame_equal(frame, before)
    assert duplicate_count == 0
    assert issues == ()
    by_name = {profile.name: profile for profile in profiles}
    assert by_name["amount"].kind == "numeric"
    assert by_name["amount"].statistics["mean"] == 20
    assert by_name["amount"].statistics["median"] == 20
    assert by_name["group"].kind == "categorical"
    assert by_name["group"].statistics["top_values"][0] == {"value": "east", "count": 2}
    assert by_name["active"].kind == "boolean"
    assert by_name["date"].kind == "datetime"
    assert by_name["date"].statistics["min"] == "2026-01-01T00:00:00"
    assert by_name["date"].statistics["max"] == "2026-01-03T00:00:00"
    json.dumps([profile.statistics for profile in profiles], allow_nan=False)


def test_missing_mixed_and_duplicate_rows():
    frame = pd.DataFrame(
        {
            "amount": [10.0, None, 10.0, 20.0],
            "mixed": ["a", 1, "a", None],
        }
    )
    profiles, duplicate_count, issues = profile_dataframe(frame)
    by_name = {profile.name: profile for profile in profiles}

    assert duplicate_count == 1
    assert by_name["amount"].missing_count == 1
    assert by_name["amount"].non_null_count == 3
    assert by_name["amount"].unique_count == 2
    assert by_name["mixed"].kind == "categorical"
    assert by_name["mixed"].missing_count == 1
    assert by_name["mixed"].unique_count == 2
    assert [issue.code for issue in issues] == [
        "missing_values", "missing_values", "mixed_types", "duplicate_rows"
    ]


def test_empty_frame_and_all_missing_column():
    empty = pd.DataFrame({"empty": pd.Series(dtype="object")})
    profiles, duplicates, issues = profile_dataframe(empty)
    assert duplicates == 0
    assert issues == ()
    assert profiles[0].kind == "unknown"
    assert profiles[0].missing_count == 0
    assert profiles[0].unique_count == 0

    all_missing = pd.DataFrame({"empty": [None, pd.NA]})
    profiles, duplicates, issues = profile_dataframe(all_missing)
    assert profiles[0].missing_count == 2
    assert profiles[0].non_null_count == 0
    assert profiles[0].statistics == {}
    assert duplicates == 1
    assert {issue.code for issue in issues} == {"missing_values", "duplicate_rows"}


def test_nested_values_are_counted_without_hash_errors():
    frame = pd.DataFrame({"items": [[1, 2], [1, 2], {"x": 3}]})
    profiles, duplicates, issues = profile_dataframe(frame)
    assert profiles[0].unique_count == 2
    assert duplicates == 1
    assert any(issue.code == "duplicate_rows" for issue in issues)
    json.dumps(profiles[0].statistics, allow_nan=False)


def test_date_strings_are_recognized_and_single_value_std_is_json_safe():
    frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-03"], "number": [4, None]})
    profiles, _, _ = profile_dataframe(frame)
    assert profiles[0].kind == "datetime"
    assert profiles[0].statistics == {"min": "2026-01-01T00:00:00", "max": "2026-01-03T00:00:00"}
    assert profiles[1].statistics["std"] is None
    json.dumps([profile.statistics for profile in profiles], allow_nan=False)
