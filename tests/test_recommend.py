from datalens.models import ColumnProfile
from datalens.recommend import recommend_charts


def column(name: str, kind: str) -> ColumnProfile:
    return ColumnProfile(name, kind, "object", 5, 0, 5)


def test_recommends_type_based_charts_with_columns_and_reasons():
    profiles = (
        column("date", "datetime"),
        column("revenue", "numeric"),
        column("units", "numeric"),
        column("region", "categorical"),
        column("active", "boolean"),
    )

    recommendations = recommend_charts(profiles)

    assert [item.chart_type for item in recommendations] == [
        "histogram", "histogram", "bar", "line", "scatter"
    ]
    assert recommendations[3].columns == ("date", "revenue")
    assert recommendations[4].columns == ("revenue", "units")
    assert all(item.reason and all(name in item.reason for name in item.columns)
               for item in recommendations)
    assert all("confidence" not in item.reason.lower() for item in recommendations)


def test_empty_and_unplottable_profiles_have_no_recommendations():
    assert recommend_charts(()) == ()
    assert recommend_charts((column("flag", "boolean"), column("raw", "unknown"))) == ()


def test_does_not_mutate_profiles():
    profiles = (column("value", "numeric"),)
    assert len(recommend_charts(profiles)) == 1
    assert len(profiles) == 1


def test_titanic_like_profiles_avoid_identifier_and_high_cardinality_charts():
    def profile(name: str, kind: str, observed: int, unique: int) -> ColumnProfile:
        return ColumnProfile(name, kind, "object", observed, 0, unique)

    profiles = (
        profile("PassengerId", "numeric", 891, 891),
        profile("Survived", "numeric", 891, 2),
        profile("Pclass", "numeric", 891, 3),
        profile("Name", "categorical", 891, 891),
        profile("Ticket", "categorical", 891, 681),
        profile("Sex", "categorical", 891, 2),
        profile("Age", "numeric", 714, 88),
        profile("Fare", "numeric", 891, 248),
    )

    recommendations = recommend_charts(profiles)

    assert not any(
        name in item.columns
        for item in recommendations
        for name in ("PassengerId", "Name", "Ticket")
    )
    assert any(item.chart_type == "bar" and item.columns == ("Survived",) for item in recommendations)
    assert any(item.chart_type == "bar" and item.columns == ("Pclass",) for item in recommendations)
    assert any(item.chart_type == "scatter" and item.columns == ("Age", "Fare") for item in recommendations)
