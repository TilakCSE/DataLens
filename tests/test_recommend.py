from datalens.models import AnalysisReport, ChartRecommendation, ColumnProfile
from datalens.recommend import rank_recommendations, recommend_charts


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


def test_ranking_diversifies_types_then_uses_measured_column_completeness():
    profiles = (
        ColumnProfile("full", "numeric", "float64", 100, 0, 80),
        ColumnProfile("partial", "numeric", "float64", 60, 40, 50),
        ColumnProfile("group", "categorical", "object", 90, 10, 3),
        ColumnProfile("date", "datetime", "object", 100, 0, 100),
    )
    ideas = (
        ChartRecommendation("histogram", "Full", ("full",), "numeric"),
        ChartRecommendation("histogram", "Partial", ("partial",), "numeric"),
        ChartRecommendation("scatter", "Pair", ("full", "partial"), "numeric pair"),
        ChartRecommendation("bar", "Groups", ("group",), "categorical"),
        ChartRecommendation("line", "Trend", ("date", "partial"), "dated measure"),
    )
    report = AnalysisReport(100, 4, 0, profiles, recommendations=ideas)

    ranked = rank_recommendations(report)

    assert [item.number for item in ranked] == [1, 4, 5, 3, 2]
    assert [item.lowest_non_null_count for item in ranked] == [100, 90, 60, 60, 60]
    assert [item.first_of_type for item in ranked] == [True, True, True, True, False]
    assert "60/100" in ranked[2].ranking_reason
    assert report.recommendations == ideas
    assert rank_recommendations(AnalysisReport(0, 0, 0, ())) == ()
