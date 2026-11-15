# Chart ranking method

DataLens can suggest many charts for a wide table. `datalens visualise file.csv`
shows five quick picks; `--all` shows the complete ranked list. The compact
`analyse` output shows the top three. The guided workflow lists all chart ideas
in the same ranking order.

The ranking uses only the `AnalysisReport` produced by `dl.analyse(df)`:

1. For every chart, find the lowest `non_null_count` among its required columns.
2. Select the chart with the highest such count within each available chart
   type. This gives a first pick for a time line, category bar, histogram, and
   scatter plot when those types exist.
3. Order these first picks by that count. Ties use the fixed type order: line,
   bar, histogram, scatter.
4. Order the remaining charts by the same count. Ties keep the original
   recommendation number.

Each displayed idea includes its original 1-based number and a `Priority` line
with the measured count. `dl.rank_charts(df)` returns these numbers and reasons
to Python users. `dl.visualise(df, chart=N)` and `report.recommendations[N-1]`
retain their original meaning; ranking changes display order only. The JSON
analysis report retains the original recommendation order.

This is a deterministic browsing heuristic. It does not estimate chart quality,
statistical significance, or the likelihood of a useful insight. For two-column
charts, the lower of the two non-missing counts is **not** the number of rows
where both values are present. The rendered chart reports actual plotted rows
and exposes the exact `plot_data` used for the figure.

On the locally saved validation datasets, the ranked top five for Breast Cancer
now include a diagnosis count bar, a numeric histogram, and a numeric scatter
plot among its 32 ideas. The earlier source order placed histograms first.
This confirms that the list is more varied; it does not establish that the
selected charts are the most informative for every user's question.
