## Bottleneck Forecasting Under a Stricter Evaluation

Discrete-event simulation of a 5-station production line (Furnace → Rolling → Cooling → Coating → Packaging), driven by a real energy dataset, plus a comparison of statistical and machine-learning forecasts of the bottleneck's utilization. The line is **illustrative** (station parameters are assumed); the demand signal is real.

**Data:** [Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851) (UCI #851), 15-minute readings for 2018.

## What it does

1. **Simulation + bottleneck ID:** SimPy line, with the bottleneck checked under ±20% changes to every service time.
2. **Forecasting:** seasonal naive, ARIMA, SARIMA, hybrid SARIMA+XGBoost and recursive XGBoost, compared at 1 h, 4 h, 1 day and 10 days on rolling weekly origins. No model sees data after its forecast origin. Diebold-Mariano tests and 90% prediction intervals included.
3. **Alerts:** an XGBoost classifier that warns 15 min and 1 h before high utilization, at a fixed false-alarm budget.

## Results

<!-- RESULTS:START -->
- **Furnace** is the bottleneck: highest mean utilization (29.6%) and still the bottleneck in all 10 one-at-a-time and 30 random ±20% scenarios.

| Model (MASE, lower is better) | 1 h | 4 h | 1 day | 10 days |
|---|---|---|---|---|
| Seasonal naive (weekly) | 0.444 | 0.440 | 0.719 | 0.936 |
| ARIMA | 0.395 | 0.520 | 0.830 | 1.180 |
| SARIMA | 0.432 | 0.441 | 0.864 | 0.858 |
| Hybrid SARIMA+XGB | 0.376 | 0.400 | 0.834 | 0.826 |
| **XGBoost (recursive)** | **0.346** | **0.330** | **0.640** | **0.765** |

XGBoost is significantly better than the best classical model at 4 h and 10 days, and not significantly different at 1 h and 1 day.
<!-- RESULTS:END -->

This block is rewritten by `scripts/update_readme.py` at the end of each pipeline run; the full tables are in `results/results.json`.

## Layout

```
src/bottleneck_sim/   simulation, evaluation, forecasting models
scripts/              run_pipeline.py (end to end), update_readme.py
tests/                pytest suite
analysis/
  shared.py           helpers used by both analyses
  decision/           decision-aligned analysis
  horizon_origin/     horizon and origin effects
data/  results/  figures/
```

## Run

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py        # about 40-60 min; --quick for a 5-min smoke test
python analysis/decision/decision_analysis.py --jobs 4
python analysis/horizon_origin/interval_check.py
pytest tests/
```

Both `analysis/` folders are independent of each other but need the main pipeline to run first. Intermediate results are cached in `results/cache/`, so interrupted runs resume.

## Limitations

- Station parameters are assumptions, and the forecast target is simulated, so results describe this simulated system, not a real plant.
- The sensitivity analysis covers mean service times only, not variability, breakdowns or buffers.
- Alerts are scored per 15-minute period, not per alert episode.


## References

Box & Jenkins (1970); Goldratt & Cox (1984); Roser, Nakano & Tanaka (2001);
Law (2015); Hyndman & Athanasopoulos (2021); Hyndman & Koehler (2006);
Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997); Holm (1979);
Tashman (2000); Lei et al. (2018); Chen & Guestrin (2016);
Sathishkumar, Shin & Cho (2021) — full citations in the report.

## License

MIT — see [`LICENSE`](LICENSE).