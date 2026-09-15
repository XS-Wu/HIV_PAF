# HIV cascade PAF analysis

This repository contains the core analysis used to estimate HIV infections and incidence rates attributable to unmet 95-95-95 cascade targets. It starts with an analysis-ready country-level file.

The example input file is simulated and is included only to make the repository runnable.


## Computational requirements

The example was developed and tested with:

- Python 3.11
- pandas 2.3.1
- numpy 2.4.4


Operating systems:
- macOS
- Linux
- Windows


## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.run_review_example
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

The runner writes the scenario summary and the fixed-multiplier sensitivity results to `outputs/`.


## Runtime
The complete workflow typically runs within 2 minutes on a standard laptop.


## Files

| File | Purpose |
|---|---|
| `src/core_estimation.py` | PAF calculation, uncertainty propagation, scenarios, and sensitivity analysis |
| `src/run_review_example.py` | Example analysis run |
