# Test run record — OnTime v6 integrated revision

- Python test command: `python -m pytest -q`
- Result: **220 passed**
- Runtime: 45.42 seconds
- Warnings: 2 model-serialization/version warnings involving XGBoost (installed 3.1.3 versus model metadata 3.4.1, and legacy pickle serialization warning).
- Environment: Linux sandbox, Python 3.13.
- Not performed: clean Windows install/launcher run, manual browser QA on target machine, live operational data validation.
- Note: a startup warmup diagnostic from the surrounding execution environment appeared before pytest output; pytest itself exited with code 0 and reported all 220 tests passed.
