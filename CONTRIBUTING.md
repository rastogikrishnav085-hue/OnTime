# Working on OnTime

1. Fork or clone, then `setup.bat full` (Windows) or `pip install -r requirements.txt`.
2. Create a branch: `git checkout -b feature/short-name`.
3. Make the change, then run `python -m pytest tests -q` (or `run_tests.bat`). CI runs the same suite on Windows and Linux.
4. If you change the model, the features or the clustering, rebuild the app data with `rebuild_app_data.bat`
   (or `python export_portable_model.py && python backend/prepare_app_data.py`) and commit the updated files.
5. Open a pull request describing what changed and why.

House rules: no number in the UI without a source in the data; anything simulated is labelled "simulated";
every new recommendation rule gets a test.
