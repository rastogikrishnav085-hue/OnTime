# Round 5 changes (labels and polish only, no new features)

- The assistant's hosted engine is now labelled **Cloud AI** in the app (engine buttons, Settings, messages). Internal ids, the settings file and the GEMINI_API_KEY environment variable are unchanged, so existing keys keep working.
- A start-up splash (logo, spinner, progress bar) shows while the app loads. It fades out by itself and is static when the system asks for reduced motion.
- Report and presentation regenerated: headline stays ROC-AUC 0.717 on the newest real orders; 0.915 and 0.943 are labelled simulated / generated.
- Not done here: clean Windows launch and the full pytest run on the final package. Run run_tests.bat and the launcher on your laptop.
