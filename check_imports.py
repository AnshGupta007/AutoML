import sys, traceback

mods = [
    "src.features.imputer",
    "src.models.trainer",
    "src.features.pipeline",
    "src.models.xgboost_model",
    "src.models.linear_model",
    "src.models.base_model",
    "src.models.ensemble",
    "src.monitoring.model_performance",
    "src.monitoring.retraining_trigger",
    "src.orchestration.steps",
]
for mod in mods:
    try:
        __import__(mod)
        print(f"OK  {mod}")
    except Exception:
        exc = traceback.format_exc().strip()
        # Print last 3 lines of traceback (most informative)
        lines = exc.splitlines()
        brief = " | ".join(lines[-3:])[:400]
        print(f"ERR {mod}: {brief}")

sys.stdout.flush()
