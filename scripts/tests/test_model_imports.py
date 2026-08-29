from pathlib import Path
import importlib
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


MODULES = [
    "classes.models.hydra_original",
    "classes.models.optimised_hydra",
    "classes.models.hydra_explainable",
    "classes.models.lr_explainable",
    "classes.models.mrsqm_explainable",
]


def main():
    print("Project root:", PROJECT_ROOT)

    failed = []

    for module_name in MODULES:
        print(f"\nTesting import: {module_name}")

        try:
            module = importlib.import_module(module_name)

            public_names = [name for name in dir(module) if not name.startswith("_")]

            print(f"Imported: {module_name}")
            print("Public objects:")
            for name in public_names:
                print(f"{name}")

        except Exception:
            failed.append(module_name)
            print(f"FAILED: {module_name}")
            traceback.print_exc()

    print("\nImport test complete.")

    if failed:
        print("\nFailed modules:")
        for module_name in failed:
            print(f"  - {module_name}")
        raise SystemExit(1)

    print("All model imports passed.")


if __name__ == "__main__":
    main()