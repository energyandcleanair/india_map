"""Run all the necessary steps for processing PM2.5 data in a single environment."""

from runpy import run_module


def _main() -> None:
    run_module("pm25ml.run.s01_fetch_and_combine", run_name="__main__")
    run_module("pm25ml.run.s02_generate_features", run_name="__main__")
    run_module("pm25ml.run.s03_sample_for_imputation", run_name="__main__")
    run_module("pm25ml.run.s04_train_aod_imputer", run_name="__main__")
    run_module("pm25ml.run.s04_train_co_imputer", run_name="__main__")
    run_module("pm25ml.run.s04_train_no2_imputer", run_name="__main__")
    run_module("pm25ml.run.s05_impute", run_name="__main__")
    run_module("pm25ml.run.s06_prep_for_full_model", run_name="__main__")
    run_module("pm25ml.run.s07_train_full_model", run_name="__main__")


if __name__ == "__main__":
    _main()
