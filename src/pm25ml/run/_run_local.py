"""Run all the necessary steps for processing PM2.5 data in a single environment."""

from runpy import run_module


def _main() -> None:
    run_module("pm25ml.run.fetch_and_combine", run_name="__main__")
    run_module("pm25ml.run.generate_features", run_name="__main__")
    run_module("pm25ml.run.sample_for_imputation", run_name="__main__")
    run_module("pm25ml.run.train_aod_imputer", run_name="__main__")
    run_module("pm25ml.run.train_co_imputer", run_name="__main__")
    run_module("pm25ml.run.train_no2_imputer", run_name="__main__")
    run_module("pm25ml.run.impute", run_name="__main__")
    run_module("pm25ml.run.prep_for_full_model", run_name="__main__")
    run_module("pm25ml.run.train_full_model", run_name="__main__")


if __name__ == "__main__":
    _main()
