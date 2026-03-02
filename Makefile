PYTHONPATH=src
PYTHON=python3

.PHONY: run test spark-dry plot all

run:
	$(PYTHONPATH) $(PYTHON) scripts/run_pipeline.py --config configs/storm_may2024.json --generate-synthetic

test:
	$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p "test_*.py"

spark-dry:
	$(PYTHONPATH) $(PYTHON) -m gnss_risk.spark_job --dry-run --processed-csv outputs/latest/dataset_labeled.csv --output-dir outputs/spark --config configs/storm_may2024.json

plot:
	$(PYTHONPATH) $(PYTHON) scripts/plot_metrics_and_comparison.py --run-dir outputs/latest

all:
	bash scripts/run_all_with_plots.sh
