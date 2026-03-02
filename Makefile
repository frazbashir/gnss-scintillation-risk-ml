PYTHONPATH=src

.PHONY: run test spark-dry

run:
	$(PYTHONPATH) python3 scripts/run_pipeline.py --config configs/storm_may2024.json --generate-synthetic

test:
	$(PYTHONPATH) python3 -m unittest discover -s tests -p "test_*.py"

spark-dry:
	$(PYTHONPATH) python3 -m gnss_risk.spark_job --dry-run --processed-csv outputs/latest/dataset_labeled.csv --output-dir outputs/spark --config configs/storm_may2024.json
