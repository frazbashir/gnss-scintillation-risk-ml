from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_risk.config import load_config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spark job for GNSS risk modeling")
    parser.add_argument("--processed-csv", required=True, help="Path to processed/labeled CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for Spark results")
    parser.add_argument("--config", default=None, help="Optional JSON config path")
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan without Spark runtime")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    train_fraction = float(config["training"]["train_fraction"]) if config else 0.7

    if args.dry_run:
        print("Spark job dry run")
        print(f"processed_csv={args.processed_csv}")
        print(f"output_dir={args.output_dir}")
        print(f"train_fraction={train_fraction}")
        return

    try:
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
    except ImportError as exc:
        raise RuntimeError(
            "pyspark is not available. Install requirements.txt or run on GDX Spark cluster."
        ) from exc

    spark = SparkSession.builder.appName("gnss-risk-ml").getOrCreate()

    df = spark.read.option("header", True).option("inferSchema", True).csv(args.processed_csv)
    if "label" not in df.columns:
        raise ValueError("Processed CSV must include 'label' column")

    excluded = {"timestamp", "label", "position_error_m"}

    numeric_types = {
        "double",
        "float",
        "int",
        "bigint",
        "smallint",
        "tinyint",
        "decimal",
        "long",
    }

    feature_cols = []
    for col_name, dtype in df.dtypes:
        dtype_l = dtype.lower()
        if col_name in excluded:
            continue
        if col_name.startswith("vpl") or col_name.startswith("d_vpl") or "_vpl_" in col_name:
            continue
        if any(dtype_l.startswith(t) for t in numeric_types):
            feature_cols.append(col_name)

    if not feature_cols:
        raise ValueError("No usable numeric feature columns found in processed CSV")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    model_df = assembler.transform(df).select("timestamp", "label", "features")

    row_window = Window.orderBy("timestamp")
    model_df = model_df.withColumn("row_idx", F.row_number().over(row_window))
    total_rows = model_df.count()
    split_idx = max(1, min(total_rows - 1, int(total_rows * train_fraction)))

    train_df = model_df.filter(F.col("row_idx") <= split_idx)
    test_df = model_df.filter(F.col("row_idx") > split_idx)

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=80)
    lr_model = lr.fit(train_df)

    pred = lr_model.transform(test_df)

    evaluator_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )

    auc_roc = evaluator_roc.evaluate(pred)
    auc_pr = evaluator_pr.evaluate(pred)

    prob_col = F.col("probability").getItem(1)
    pred_out = pred.select(
        "timestamp",
        F.col("label").cast("int").alias("target_label"),
        prob_col.alias("probability"),
        (prob_col >= F.lit(0.5)).cast("int").alias("predicted_label"),
    )

    output_dir = args.output_dir.rstrip("/")
    pred_out.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/predictions")

    metrics_df = spark.createDataFrame(
        [
            {
                "samples": int(test_df.count()),
                "roc_auc": float(auc_roc),
                "pr_auc": float(auc_pr),
                "feature_count": int(len(feature_cols)),
            }
        ]
    )
    metrics_df.coalesce(1).write.mode("overwrite").json(f"{output_dir}/metrics")

    features_df = spark.createDataFrame([{"feature": name} for name in feature_cols])
    features_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/features")

    print(json.dumps({"roc_auc": auc_roc, "pr_auc": auc_pr, "feature_count": len(feature_cols)}, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
