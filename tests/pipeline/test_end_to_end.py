"""Сквозной прогон на синтетических данных b1: сборка БД → каскад → pred_* → метрики → файл сдачи."""
import duckdb
import pandas as pd

from attrex.contracts import PREDICTION_COLUMNS, BuildContext
from attrex.db.build import BuildConfig, build
from attrex.metrics.report import RunInfo, render
from attrex.metrics.scoring import comparison_same, evaluate, load_outcomes
from attrex.pipeline import build_pipeline, load_config
from attrex.runs import RunSpec, export, run, store

ALL_FOLDS = ("train", "valid", "test", "holdout_category")


def test_pipeline_end_to_end(tmp_path, raw_dir):
    db = tmp_path / "attrex.duckdb"
    build(BuildConfig(raw_dir=raw_dir, interim_dir=tmp_path / "interim", db_path=db, memory_limit="512MB", threads=1),
          log=lambda _: None)
    cfg = load_config()
    spec = RunSpec("20261018-0000-test", ("rules", "parse"), knowledge_folds=ALL_FOLDS)

    con = duckdb.connect(str(db), read_only=True)
    pipeline = build_pipeline(BuildContext(con, cfg, ALL_FOLDS), spec.steps)
    assert pipeline.step_names == ["rules", "parse"]
    result = run(pipeline, spec, con, tmp_path / "outputs")
    con.close()
    assert result.n_cards == 5
    store(db, tmp_path / "outputs" / spec.run_id)

    con = duckdb.connect(str(db), read_only=True)
    preds = con.execute("SELECT * FROM pred_predictions").df()
    assert set(preds.run_id) == {spec.run_id}
    assert not preds.duplicated(["sku_id", "attr_id"]).any()           # одна строка на пару
    brand = preds.set_index(["sku_id", "attr_id"]).loc[(30000001, "brand")]
    assert (brand.value, brand.evidence, brand.step) == ("samsung", "Samsung", "rules")
    assert set(preds.status) <= {"accepted", "review", "empty"}

    df = load_outcomes(con, ALL_FOLDS, comparison_same(), run_id=spec.run_id)
    metrics = evaluate(df)
    assert metrics.overall["gold"] > 0 and metrics.overall["P"] > 0
    report = render(RunInfo(spec.run_id, spec.steps, ALL_FOLDS, result.n_cards), metrics, [])
    assert "## По атрибутам" in report and "brand" in report

    files = export(con, spec.run_id, tmp_path / "submit")
    con.close()
    sub = pd.read_parquet(files[0])
    assert list(sub.columns) == PREDICTION_COLUMNS and len(sub) == len(preds)
    assert files[1].read_text(encoding="utf-8").splitlines()[0] == ",".join(PREDICTION_COLUMNS)
