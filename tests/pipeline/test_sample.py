"""Выборка sample/raw (реальные карточки для демо) собирается в БД и проходит каскад: формат не разошёлся с кодом."""
import duckdb
import pytest

from attrex.contracts import BuildContext
from attrex.db.build import BuildConfig, build
from attrex.paths import ROOT
from attrex.pipeline import build_pipeline, load_config
from attrex.runs import RunSpec, run

SAMPLE = ROOT / "sample" / "raw"


@pytest.mark.skipif(not (SAMPLE / "skus.pkl").exists(), reason="нет sample/raw")
def test_sample_builds_and_runs(tmp_path):
    db = tmp_path / "attrex.duckdb"
    counts = build(BuildConfig(raw_dir=SAMPLE, interim_dir=tmp_path / "interim", db_path=db, memory_limit="1GB",
                               threads=2), log=lambda _: None)
    assert counts["raw_offers"] == 570 and counts["core_cards"] == 810

    con = duckdb.connect(str(db), read_only=True)
    try:
        assert con.execute("SELECT count(DISTINCT fold) FROM core_splits").fetchone()[0] >= 3
        cfg = load_config()
        pipeline = build_pipeline(BuildContext(con, cfg, ("train",)), ("rules", "parse"))
        result = run(pipeline, RunSpec("sample", ("rules", "parse"), limit=200), con, tmp_path / "out")
    finally:
        con.close()
    assert result.n_cards == 200 and result.n_rows > 200
