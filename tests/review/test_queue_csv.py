"""Очередь, сохранённая модератором: русский Excel пишет «;» и cp1251, колонки могут пропасть."""
import pandas as pd
import pytest

from attrex.review.decisions import read_filled_queue
from attrex.review.queue import QUEUE_COLUMNS, read_queue_csv


def queue_df() -> pd.DataFrame:
    row = {c: "" for c in QUEUE_COLUMNS}
    row.update(rank="1", sku_id="30000001", attr_id="color", value="черный", value_raw="Black", decision="accept")
    fix = dict(row, rank="2", attr_id="brand", value="xiaomi redmi", decision="fix", fixed_value="xiaomi")
    return pd.DataFrame([row, fix])


def test_russian_excel_semicolon_cp1251(tmp_path):
    path = tmp_path / "queue.csv"
    path.write_bytes(queue_df().to_csv(index=False, sep=";").encode("cp1251"))
    got = read_filled_queue(path, "run", "anna")
    assert list(got.decision) == ["accept", "fix"] and list(got.value) == ["черный", "xiaomi"]


def test_header_case_and_spaces(tmp_path):
    df = queue_df()
    df.columns = [f" {c.upper()} " for c in df.columns]
    path = tmp_path / "queue.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    assert len(read_queue_csv(path)) == 2


def test_missing_column_explains_how_to_save(tmp_path):
    path = tmp_path / "queue.csv"
    queue_df().drop(columns=["fixed_value"]).to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=r"нет колонок \['fixed_value'\].*CSV UTF-8"):
        read_filled_queue(path, "run", "anna")


def test_missing_file(tmp_path):
    with pytest.raises(ValueError, match="нет файла"):
        read_queue_csv(tmp_path / "nope.csv")
