import numpy as np
import pytest

from attrex.confidence.calibration import Curve, IsotonicCalibrator, choose_thresholds, isotonic
from attrex.contracts import Step


def test_isotonic_is_monotone_and_pools_violators():
    xs, ys = isotonic([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    assert ys == sorted(ys)
    assert ys == [0.0, 0.5, 1.0] and xs == pytest.approx([0.1, 0.25, 0.4])
    xs, ys = isotonic([0.1, 0.2, 0.3], [1, 1, 1])
    assert ys == [1.0] and xs == pytest.approx([0.2])                     # одинаковые уровни сливаются
    assert isotonic([0.1, 0.2], [1, 0], w=[3, 1])[1] == [0.75]


def test_curve_is_compact_on_many_points():
    rng = np.random.default_rng(1)
    raw = rng.random(5000)
    curve = Curve.fit(raw, rng.random(5000) < raw)                      # вероятность верного = сырой скор
    assert len(curve.xs) <= 20 and curve.n == 5000
    assert curve(0.9) > curve(0.1) and abs(curve(0.5) - 0.5) < 0.1


def test_curve_interpolates_and_clips():
    c = Curve((0.2, 0.8), (0.1, 0.9), n=10)
    assert c(0.5) == pytest.approx(0.5)
    assert c(0.0) == pytest.approx(0.1) and c(1.0) == pytest.approx(0.9)


def make_rows():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):            # правила по бренду: верны почти всегда, сырой скор ~0.72
        rows.append(("brand", "rules", 0.72, bool(rng.random() < 0.97)))
    for _ in range(100):            # retrieval по цвету: часто ошибается при высоком сыром скоре
        rows.append(("color", "retrieval", 0.8, bool(rng.random() < 0.3)))
    for _ in range(20):             # мало данных: кривая атрибута, а не пары (атрибут, шаг)
        rows.append(("color", "parse", 0.6, True))
    return rows


def test_fit_per_step_with_fallback():
    cal = IsotonicCalibrator.fit(make_rows(), min_n=50)
    assert cal("brand", Step.RULES, 0.72) > 0.9                  # сырой 0.72 → правила по бренду надёжны
    assert cal("color", Step.RETRIEVAL, 0.8) < 0.5
    assert "color/parse" not in cal.curves and "color" in cal.curves
    assert cal("weight", Step.RULES, 0.5) == pytest.approx(cal.curves["*"](0.5))   # неизвестный атрибут — общая кривая


def test_thresholds_reach_target_precision():
    rows = [("brand", 0.95, True)] * 90 + [("brand", 0.4, False)] * 20 + [("color", 0.3, False)] * 40
    th = choose_thresholds(rows, target_precision=0.9, default=0.75)
    assert th["brand"] == 0.95                    # порог 0.4 взял бы все 110 значений: точность 82% < 90%
    assert th["color"] > 0.3                      # точности 90% не достичь: не принимаем ничего
    loose = [("brand", 0.95, True)] * 90 + [("brand", 0.4, False)] * 10
    assert choose_thresholds(loose, 0.9, 0.75, floor=0.0)["brand"] == 0.4   # ровно 90% на всех — порог ниже
    assert choose_thresholds(loose, 0.9, 0.75)["brand"] == 0.5               # но не ниже floor


def test_save_load_roundtrip(tmp_path):
    cal = IsotonicCalibrator.fit(make_rows(), min_n=50)
    cal.thresholds = {"brand": 0.8}
    cal.meta = {"fold": "valid"}
    path = tmp_path / "calibration.yaml"
    cal.save(path)
    loaded = IsotonicCalibrator.load(path)
    assert loaded.thresholds == {"brand": 0.8} and loaded.meta["fold"] == "valid"
    assert loaded("brand", Step.RULES, 0.72) == pytest.approx(cal("brand", Step.RULES, 0.72), abs=1e-3)
