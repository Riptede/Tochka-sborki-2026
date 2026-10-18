import copy
from pathlib import Path

import pytest
import yaml

from attrex.config import (
    ConfigError,
    validate_all,
    validate_attributes,
    validate_calibration,
    validate_normalize,
    validate_pipeline,
)
from attrex.paths import CONFIGS


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


PIPELINE = load("pipeline.yaml")


def broken(**changes) -> dict:
    """pipeline.yaml с правками: ключ — путь через __ («llm__budget_rub»)."""
    cfg = copy.deepcopy(PIPELINE)
    for path, value in changes.items():
        *parents, key = path.split("__")
        node = cfg
        for p in parents:
            node = node[p]
        node[key] = value
    return cfg


def test_committed_configs_are_valid():
    assert validate_all() == []


@pytest.mark.parametrize("changes, message", [
    ({"knowledge_folds": ["train", "test"]}, "knowledge_folds"),            # test — никогда не в словарях
    ({"confidence__accept_threshold": 1.5}, "confidence.accept_threshold"),
    ({"confidence__acept_threshold": 0.7}, "acept_threshold"),               # опечатка в ключе
    ({"schema__fallback_required": ["brand", "colour"]}, "colour"),          # attr_id не из реестра
    ({"cascade": {"rules": "attrex.normalize.no_such_module"}}, "no_such_module"),
    ({"cascade": {"magic": "attrex.normalize.rules"}}, "cascade.magic"),     # шаг не из Step
    ({"llm__enabled": True, "llm__budget_rub": 0}, "budget_rub"),
    ({"llm__model": "deepseek-ai/DeepSeek-V4-Pro"}, "нет цены"),
    ({"split__test_share": 0.9}, "сумма долей"),
])
def test_pipeline_errors_name_the_key(changes, message):
    with pytest.raises(ConfigError) as e:
        validate_pipeline(broken(**changes), Path("pipeline.yaml"))
    assert "pipeline.yaml" in str(e.value) and message in str(e.value)


def test_attributes_number_needs_unit():
    cfg = load("attributes.yaml")
    cfg["attributes"]["weight"].pop("unit")
    with pytest.raises(ConfigError, match="attributes.weight"):
        validate_attributes(cfg)
    cfg = load("attributes.yaml")
    cfg["categories"][761]["required"].append("colour")
    with pytest.raises(ConfigError, match="colour"):
        validate_attributes(cfg)


def test_normalize_unknown_attr():
    cfg = load("normalize.yaml")
    cfg["aliases"]["colour"] = ["цвет"]
    with pytest.raises(ConfigError, match="aliases"):
        validate_normalize(cfg)


def test_calibration_curve_must_be_monotone():
    cfg = {"thresholds": {"brand": 0.5}, "curves": {"*": {"n": 10, "x": [0.1, 0.5], "y": [0.9, 0.2]}}}
    with pytest.raises(ConfigError, match="неубывающей"):
        validate_calibration(cfg)
