import os
import subprocess
import sys

import pytest

from attrex.paths import ROOT, load_env


def test_load_env_keeps_existing_and_skips_empty(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# комментарий\nATTREX_T_NEW='value'\nexport ATTREX_T_SET=from_file\nATTREX_T_EMPTY=\n",
                   encoding="utf-8")
    monkeypatch.delenv("ATTREX_T_NEW", raising=False)
    monkeypatch.delenv("ATTREX_T_EMPTY", raising=False)
    monkeypatch.setenv("ATTREX_T_SET", "from_env")
    assert load_env(env) == ["ATTREX_T_NEW"]
    assert os.environ["ATTREX_T_NEW"] == "value"
    assert os.environ["ATTREX_T_SET"] == "from_env"        # окружение важнее файла
    assert "ATTREX_T_EMPTY" not in os.environ
    monkeypatch.delenv("ATTREX_T_NEW")


def test_import_does_not_touch_environment():
    """Импорт attrex не читает .env: окружение меняют только точки входа, явно вызвав load_env()."""
    env = ROOT / ".env"
    if env.exists():
        pytest.skip("в корне есть свой .env — не трогаем его")
    env.write_text("ATTREX_T_IMPORT=1\n", encoding="utf-8")
    try:
        code = "import os, attrex.pipeline; print('ATTREX_T_IMPORT' in os.environ)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={k: v for k, v in os.environ.items() if k != "ATTREX_T_IMPORT"})
    finally:
        env.unlink()
    assert out.stdout.strip() == "False"


def test_env_example_lists_api_variables():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_BASE_URL=" in text and "OPENAI_API_KEY=" in text
