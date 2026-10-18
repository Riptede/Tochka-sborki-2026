"""Фикстуры b4: синтетические файлы организатора (как у b1) и маленькие ресурсы нормализатора."""
import pytest

from attrex.normalize.resources import NormalizeResources
from tests.db.conftest import raw_dir  # noqa: F401 — синтетический фид, описания и клики из тестов b1


@pytest.fixture(scope="session")
def res() -> NormalizeResources:
    return NormalizeResources.from_rows([
        ("brand", "samsung", "samsung", 10), ("color", "черный", "черный", 10), ("color", "белый", "белый", 10),
    ])
