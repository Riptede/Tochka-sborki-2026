"""Сборка data/attrex.duckdb из файлов организатора (блок b1).

Порядок: feed (skus.pkl → parquet) → raw_* → core_attr_name_map → core_cards → core_gold → core_ref_values →
core_gold.in_text → core_category_schema → core_splits → core_traffic_weights.

БД собирается во временный файл и в конце атомарно заменяет основной: читатели никогда не видят недостроенную БД.
Всё тяжёлое делает DuckDB с лимитом памяти; Python обрабатывает только уникальные значения атрибутов.
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from attrex import paths
from attrex.db.feed import feed_cached, parse_feed
from attrex.db.in_text import Locator, locate_chunks
from attrex.normalize.units import canon_model, canon_text, to_canonical_number
from attrex.registry import Registry, load_registry

SCHEMA_SQL = Path(__file__).with_name("schema.sql")

# составной фасет «серый/серебристый», «полипропилен/резина» — несколько значений эталона
SPLIT_COMPOSITE = ("color", "material")
# шаблонные фразы продавцов, которые не несут характеристик
BOILERPLATE = ["Лучшие товары из Китая"]
# порядок важен: сначала &amp; (бывает двойное кодирование «&amp;laquo;»), потом остальные, в том числе &lt;/&gt;:
# у части продавцов теги закодированы («&lt;br/&gt;») и удаляются только после раскодирования
HTML_ENTITIES = {"&amp;": "&", "&nbsp;": " ", "&quot;": '"', "&laquo;": "«", "&raquo;": "»", "&mdash;": "—",
                 "&ndash;": "–", "&lt;": "<", "&gt;": ">", "&#39;": "'"}


@dataclass
class BuildConfig:
    raw_dir: Path = paths.RAW
    interim_dir: Path = paths.INTERIM
    db_path: Path = paths.DB
    reparse: bool = False
    fast_pickle: bool = False
    memory_limit: str = "3GB"
    threads: int = 4


def _sql_path(p: Path) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


def _pipeline_cfg() -> dict:
    with open(paths.CONFIGS / "pipeline.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _Log:
    def __init__(self, log):
        self.log, self.t0 = log, time.time()

    def __call__(self, msg: str) -> None:
        self.log(f"[{time.time() - self.t0:6.0f} с] {msg}")


def load_raw(con: duckdb.DuckDBPyConnection, cfg: BuildConfig) -> None:
    i, r = cfg.interim_dir, cfg.raw_dir
    con.execute(f"""
        INSERT INTO raw_categories
        SELECT TRY_CAST(id AS INTEGER), TRY_CAST(parent_id AS INTEGER), name
        FROM read_parquet('{_sql_path(i / "categories.parquet")}')""")
    con.execute(f"""
        INSERT INTO raw_offers
        SELECT CAST(offer_id AS BIGINT), TRY_CAST(category_id AS INTEGER), vendor, model, name
        FROM read_parquet('{_sql_path(i / "offers.parquet")}')""")
    con.execute(f"""
        INSERT INTO raw_params
        SELECT CAST(offer_id AS BIGINT), param_name, value, unit
        FROM read_parquet('{_sql_path(i / "params.parquet")}')""")
    con.execute(f"""
        INSERT INTO raw_clicks_sku
        SELECT CAST(sku_id AS BIGINT), CAST(min(sku_seo_id) AS INTEGER), count(*)
        FROM read_parquet('{_sql_path(r / "query_clicks.parquet")}')
        GROUP BY 1""")


def build_name_map(con: duckdb.DuckDBPyConnection, reg: Registry) -> None:
    rows = []
    for a, s in reg.attrs.items():
        rows += [(n.strip(), a, "facet") for n in s.feed_facet]
        rows += [(n.strip(), a, "search") for n in s.feed_search]
    con.executemany("INSERT INTO core_attr_name_map VALUES (?, ?, ?)", rows)


def build_cards(con: duckdb.DuckDBPyConnection, cfg: BuildConfig) -> None:
    def lit(s: str) -> str:
        return "'" + s.replace("'", "''") + "'"

    ent = "description"
    for k, v in HTML_ENTITIES.items():
        ent = f"replace({ent}, {lit(k)}, {lit(v)})"
    boiler = "d"
    for b in BOILERPLATE:
        boiler = f"replace({boiler}, {lit(b)}, '')"
    con.execute(f"""
        CREATE TEMP TABLE desc_clean AS
        WITH a AS (
            SELECT CAST(sku_id AS BIGINT) AS sku_id, title, {ent} AS description
            FROM read_parquet('{_sql_path(cfg.raw_dir / "sku_desc.parquet")}')
        ), b AS (
            SELECT sku_id, title,
                   regexp_replace(description, '(?i)<br\\s*/?>|</p>|</li>|</div>', '\n', 'g') AS description
            FROM a
        ), c AS (
            SELECT sku_id, title, regexp_replace(description, '</?[a-zA-Z][^<>]*>', ' ', 'g') AS d FROM b
        )
        SELECT sku_id, trim(title) AS title,
               trim(regexp_replace(regexp_replace(regexp_replace({boiler}, '[ \t]+', ' ', 'g'),
                                                  '\n[ \t]+', '\n', 'g'), '\n{{3,}}', '\n\n', 'g')) AS description
        FROM c""")
    con.execute("""
        INSERT INTO core_cards
        WITH seller AS (
            SELECT sku_id, any_value(value) AS seller FROM raw_params WHERE param_name = 'Продавец' GROUP BY 1
        ), seller_n AS (
            SELECT seller, count(*) AS n FROM seller GROUP BY 1
        ), cards AS (
            SELECT coalesce(d.sku_id, o.sku_id) AS sku_id,
                   coalesce(d.title, o.name) AS title,
                   d.description,
                   o.sku_id IS NOT NULL AS has_offer,
                   o.category_id AS feed_cat
            FROM desc_clean d FULL OUTER JOIN raw_offers o ON d.sku_id = o.sku_id
        )
        SELECT c.sku_id, c.title, c.description, c.has_offer,
               CASE WHEN c.feed_cat IS NOT NULL THEN c.feed_cat
                    WHEN cat.category_id IS NOT NULL THEN k.seo_id END,
               CASE WHEN c.feed_cat IS NOT NULL THEN 'feed'
                    WHEN cat.category_id IS NOT NULL THEN 'clicks' END,
               CASE WHEN c.feed_cat IS NOT NULL THEN 1.0
                    WHEN cat.category_id IS NOT NULL THEN 0.97 END,   -- seo_id = categoryId в 97% (EDA, раздел 9)
               CASE WHEN s.seller = 'М.Видео' THEN '1P' WHEN s.seller IS NOT NULL THEN '3P' ELSE 'unknown' END,
               CASE WHEN sn.n IS NULL THEN NULL WHEN sn.n <= 10 THEN '1–10' WHEN sn.n <= 100 THEN '11–100'
                    WHEN sn.n <= 1000 THEN '101–1k' WHEN sn.n <= 10000 THEN '1k–10k' ELSE '>10k' END,
               regexp_matches(lower(replace(c.title, 'ё', 'е')), '^(восстановлен|уценен)'),
               coalesce(k.clicks, 0)
        FROM cards c
        LEFT JOIN raw_clicks_sku k ON k.sku_id = c.sku_id
        LEFT JOIN raw_categories cat ON cat.category_id = k.seo_id
        LEFT JOIN seller s ON s.sku_id = c.sku_id
        LEFT JOIN seller_n sn ON sn.seller = s.seller""")
    con.execute("DROP TABLE desc_clean")


def dedupe_tokens(s: str | None) -> str | None:
    """Повтор в записи продавца: «apple m2 max apple m2 max 3.5 ггц» → «apple m2 max 3.5 ггц»,
    «intel core i3 i3-1115g4» → «intel core i3-1115g4»."""
    if s is None:
        return None
    toks = s.split()
    changed = True
    while changed:
        changed = False
        for n in range(len(toks) // 2, 0, -1):
            for i in range(len(toks) - 2 * n + 1):
                if toks[i:i + n] == toks[i + n:i + 2 * n]:
                    del toks[i + n:i + 2 * n]
                    changed = True
                    break
            if changed:
                break
    toks = [t for i, t in enumerate(toks) if not (i + 1 < len(toks) and toks[i + 1].startswith(t + "-"))]
    return " ".join(toks)


def _canon_str(attr_id: str, value: str | None) -> str | None:
    # модель в фиде записана с памятью, цветом и артикулом — в эталон идёт только название модели
    if attr_id == "model":
        return canon_model(value)
    return dedupe_tokens(canon_text(value)) if attr_id == "cpu" else canon_text(value)


def _canon_values(con: duckdb.DuckDBPyConnection, reg: Registry) -> pd.DataFrame:
    """Уникальные (attr_id, layer, value, unit) → каноническое значение. Python видит только уникальные значения."""
    distinct = con.execute("""
        SELECT DISTINCT m.attr_id, m.layer, p.value, coalesce(p.unit, '') AS unit
        FROM raw_params p JOIN core_attr_name_map m ON trim(p.param_name) = m.param_name""").df()
    canon, ph = [], []
    for a, value, unit in zip(distinct.attr_id, distinct.value, distinct.unit):
        spec = reg[a]
        is_ph = reg.is_placeholder(value)
        ph.append(is_ph)
        if is_ph:
            canon.append(None)
        elif spec.kind == "number":
            canon.append(to_canonical_number(value, unit or None, spec.unit))
        else:
            canon.append(_canon_str(a, f"{value} {unit}" if unit else value))
    distinct["value_canon"] = canon
    distinct["is_placeholder"] = ph
    return distinct


def build_gold(con: duckdb.DuckDBPyConnection, reg: Registry, log) -> None:
    canon = _canon_values(con, reg)
    con.register("canon_df", canon)
    kinds = pd.DataFrame({"attr_id": list(reg.attrs), "kind": [s.kind for s in reg.attrs.values()],
                          "unit": [s.unit for s in reg.attrs.values()]})
    con.register("kinds_df", kinds)
    # params → значения в каноне; для числа берём точное _search, для категории — фасет (канон витрины)
    con.execute("""
        CREATE TEMP TABLE gold_params AS
        WITH p AS (
            SELECT p.sku_id, m.attr_id, m.layer, p.value, p.unit, c.value_canon, c.is_placeholder
            FROM raw_params p
            JOIN core_attr_name_map m ON trim(p.param_name) = m.param_name
            JOIN canon_df c ON c.attr_id = m.attr_id AND c.layer = m.layer AND c.value = p.value
                           AND c.unit = coalesce(p.unit, '')
        ), agg AS (
            SELECT sku_id, attr_id,
                   list(DISTINCT value_canon) FILTER (WHERE layer = 'facet' AND value_canon IS NOT NULL) AS f_vals,
                   list(DISTINCT value_canon) FILTER (WHERE layer = 'search' AND value_canon IS NOT NULL) AS s_vals,
                   string_agg(DISTINCT value || coalesce(' ' || unit, ''), ' | ') FILTER (WHERE layer = 'search') AS raw_s,
                   string_agg(DISTINCT value, ' | ') FILTER (WHERE layer = 'facet') AS raw_f,
                   bool_or(is_placeholder) AS any_ph
            FROM p GROUP BY 1, 2
        )
        SELECT a.*, k.kind, k.unit,
               CASE WHEN k.kind = 'number'
                    THEN CASE WHEN len(coalesce(s_vals, [])) > 0 THEN s_vals ELSE f_vals END
                    ELSE CASE WHEN len(coalesce(f_vals, [])) > 0 THEN f_vals ELSE s_vals END END AS chosen
        FROM agg a JOIN kinds_df k USING (attr_id)""")
    # поля оффера (vendor, model) — приоритетнее одноимённых параметров
    field_rows = []
    for a, s in reg.attrs.items():
        if s.feed_field:
            field_rows.append(f"""
                SELECT sku_id, '{a}' AS attr_id, {s.feed_field} AS raw FROM raw_offers WHERE {s.feed_field} IS NOT NULL""")
    con.execute(f"CREATE TEMP TABLE gold_fields AS {' UNION ALL '.join(field_rows)}")
    fields = con.execute("SELECT DISTINCT attr_id, raw FROM gold_fields").df()
    fields["value_canon"] = [_canon_str(a, v) for a, v in zip(fields.attr_id, fields.raw)]
    # pandas хранит None как NaN: проверяем через isna. Модель из одного цвета («Black») — тоже заглушка
    fields["is_placeholder"] = [reg.is_placeholder(v) or pd.isna(c) for v, c in zip(fields.raw, fields.value_canon)]
    con.register("field_canon", fields)
    con.execute("""
        INSERT INTO core_gold
        SELECT f.sku_id, f.attr_id, CASE WHEN c.is_placeholder THEN NULL ELSE c.value_canon END,
               NULL, f.raw, NULL, c.is_placeholder, NULL
        FROM gold_fields f JOIN field_canon c USING (attr_id, raw)""")
    placeholders = sorted({canon_text(p) for p in reg.placeholders} - {None})
    con.execute("""
        -- составной фасет «серый/серебристый» — несколько значений: верно любое из них
        INSERT INTO core_gold
        SELECT DISTINCT g.sku_id, g.attr_id, trim(p.part), CASE WHEN g.kind = 'number' THEN g.unit END,
               g.raw_s, g.raw_f, false, NULL
        FROM gold_params g, unnest(g.chosen) AS v(value),
             unnest(CASE WHEN g.attr_id IN (SELECT unnest($split)) THEN string_split(v.value, '/')
                         ELSE [v.value] END) AS p(part)
        WHERE len(coalesce(g.chosen, [])) > 0 AND trim(p.part) <> '' AND trim(p.part) NOT IN (SELECT unnest($ph))
          AND NOT EXISTS (SELECT 1 FROM gold_fields f WHERE f.sku_id = g.sku_id AND f.attr_id = g.attr_id)""",
                {"split": list(SPLIT_COMPOSITE), "ph": placeholders})
    con.execute("""
        -- только заглушки: эталон есть, но это пропуск; метрики не считают его ошибкой модели
        INSERT INTO core_gold
        SELECT g.sku_id, g.attr_id, NULL, NULL, g.raw_s, g.raw_f, true, NULL
        FROM gold_params g
        WHERE len(coalesce(g.chosen, [])) = 0 AND g.any_ph
          AND NOT EXISTS (SELECT 1 FROM gold_fields f WHERE f.sku_id = g.sku_id AND f.attr_id = g.attr_id)""")
    dropped = con.execute("""
        SELECT attr_id, count(*) FROM gold_params
        WHERE len(coalesce(chosen, [])) = 0 AND NOT any_ph GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    if dropped:
        log("эталон без канона (диапазоны фасетов и непереводимые единицы): "
            + ", ".join(f"{a} {n:,}" for a, n in dropped))
    for t in ("gold_params", "gold_fields"):
        con.execute(f"DROP TABLE {t}")
    for v in ("canon_df", "field_canon", "kinds_df"):
        con.unregister(v)


def build_ref_values(con: duckdb.DuckDBPyConnection, reg: Registry) -> None:
    cats = [a for a, s in reg.attrs.items() if s.kind == "category"]
    con.execute(f"""
        INSERT INTO core_ref_values
        WITH raw_search AS (
            SELECT DISTINCT p.sku_id, m.attr_id,
                   lower(trim(regexp_replace(replace(p.value, 'ё', 'е'), '\\s*/\\s*', '/', 'g'))) AS raw
            FROM raw_params p JOIN core_attr_name_map m ON trim(p.param_name) = m.param_name
            WHERE m.layer = 'search'
        ), own AS (     -- все значения эталона карточки: у составного фасета сырое «черный» — не синоним «серый»
            SELECT sku_id, attr_id, list(value) AS vals FROM core_gold WHERE value IS NOT NULL GROUP BY 1, 2
        )
        SELECT g.attr_id, g.value,
               coalesce(list(DISTINCT s.raw) FILTER (WHERE s.raw IS NOT NULL AND NOT list_contains(o.vals, s.raw)), []),
               count(DISTINCT g.sku_id)
        FROM core_gold g JOIN own o USING (sku_id, attr_id)
        LEFT JOIN raw_search s ON s.sku_id = g.sku_id AND s.attr_id = g.attr_id
        WHERE g.value IS NOT NULL AND g.attr_id IN ({", ".join(f"'{a}'" for a in cats)})
        GROUP BY 1, 2""")


def _synonyms(con: duckdb.DuckDBPyConnection) -> dict[tuple[str, str], list[str]]:
    """Написания значения: сырые _search из фида (core_ref_values) и ручные синонимы configs/normalize.yaml."""
    out: dict[tuple[str, str], list[str]] = {}
    for a, v, syn in con.execute("SELECT attr_id, value, synonyms FROM core_ref_values").fetchall():
        out[(a, v)] = list(syn or [])
    manual = yaml.safe_load((paths.CONFIGS / "normalize.yaml").read_text(encoding="utf-8")).get("synonyms") or {}
    for a, spellings in manual.items():
        for spelling, canon in (spellings or {}).items():
            out.setdefault((a, canon_text(str(canon))), []).append(str(spelling))
    return out


def build_in_text(con: duckdb.DuckDBPyConnection, reg: Registry, log, workers: int = 1, chunk: int = 10_000) -> None:
    """core_gold.in_text: где в тексте карточки лежит значение эталона (db/in_text.py). Карточки — пачками по sku_id,
    пачки разбираются в workers процессах: основа каждого слова описания — самое долгое в сборке."""
    locator = Locator({a: (s.kind, s.unit) for a, s in reg.attrs.items()}, _synonyms(con))
    skus = [r[0] for r in con.execute(
        "SELECT DISTINCT sku_id FROM core_gold WHERE value IS NOT NULL ORDER BY 1").fetchall()]
    con.execute("CREATE TEMP TABLE gold_in_text (sku_id BIGINT, attr_id VARCHAR, value VARCHAR, in_text VARCHAR)")

    def chunks():
        for i in range(0, len(skus), chunk):
            lo, hi = skus[i], skus[min(i + chunk, len(skus)) - 1]
            cards = con.execute("SELECT sku_id, title, coalesce(description, '') FROM core_cards "
                                "WHERE sku_id BETWEEN ? AND ?", [lo, hi]).fetchall()
            rows = con.execute("SELECT DISTINCT sku_id, attr_id, value FROM core_gold "
                               "WHERE value IS NOT NULL AND sku_id BETWEEN ? AND ?", [lo, hi]).fetchall()
            yield cards, rows

    for located in locate_chunks(locator, chunks(), workers):
        part = pd.DataFrame(located, columns=["sku_id", "attr_id", "value", "in_text"])
        con.register("in_text_part", part)
        con.execute("INSERT INTO gold_in_text SELECT * FROM in_text_part")
        con.unregister("in_text_part")
    con.execute("""
        UPDATE core_gold g SET in_text = t.in_text FROM gold_in_text t
        WHERE g.sku_id = t.sku_id AND g.attr_id = t.attr_id AND g.value = t.value""")
    con.execute("DROP TABLE gold_in_text")
    found = con.execute("""
        SELECT round(100 * avg((in_text IN ('title', 'description', 'synonym'))::INT), 1)
        FROM core_gold WHERE value IS NOT NULL""").fetchone()[0]
    log(f"значение эталона есть в тексте карточки: {found}% строк")


def build_category_schema(con: duckdb.DuckDBPyConnection, reg: Registry, pcfg: dict) -> None:
    thr = pcfg["schema"]["required_fill_rate"]
    min_n = pcfg["schema"]["min_offers_for_own_schema"]
    own = con.execute("""
        WITH sizes AS (SELECT category_id, count(*) AS n FROM raw_offers GROUP BY 1)
        SELECT o.category_id, g.attr_id, count(DISTINCT o.sku_id) / any_value(s.n) AS fill, any_value(s.n) AS n
        FROM core_gold g JOIN raw_offers o USING (sku_id) JOIN sizes s USING (category_id)
        WHERE g.value IS NOT NULL GROUP BY 1, 2""").df()
    parent = con.execute("""
        WITH oc AS (SELECT o.sku_id, c.parent_id FROM raw_offers o JOIN raw_categories c USING (category_id)),
             sizes AS (SELECT parent_id, count(*) AS n FROM oc GROUP BY 1)
        SELECT oc.parent_id, g.attr_id, count(DISTINCT oc.sku_id) / any_value(s.n) AS fill
        FROM core_gold g JOIN oc USING (sku_id) JOIN sizes s USING (parent_id)
        WHERE g.value IS NOT NULL GROUP BY 1, 2""").df()
    sizes = con.execute("""
        SELECT o.category_id, any_value(c.parent_id) AS parent_id, count(*) AS n
        FROM raw_offers o LEFT JOIN raw_categories c USING (category_id)
        WHERE o.category_id IS NOT NULL GROUP BY 1""").df()
    own_fill = {(r.category_id, r.attr_id): r.fill for r in own.itertuples()}
    par_fill = {(r.parent_id, r.attr_id): r.fill for r in parent.itertuples()}
    rows = []
    for r in sizes.itertuples():
        cid, pid = int(r.category_id), (None if pd.isna(r.parent_id) else int(r.parent_id))
        inherit = r.n < min_n and pid is not None
        src = par_fill if inherit else own_fill
        key = pid if inherit else cid
        tz = set(reg.required.get(cid, ()))
        for a in reg.attrs:
            fill = float(src.get((key, a), 0.0))
            if fill > 0 or a in tz:
                # явный список из ТЗ важнее правила заполненности; для остальных категорий — fill ≥ порога
                required = a in tz if tz else fill >= thr
                rows.append((cid, a, fill, bool(required), pid if inherit else None))
    con.executemany("INSERT INTO core_category_schema VALUES (?, ?, ?, ?, ?)", rows)


def build_splits(con: duckdb.DuckDBPyConnection, pcfg: dict) -> None:
    s = pcfg["split"]
    seed, test, valid, hold = s["seed"], s["test_share"], s["valid_share"], s["holdout_category_share"]
    # группа = бренд + модель без памяти и цвета: варианты одной модели не расходятся по train и test
    models = con.execute("SELECT DISTINCT coalesce(model, name, '') AS raw FROM raw_offers").df()
    models["model_key"] = [canon_model(v) or "" for v in models.raw]
    con.register("model_keys", models)
    con.execute(f"""
        INSERT INTO core_splits
        WITH base AS (
            SELECT o.sku_id, o.category_id,
                   coalesce(lower(o.vendor), '') || '|' || m.model_key AS model_key,
                   md5(d.description) AS desc_hash
            FROM raw_offers o
            JOIN model_keys m ON m.raw = coalesce(o.model, o.name, '')
            LEFT JOIN core_cards d USING (sku_id)
        ), grp AS (
            -- дубли описаний попадают в одну группу с вариантами модели
            SELECT sku_id, category_id,
                   CASE WHEN desc_hash IS NULL THEN model_key
                        ELSE min(model_key) OVER (PARTITION BY desc_hash) END AS group_key
            FROM base
        )
        SELECT sku_id,
               CASE WHEN md5_number(category_id::VARCHAR || ':{seed}') % 1000 < {int(hold * 1000)} THEN 'holdout_category'
                    WHEN md5_number(group_key || ':{seed}') % 1000 < {int(test * 1000)} THEN 'test'
                    WHEN md5_number(group_key || ':{seed}') % 1000 < {int((test + valid) * 1000)} THEN 'valid'
                    ELSE 'train' END,
               group_key
        FROM grp""")
    con.unregister("model_keys")


def build_traffic_weights(con: duckdb.DuckDBPyConnection) -> None:
    """v0: вес категории — доля кликов, внутри категории вес делится поровну между обязательными атрибутами.

    Веса атрибутов по упоминаниям в запросах (EDA, раздел 8.1) — следующая версия, за неё отвечает b4.
    """
    con.execute("""
        INSERT INTO core_traffic_weights
        WITH cat AS (
            SELECT category_id, sum(clicks) AS clicks FROM core_cards
            WHERE category_id IS NOT NULL GROUP BY 1
        ), req AS (
            SELECT category_id, attr_id, count(*) OVER (PARTITION BY category_id) AS k
            FROM core_category_schema WHERE required
        )
        SELECT r.category_id, r.attr_id, c.clicks / (SELECT sum(clicks) FROM cat) / r.k
        FROM req r JOIN cat c USING (category_id)
        WHERE c.clicks > 0""")


def summary(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    tables = [r[0] for r in con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_type = 'BASE TABLE' AND table_schema = 'main' ORDER BY 1""").fetchall()]
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}


def build(cfg: BuildConfig | None = None, log=print) -> dict[str, int]:
    cfg = cfg or BuildConfig()
    log = _Log(log)
    reg, pcfg = load_registry(), _pipeline_cfg()

    if cfg.reparse or not feed_cached(cfg.interim_dir):
        log(f"разбор {cfg.raw_dir / 'skus.pkl'} ({'целиком' if cfg.fast_pickle else 'потоково'})")
        n_off, n_cat = parse_feed(cfg.raw_dir / "skus.pkl", cfg.interim_dir, fast=cfg.fast_pickle, log=log)
        log(f"фид: {n_off:,} офферов, {n_cat:,} категорий")
    else:
        log(f"фид из кэша {cfg.interim_dir}")

    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.db_path.with_name(cfg.db_path.name + ".tmp")
    for p in (tmp, tmp.with_name(tmp.name + ".wal")):
        p.unlink(missing_ok=True)
    # новый формат хранения: в нём DuckDB сжимает длинные тексты (zstd); в формате по умолчанию описания
    # лежат без сжатия и БД весит в 2–3 раза больше. Файл читается DuckDB той же версии (зафиксирована в uv.lock).
    con = duckdb.connect(str(tmp), config={"storage_compatibility_version": "latest"})
    try:
        con.execute(f"SET memory_limit = '{cfg.memory_limit}'")
        con.execute(f"SET threads = {int(cfg.threads)}")
        con.execute("SET preserve_insertion_order = false")
        con.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
        steps = [
            ("raw_*", lambda: load_raw(con, cfg)),
            ("core_attr_name_map", lambda: build_name_map(con, reg)),
            ("core_cards", lambda: build_cards(con, cfg)),
            ("core_gold", lambda: build_gold(con, reg, log)),
            ("core_ref_values", lambda: build_ref_values(con, reg)),
            ("core_gold.in_text", lambda: build_in_text(con, reg, log, workers=cfg.threads)),
            ("core_category_schema", lambda: build_category_schema(con, reg, pcfg)),
            ("core_splits", lambda: build_splits(con, pcfg)),
            ("core_traffic_weights", lambda: build_traffic_weights(con)),
        ]
        for name, fn in steps:
            fn()
            log(f"готово: {name}")
        counts = summary(con)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    os.replace(tmp, cfg.db_path)
    log(f"БД: {cfg.db_path} ({cfg.db_path.stat().st_size / 2**20:,.0f} МБ)")
    return counts
