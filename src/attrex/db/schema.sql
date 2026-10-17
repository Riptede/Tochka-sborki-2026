-- Схема БД проекта (DuckDB, data/attrex.duckdb). Владелец: b1.
-- Префиксы: raw_ — данные организатора как есть; core_ — очищенные данные и эталон;
--           pred_ — результаты пайплайна; rev_ — ручная проверка.
-- attr_id везде из configs/attributes.yaml.

-- raw: разбор skus.pkl, sku_desc.parquet, query_clicks.parquet ----------------
CREATE TABLE IF NOT EXISTS raw_categories (
    category_id   INTEGER PRIMARY KEY,
    parent_id     INTEGER,
    name          VARCHAR
);

CREATE TABLE IF NOT EXISTS raw_offers (
    sku_id        BIGINT PRIMARY KEY,
    category_id   INTEGER,
    vendor        VARCHAR,
    model         VARCHAR,
    name          VARCHAR
);

CREATE TABLE IF NOT EXISTS raw_params (
    sku_id        BIGINT,
    param_name    VARCHAR,        -- как в фиде, включая суффикс _search и пробелы по краям
    value         VARCHAR,
    unit          VARCHAR         -- атрибут @unit: единица часто лежит здесь, а не в value
);

CREATE TABLE IF NOT EXISTS raw_descriptions (
    sku_id        BIGINT PRIMARY KEY,
    title         VARCHAR,
    description   VARCHAR         -- с HTML
);

-- клики агрегированы по (запрос, sku): полная таблица 31 млн строк читается из parquet напрямую
CREATE TABLE IF NOT EXISTS raw_clicks_sku (
    sku_id        BIGINT PRIMARY KEY,
    seo_id        INTEGER,        -- совпадает с category_id в 97% случаев
    clicks        BIGINT
);

-- core: очищенные данные --------------------------------------------------------
CREATE TABLE IF NOT EXISTS core_cards (
    sku_id          BIGINT PRIMARY KEY,
    title           VARCHAR,
    description     VARCHAR,       -- без HTML и шаблонных фраз
    has_offer       BOOLEAN,       -- есть ли карточка в фиде (есть ли эталон)
    category_id     INTEGER,
    category_source VARCHAR,       -- feed | clicks | title
    category_conf   DOUBLE,
    seller_seg      VARCHAR,       -- 1P | 3P | unknown — по полю «Продавец», не по диапазону ID
    seller_size     VARCHAR,       -- 1–10 | 11–100 | 101–1k | 1k–10k | >10k
    is_refurb       BOOLEAN,       -- восстановленный или уценённый: описание шаблонное
    clicks          BIGINT
);

-- эталон из фида, приведённый к attr_id и канону
CREATE TABLE IF NOT EXISTS core_gold (
    sku_id          BIGINT,
    attr_id         VARCHAR,
    value           VARCHAR,       -- канон: число в unit реестра или значение справочника
    unit            VARCHAR,
    value_raw       VARCHAR,       -- значение _search с приклеенной @unit
    facet_raw       VARCHAR,       -- значение фасета как в фиде
    is_placeholder  BOOLEAN,       -- заглушка: в метрике считается пропуском
    in_text         VARCHAR        -- title | description | synonym | partial | none (EDA, раздел 7)
);

-- соответствие имён полей фида и attr_id (фасет ↔ _search не всегда совпадают по имени)
CREATE TABLE IF NOT EXISTS core_attr_name_map (
    param_name      VARCHAR PRIMARY KEY,
    attr_id         VARCHAR,
    layer           VARCHAR        -- facet | search
);

-- схема категории: какие атрибуты обязательны
CREATE TABLE IF NOT EXISTS core_category_schema (
    category_id     INTEGER,
    attr_id         VARCHAR,
    fill_rate       DOUBLE,
    required        BOOLEAN,
    inherited_from  INTEGER        -- NULL, если схема своя
);

-- справочник допустимых значений
CREATE TABLE IF NOT EXISTS core_ref_values (
    attr_id         VARCHAR,
    value           VARCHAR,       -- каноническое значение
    synonyms        VARCHAR[],     -- известные написания, в том числе английские
    n_offers        INTEGER
);

-- веса атрибутов по трафику для метрики (EDA, раздел 8.1)
CREATE TABLE IF NOT EXISTS core_traffic_weights (
    category_id     INTEGER,
    attr_id         VARCHAR,
    weight          DOUBLE
);

-- разбиение для обучения и оценки
CREATE TABLE IF NOT EXISTS core_splits (
    sku_id          BIGINT PRIMARY KEY,
    fold            VARCHAR,       -- train | valid | test | holdout_category
    group_key       VARCHAR        -- vendor + модель без цвета; дубли описаний в одной группе
);

-- pred: результаты пайплайна ------------------------------------------------------
CREATE TABLE IF NOT EXISTS pred_predictions (
    run_id          VARCHAR,
    sku_id          BIGINT,
    category_id     INTEGER,
    category_conf   DOUBLE,
    attr_id         VARCHAR,
    value           VARCHAR,
    unit            VARCHAR,
    confidence      DOUBLE,
    status          VARCHAR,       -- accepted | review | empty
    step            VARCHAR,       -- rules | parse | retrieval | llm | model_match | none
    source          VARCHAR,       -- title | description | seller_attrs | other_card | none
    value_raw       VARCHAR,
    evidence        VARCHAR
);

CREATE TABLE IF NOT EXISTS pred_runs (
    run_id          VARCHAR PRIMARY KEY,
    started_at      TIMESTAMP,
    git_commit      VARCHAR,
    config          JSON,
    llm_calls       INTEGER,
    llm_cost_usd    DOUBLE
);

-- rev: ручная проверка ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS rev_decisions (
    sku_id          BIGINT,
    attr_id         VARCHAR,
    run_id          VARCHAR,
    decision        VARCHAR,       -- accept | fix | absent
    value           VARCHAR,       -- итоговое значение после решения
    reviewer        VARCHAR,
    decided_at      TIMESTAMP
);
