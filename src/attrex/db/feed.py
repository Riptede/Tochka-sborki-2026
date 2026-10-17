"""Разбор skus.pkl (xmltodict-дамп YML-фида) в parquet: offers, params, categories.

Файл читается строго как данные: опкоды, которые создают объекты классов (GLOBAL, REDUCE и т. п.), запрещены.
Два режима:
- stream (по умолчанию) — мини-VM поверх pickletools.genops, офферы пишутся в parquet по мере чтения; ≈3 ГБ RAM;
- fast — DataOnlyUnpickler грузит файл целиком; ≈10 ГБ RAM, на ВМ хакатона не использовать.

Формат выходных файлов совпадает с кэшем ноутбука 01_eda.ipynb (data/interim/), поэтому кэш общий.
"""
import json
import pickle
import pickletools
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING"}
INT_OPS = {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "LONG"}
UNSAFE = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD", "PERSID", "BINPERSID"}
GET_OPS = {"BINGET", "LONG_BINGET", "GET"}
FLUSHED = object()  # заглушка вместо оффера, уже отданного писателю

OFFER_KEYS = {"@id", "@available", "vendor", "model", "name", "categoryId", "categories", "param"}
FEED_FILES = ("offers.parquet", "params.parquet", "categories.parquet")


class UnsafePickleError(RuntimeError):
    pass


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise UnsafePickleError(f"blocked {module}.{name}")


class _Sink:
    """Копит офферы и параметры и сбрасывает их в parquet пачками."""

    def __init__(self, out_dir: Path, chunk: int = 50_000):
        self.out_dir, self.chunk = out_dir, chunk
        self.offers, self.params = [], []
        self.w_off = self.w_par = None
        self.n = 0

    def add(self, o: dict) -> None:
        oid = o.get("@id")
        params = o.get("param") or []
        if isinstance(params, dict):
            params = [params]
        for p in params:
            if isinstance(p, dict):
                self.params.append((oid, p.get("@name"), p.get("#text"), p.get("@unit")))
        cats = o.get("categories")
        cid = o.get("categoryId") or (cats.get("categoryId") if isinstance(cats, dict) else None)
        if isinstance(cid, list):
            cid = ",".join(map(str, cid))
        extra = {k: v for k, v in o.items() if k not in OFFER_KEYS}
        self.offers.append({
            "offer_id": oid, "available": o.get("@available"), "vendor": o.get("vendor"), "model": o.get("model"),
            "name": o.get("name"), "category_id": cid, "n_params": len(params),
            "extra": json.dumps(extra, ensure_ascii=False) if extra else None,
        })
        self.n += 1
        if len(self.offers) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if self.offers:
            t = pa.Table.from_pylist(self.offers, schema=_OFFER_SCHEMA)
            self.w_off = self.w_off or pq.ParquetWriter(self.out_dir / "offers.parquet", t.schema)
            self.w_off.write_table(t)
            self.offers = []
        if self.params:
            cols = list(zip(*self.params))
            t = pa.table({k: pa.array(v, pa.string()) for k, v in zip(["offer_id", "param_name", "value", "unit"], cols)})
            self.w_par = self.w_par or pq.ParquetWriter(self.out_dir / "params.parquet", t.schema)
            self.w_par.write_table(t)
            self.params = []

    def close(self) -> None:
        self.flush()
        for w in (self.w_off, self.w_par):
            if w:
                w.close()


_OFFER_SCHEMA = pa.schema([
    ("offer_id", pa.string()), ("available", pa.string()), ("vendor", pa.string()), ("model", pa.string()),
    ("name", pa.string()), ("category_id", pa.string()), ("n_params", pa.int64()), ("extra", pa.string()),
])


def _referenced_memo(src: Path) -> set:
    """Проход 1: какие ячейки memo читаются через GET (обычно ключи dict). Заодно проверка на опасные опкоды."""
    refs = set()
    with open(src, "rb") as f:
        for op, arg, _ in pickletools.genops(f):
            if op.name in GET_OPS:
                refs.add(arg)
            elif op.name in UNSAFE:
                raise UnsafePickleError(f"unsafe opcode {op.name}")
    return refs


def _stream(src: Path, sink: _Sink, refs: set, log=print):
    """Проход 2: мини-VM. Офферы из списка под ключом "offer" сразу уходят в sink, в памяти не копятся."""
    stack, metastack, memo = [], [], {}
    memo_n, t0 = 0, time.time()
    with open(src, "rb") as f:
        for op, arg, _ in pickletools.genops(f):
            name = op.name
            if name in STR_OPS or name in INT_OPS or name == "BINFLOAT":
                stack.append(arg)
            elif name == "EMPTY_DICT":
                stack.append({})
            elif name == "EMPTY_LIST":
                stack.append([])
            elif name == "MARK":
                metastack.append(stack)
                stack = []
            elif name == "SETITEMS":
                items, stack = stack, metastack.pop()
                d = stack[-1]
                for i in range(0, len(items), 2):
                    d[items[i]] = items[i + 1]
            elif name == "SETITEM":
                v = stack.pop()
                k = stack.pop()
                stack[-1][k] = v
            elif name in ("APPENDS", "APPEND"):
                if name == "APPENDS":
                    items, stack = stack, metastack.pop()
                else:
                    items = [stack.pop()]
                lst = stack[-1]
                # список офферов: ключ "offer" лежит на стеке прямо под списком
                if len(stack) >= 2 and stack[-2] == "offer":
                    for it in items:
                        sink.add(it)
                        if sink.n % 100_000 == 0:
                            log(f"  {sink.n:,} офферов, {time.time() - t0:.0f} с")
                    lst.append(FLUSHED)
                else:
                    lst.extend(items)
            elif name == "MEMOIZE":
                if memo_n in refs:
                    memo[memo_n] = stack[-1]
                memo_n += 1
            elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
                if arg in refs:
                    memo[arg] = stack[-1]
            elif name in GET_OPS:
                stack.append(memo[arg])
            elif name == "NONE":
                stack.append(None)
            elif name in ("NEWTRUE", "NEWFALSE"):
                stack.append(name == "NEWTRUE")
            elif name == "TUPLE":
                items, stack = stack, metastack.pop()
                stack.append(tuple(items))
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                k = int(name[-1])
                t = tuple(stack[-k:])
                del stack[-k:]
                stack.append(t)
            elif name in ("PROTO", "FRAME"):
                pass
            elif name == "STOP":
                return stack.pop()
            else:
                raise UnsafePickleError(f"unsupported opcode {name}")
    raise UnsafePickleError("pickle без STOP")


def _write_categories(shop: dict, out_dir: Path) -> int:
    cats = shop["categories"]["category"]
    if isinstance(cats, dict):
        cats = [cats]
    rows = [{"id": c.get("@id"), "parent_id": c.get("@parentId"), "name": c.get("#text")} for c in cats]
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema(
        [("id", pa.string()), ("parent_id", pa.string()), ("name", pa.string())])), out_dir / "categories.parquet")
    return len(rows)


def parse_feed(src: Path, out_dir: Path, fast: bool = False, log=print) -> tuple[int, int]:
    """skus.pkl → out_dir/{offers,params,categories}.parquet. Возвращает (офферов, категорий)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = _Sink(out_dir)
    try:
        if fast:
            with open(src, "rb") as f:
                shop = DataOnlyUnpickler(f).load()["yml_catalog"]["shop"]
            for o in shop["offers"]["offer"]:
                sink.add(o)
        else:
            root = _stream(src, sink, _referenced_memo(src), log=log)
            shop = root["yml_catalog"]["shop"]
    finally:
        sink.close()
    return sink.n, _write_categories(shop, out_dir)


def feed_cached(out_dir: Path) -> bool:
    return all((out_dir / f).exists() for f in FEED_FILES)
