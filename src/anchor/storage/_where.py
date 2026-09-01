"""Shared where-dict + namespace-scope compilation (front #3).

One operator table for every backend: the convergence core the market
agrees on (`$eq $ne $in $nin $gt $gte $lt $lte`; plain values mean
equality). Unknown ``$``-operators raise — a silently ignored filter is
a data leak, not a convenience.

Namespace scoping compiles to boundary-aware prefix RANGES
(``ns = p OR (ns >= p||'/' AND ns < p||'0')`` — ``'0'`` is the character
after ``'/'``), never ``LIKE``: the range form is collation-proof in
SQLite and index-friendly everywhere, and it is the shape sqlite-vec's
vec0 can push down (LIKE it cannot).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anchor.models.scope import ROOT_NAMESPACE, RetrievalScope

_SQL_OPS: dict[str, str] = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}
_SET_OPS = ("$in", "$nin")

# The character immediately after '/' in ASCII — upper bound of a
# prefix range. Safe for UTF-8 namespaces.
_AFTER_SLASH = "0"


def _is_op_dict(cond: Any) -> bool:
    return isinstance(cond, dict) and any(
        isinstance(k, str) and k.startswith("$") for k in cond
    )


def _check_op(op: str) -> None:
    if op not in _SQL_OPS and op not in _SET_OPS:
        msg = (
            f"unknown where operator {op!r} — supported: "
            f"{sorted([*_SQL_OPS, *_SET_OPS])}"
        )
        raise ValueError(msg)


def matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Pure-Python evaluation of a where dict (in-memory backends)."""
    if not where:
        return True
    for key, cond in where.items():
        val = metadata.get(key)
        if not _is_op_dict(cond):
            if val != cond:
                return False
            continue
        for op, operand in cond.items():
            _check_op(op)
            if op == "$eq" and val != operand:
                return False
            if op == "$ne" and val == operand:
                return False
            if op == "$in" and val not in operand:
                return False
            if op == "$nin" and val in operand:
                return False
            if op in ("$gt", "$gte", "$lt", "$lte"):
                if val is None:
                    return False
                try:
                    if op == "$gt" and not val > operand:
                        return False
                    if op == "$gte" and not val >= operand:
                        return False
                    if op == "$lt" and not val < operand:
                        return False
                    if op == "$lte" and not val <= operand:
                        return False
                except TypeError:
                    return False
    return True


def sql_where_clauses(
    where: dict[str, Any] | None,
    expr_for_key: Callable[[str], tuple[str, list[Any]]],
) -> tuple[list[str], list[Any]]:
    """Compile a where dict to qmark-style SQL clauses.

    ``expr_for_key(key)`` returns ``(sql_expression, params)`` for the
    column/JSON path holding that key (e.g. ``("json_extract(metadata_json, ?)",
    ["$.year"])``), so one compiler serves plain columns and JSON alike.
    """
    clauses: list[str] = []
    params: list[Any] = []
    for key, cond in (where or {}).items():
        expr, expr_params = expr_for_key(key)
        if not _is_op_dict(cond):
            clauses.append(f"{expr} = ?")
            params.extend([*expr_params, cond])
            continue
        for op, operand in cond.items():
            _check_op(op)
            if op in _SET_OPS:
                values = list(operand)
                if not values:
                    # IN () matches nothing; NOT IN () matches everything.
                    clauses.append("1=0" if op == "$in" else "1=1")
                    continue
                placeholders = ",".join("?" for _ in values)
                neg = "NOT " if op == "$nin" else ""
                clauses.append(f"{expr} {neg}IN ({placeholders})")
                params.extend([*expr_params, *values])
            else:
                clauses.append(f"{expr} {_SQL_OPS[op]} ?")
                params.extend([*expr_params, operand])
    return clauses, params


def _prefix_range(ns_col: str, prefix: str) -> tuple[str, list[Any]]:
    if prefix == ROOT_NAMESPACE:
        return "1=1", []
    return (
        f"({ns_col} = ? OR ({ns_col} >= ? AND {ns_col} < ?))",
        [prefix, prefix + "/", prefix + _AFTER_SLASH],
    )


def scope_sql_clauses(
    scope: RetrievalScope | None, ns_col: str,
) -> tuple[list[str], list[Any]]:
    """Compile a RetrievalScope to qmark-style SQL clauses over *ns_col*.

    Exclude wins by construction: includes are OR-ed into one clause,
    every exclude is an AND NOT on top.
    """
    if scope is None:
        return [], []
    clauses: list[str] = []
    params: list[Any] = []
    if scope.include:
        parts: list[str] = []
        for prefix in scope.include:
            frag, p = _prefix_range(ns_col, prefix)
            parts.append(frag)
            params.extend(p)
        clauses.append("(" + " OR ".join(parts) + ")")
    for prefix in scope.exclude:
        frag, p = _prefix_range(ns_col, prefix)
        clauses.append(f"NOT {frag}")
        params.extend(p)
    return clauses, params
