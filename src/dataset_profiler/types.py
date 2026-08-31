"""Type inference for a column of text.

Every field in a CSV is text, so a "type" is a claim about what the text
consistently looks like. This module makes that claim per value
(:func:`classify`) and then folds the per-value claims together with a
widening lattice (:func:`unify`), which is what lets a column of mostly
integers with one ``"12.5"`` in it come out as ``float`` rather than as
``string`` or as a lie.

The lattice is deliberately small and total:

- ``empty`` unifies with anything and disappears
- ``integer`` + ``float`` -> ``float``
- ``date`` + ``datetime`` -> ``datetime``
- everything else that disagrees -> ``string``

``string`` is therefore the top of the lattice, and unification is
associative and commutative, so the inferred type does not depend on row
order. Both properties are asserted in the tests.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime

EMPTY = "empty"
BOOLEAN = "boolean"
INTEGER = "integer"
FLOAT = "float"
DATE = "date"
DATETIME = "datetime"
STRING = "string"

#: Every kind :func:`classify` can return, in widening order.
KINDS = (EMPTY, BOOLEAN, INTEGER, FLOAT, DATE, DATETIME, STRING)

#: Kinds whose values participate in the numeric summary.
NUMERIC_KINDS = frozenset({INTEGER, FLOAT})

#: Kinds that describe a point in time.
TEMPORAL_KINDS = frozenset({DATE, DATETIME})

#: Text that, whatever its case or padding, means "no value here".
DEFAULT_NULL_TOKENS = frozenset(
    {"", "na", "n/a", "nan", "null", "none", "nil", "-", "?"}
)

TRUE_TOKENS = frozenset({"true", "yes", "t", "y"})
FALSE_TOKENS = frozenset({"false", "no", "f", "n"})

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_ISO_DATE_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}$")


def is_null(value: str, null_tokens: frozenset[str] = DEFAULT_NULL_TOKENS) -> bool:
    """Whether ``value`` is one of the tokens that mean "missing"."""
    return value.strip().lower() in null_tokens


def null_token(value: str) -> str:
    """The canonical spelling of a null token, for null-pattern counting.

    The empty string and a whitespace-only string are both reported as
    ``""`` — they are the same defect — while ``"N/A"`` keeps its case so
    a report can say which spelling the file actually uses.
    """
    stripped = value.strip()
    return stripped


def parse_boolean(value: str) -> bool | None:
    """``True``/``False`` for boolean-looking text, else ``None``.

    ``0`` and ``1`` are deliberately *not* booleans: a column of zeroes
    and ones is far more often a count or a flag that gets summed than a
    thing you want reported as ``boolean``.
    """
    token = value.strip().lower()
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    return None


def to_number(value: str) -> float | None:
    """The finite float ``value`` denotes, or ``None`` if it denotes none.

    ``inf`` and ``nan`` parse in Python but are not numbers a profile can
    summarise, so they are rejected here and fall through to ``string``.
    """
    token = value.strip()
    if not token:
        return None
    try:
        number = float(token)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_date(value: str) -> date | None:
    """The date ``value`` denotes, accepting ``YYYY-MM-DD`` and ``YYYY/MM/DD``."""
    token = value.strip()
    if not _ISO_DATE_RE.match(token):
        return None
    try:
        return date.fromisoformat(token.replace("/", "-"))
    except ValueError:
        return None


def parse_datetime(value: str) -> datetime | None:
    """The timestamp ``value`` denotes, in ISO 8601 with an optional ``Z``."""
    token = value.strip()
    if len(token) < 11:
        return None
    if token.endswith(("Z", "z")):
        token = f"{token[:-1]}+00:00"
    try:
        return datetime.fromisoformat(token)
    except ValueError:
        return None


def classify(
    value: str, null_tokens: frozenset[str] = DEFAULT_NULL_TOKENS
) -> str:
    """The kind of a single field.

    Order matters: ``"2024-01-02"`` must be seen as a date before the
    datetime parser, which also accepts it, gets a look at it.
    """
    if is_null(value, null_tokens):
        return EMPTY
    if parse_boolean(value) is not None:
        return BOOLEAN
    token = value.strip()
    if _INTEGER_RE.match(token):
        return INTEGER
    if to_number(token) is not None:
        return FLOAT
    if parse_date(token) is not None:
        return DATE
    if parse_datetime(token) is not None:
        return DATETIME
    return STRING


def unify(left: str, right: str) -> str:
    """The narrowest kind that describes both ``left`` and ``right``."""
    if left == right:
        return left
    if left == EMPTY:
        return right
    if right == EMPTY:
        return left
    pair = {left, right}
    if pair == {INTEGER, FLOAT}:
        return FLOAT
    if pair == {DATE, DATETIME}:
        return DATETIME
    return STRING


def infer(kinds: dict[str, int]) -> str:
    """Fold a histogram of per-value kinds into one inferred type.

    An all-empty column stays ``empty`` rather than being called
    ``string``: "we never saw a value" is a different fact from "we saw
    text", and only the first one should stop you trusting the column.
    """
    result = EMPTY
    for kind, count in kinds.items():
        if count:
            result = unify(result, kind)
    return result
