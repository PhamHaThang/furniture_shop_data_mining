from __future__ import annotations
from typing import Any

def to_id(value:Any) ->str:
    """
    Convert a value to a string ID. If the value is None, return an empty string.
    """
    if isinstance(value, dict):
        for key in ['id', '_id', '$oid']:
            if key in value:
                return str(value[key])
    return str(value)

def join_tokens(value:Any) ->str:
    """
    Join a value into a string token. If the value is None, return an empty string.
    If the value is a list, join its elements with a space.
    """
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)