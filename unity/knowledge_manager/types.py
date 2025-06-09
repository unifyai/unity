from enum import StrEnum


class ColumnType(StrEnum):
    str = "str"
    int = "int"
    float = "float"
    bool = "bool"
    dict = "dict"
    list = "list"
    timestamp = "timestamp"
    date = "date"
    time = "time"


column_type_schema = {
    "title": "ColumnType",
    "type": "string",
    "enum": [member.value for member in ColumnType],
    "description": "Allowed types for a column.",
}
