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
