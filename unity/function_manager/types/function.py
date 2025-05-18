from pydantic import BaseModel
from typing import Optional, Dict


class Function(BaseModel):
    function_id: int
    source: str
    referenced: Optional[Dict[int, int]]
