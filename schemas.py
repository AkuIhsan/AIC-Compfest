from __future__ import annotations
from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    batch_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    machine_id: str
    shift: str
    operator_id: str
    material_batch: str
    curing_time_actual: float
    temp_actual: float
    pressure_actual: float
    manual_override: bool
    visual_check_done: bool