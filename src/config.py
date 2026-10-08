from pydantic import BaseModel, Field
from typing import Optional

class MonitorConfig(BaseModel):
    model: str
    api_key: str
    base_url: str
    temperature: Optional[float] = Field(default=0.01, le=1.0, ge=0.0)
    top_p: Optional[float] = Field(default=1.0, le=1.0, ge=0.01)
    enable_thinking: Optional[bool] = False
