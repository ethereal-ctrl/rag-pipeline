"""Notional USD estimates based on hosted reference list prices."""

from importlib.resources import files

import yaml
from pydantic import BaseModel, TypeAdapter

from rag_pipeline.tracing import StageTrace


class ReferencePrice(BaseModel):
    reference_model: str
    usd_per_million_input_tokens: float
    usd_per_million_output_tokens: float
    source: str
    looked_up: str


def load_prices() -> dict[str, ReferencePrice]:
    data = yaml.safe_load(files("rag_pipeline").joinpath("prices.yaml").read_text("utf-8"))
    return TypeAdapter(dict[str, ReferencePrice]).validate_python(data)


def notional_cost(
    stages: tuple[StageTrace, ...], prices: dict[str, ReferencePrice]
) -> float | None:
    if not stages:
        return None
    total = 0.0
    for stage in stages:
        if stage.model is None:
            continue
        price = prices.get(stage.model)
        if price is None or stage.input_tokens is None:
            return None
        total += (
            stage.input_tokens * price.usd_per_million_input_tokens
            + stage.output_tokens * price.usd_per_million_output_tokens
        )
    return total / 1_000_000
