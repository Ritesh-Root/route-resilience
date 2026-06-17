"""Pydantic response/request models (mirrors the front-end data contract)."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

InputMode = Literal["clean", "occluded"]
ModelKind = Literal["baseline", "robust"]

# Bounded id string (prevents oversized/abusive simulate payloads).
EdgeOrNodeId = Annotated[str, StringConstraints(max_length=64)]


class Metrics(BaseModel):
    iou: float
    dice: float
    occlusionRecall: float
    resilienceIndex: int


class GatekeeperNode(BaseModel):
    id: str
    lat: float
    lng: float
    betweenness: float
    isArticulation: bool
    label: str


class ResilienceCurve(BaseModel):
    removedFraction: list[float]
    efficiency: list[float]
    giantComponent: list[float]


class SimulationRequest(BaseModel):
    city: Annotated[str, StringConstraints(max_length=64)] = "Bengaluru"
    model: ModelKind = "robust"
    input: InputMode = "clean"
    disabledEdgeIds: list[EdgeOrNodeId] = Field(default_factory=list, max_length=500)
    disabledNodeIds: list[EdgeOrNodeId] = Field(default_factory=list, max_length=500)


class SimulationResult(BaseModel):
    disabledEdgeIds: list[str]
    disabledNodeIds: list[str]
    resilienceIndexAfter: int
    avgTravelTimeIncreasePct: float
    newlyDisconnectedZones: int
    brokenRoutesSampled: int
    sampledRoutes: int


class InferResponse(BaseModel):
    note: str
    network: dict[str, Any]
