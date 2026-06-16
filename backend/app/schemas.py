"""Pydantic response/request models (mirrors the front-end data contract)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

InputMode = Literal["clean", "occluded"]
ModelKind = Literal["baseline", "robust"]


class Metrics(BaseModel):
    iou: float
    dice: float
    occlusionRecall: float
    connectivityRatio: float
    apls: float
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
    city: str = "Bengaluru"
    model: ModelKind = "robust"
    input: InputMode = "clean"
    disabledEdgeIds: list[str] = Field(default_factory=list)
    disabledNodeIds: list[str] = Field(default_factory=list)


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
