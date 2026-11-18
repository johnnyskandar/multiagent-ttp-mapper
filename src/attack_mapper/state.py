from enum import Enum
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Status(str, Enum):
    RUNNING = "running"
    NO_BEHAVIORS = "no_behaviors"
    COMPLETE = "complete"
    COMPLETE_WITH_UNMAPPED = "complete_with_unmapped"
    ERROR = "error"


class RejectionReason(str, Enum):
    UNKNOWN_ID = "unknown_id"
    DEPRECATED_ID = "deprecated_id"
    QUOTE_NOT_FOUND = "quote_not_found"
    SEMANTIC_MISMATCH = "semantic_mismatch"


class Behavior(BaseModel):
    index: int
    sentence: str  # verbatim excerpt from the report so quotes can be matched


class ExtractionResult(BaseModel):  # Extractor structured output
    behaviors: list[Behavior]


class Mapping(BaseModel):
    behavior_index: int
    technique_id: str = Field(description="ATT&CK ID such as T1566 or T1566.001")
    technique_name: str
    evidence_quote: str  # verbatim from the behavior sentence
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class MappingBatch(BaseModel):  # Mapper structured output
    mappings: list[Mapping]
    unmapped_indices: list[int] = Field(default_factory=list)


class Rejection(BaseModel):
    behavior_index: int
    technique_id: str
    reason: RejectionReason
    detail: str


class ValidationReport(BaseModel):  # Validator structured output
    accepted_indices: list[int]
    rejections: list[Rejection]


class ReportEntry(BaseModel):
    technique_id: str
    technique_name: str
    tactics: list[str] = Field(default_factory=list)
    evidence_quote: str
    confidence: float
    mitigations: list[str] = Field(default_factory=list)  # "M1017 User Training: ..."
    detection_hint: Optional[str] = None


class FinalReport(BaseModel):  # Advisor structured output
    entries: list[ReportEntry]
    unmapped_behaviors: list[Behavior] = Field(default_factory=list)
    iocs: dict[str, list[str]] = Field(default_factory=dict)
    summary: str


class FollowupAnswer(BaseModel):
    answer: str
    cited_technique_ids: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    report_text: str = ""
    behaviors: list[Behavior] = Field(default_factory=list)
    iocs: dict[str, list[str]] = Field(default_factory=dict)
    proposals: list[Mapping] = Field(default_factory=list)          # latest Mapper output
    accepted_mappings: list[Mapping] = Field(default_factory=list)  # accumulated by Validator
    rejections: list[Rejection] = Field(default_factory=list)       # latest pass only
    retry_count: int = 0
    max_retries: int = 2
    status: Status = Status.RUNNING
    final_report: Optional[FinalReport] = None
    follow_up_question: Optional[str] = None
    followup_answer: Optional[FollowupAnswer] = None
    error: Optional[str] = None
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
