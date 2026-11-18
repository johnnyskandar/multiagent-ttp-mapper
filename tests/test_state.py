import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from attack_mapper.state import (
    AgentState,
    Behavior,
    FinalReport,
    Mapping,
    MappingBatch,
    RejectionReason,
    ReportEntry,
    Status,
)


def _mapping(confidence: float) -> Mapping:
    return Mapping(
        behavior_index=0,
        technique_id="T1566.001",
        technique_name="Spearphishing Attachment",
        evidence_quote="emailed staff a Word file",
        confidence=confidence,
        reasoning="test",
    )


def test_agent_state_defaults():
    s = AgentState()
    assert s.report_text == ""
    assert s.behaviors == [] and s.proposals == [] and s.accepted_mappings == []
    assert s.rejections == [] and s.messages == []
    assert s.iocs == {}
    assert s.retry_count == 0
    assert s.max_retries == 2
    assert s.status is Status.RUNNING


def test_optional_fields_default_none_and_accept_values():
    s = AgentState()
    assert s.final_report is None
    assert s.follow_up_question is None
    assert s.followup_answer is None
    assert s.error is None
    s2 = AgentState(
        follow_up_question="What mitigations apply?",
        error="boom",
        final_report=FinalReport(entries=[], summary="none"),
    )
    assert s2.follow_up_question == "What mitigations apply?"
    assert s2.final_report.summary == "none"
    assert ReportEntry(
        technique_id="T1", technique_name="x", evidence_quote="q", confidence=0.1
    ).detection_hint is None


def test_default_factories_are_not_shared():
    a, b = AgentState(), AgentState()
    a.behaviors.append(Behavior(index=0, sentence="x"))
    assert b.behaviors == []
    assert MappingBatch(mappings=[]).unmapped_indices == []


def test_enum_values():
    assert [s.value for s in Status] == [
        "running",
        "no_behaviors",
        "complete",
        "complete_with_unmapped",
        "error",
    ]
    assert [r.value for r in RejectionReason] == [
        "unknown_id",
        "deprecated_id",
        "quote_not_found",
        "semantic_mismatch",
    ]
    assert AgentState(status="complete").status is Status.COMPLETE


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_mapping_confidence_in_bounds(confidence):
    assert _mapping(confidence).confidence == confidence


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0])
def test_mapping_confidence_out_of_bounds(confidence):
    with pytest.raises(ValidationError):
        _mapping(confidence)


def test_messages_field_uses_add_messages_reducer():
    # The reducer is attached as Annotated metadata on the field.
    metadata = AgentState.model_fields["messages"].metadata
    assert add_messages in metadata


def test_add_messages_appends_and_replaces_by_id():
    s = AgentState(messages=[HumanMessage(content="hi", id="1")])
    merged = add_messages(s.messages, [AIMessage(content="hello", id="2", name="mapper")])
    assert [m.content for m in merged] == ["hi", "hello"]
    merged = add_messages(merged, [AIMessage(content="hello v2", id="2")])
    assert [m.content for m in merged] == ["hi", "hello v2"]
    assert AgentState(messages=merged).messages[1].content == "hello v2"
