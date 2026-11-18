"""LangGraph builder, routing functions, and (for Phase 1) stub nodes.

The stub nodes return canned PARTIAL update dicts so the graph topology, routing,
and checkpointing can be tested before the real agents exist. They are replaced
by the real nodes in agents/ during Phase 2 (T2.1).
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from attack_mapper.state import (
    AgentState,
    Behavior,
    FinalReport,
    FollowupAnswer,
    Mapping,
    ReportEntry,
    Status,
)

# ---------------------------------------------------------------------------
# Routing (section 8.3)
# ---------------------------------------------------------------------------


def route_entry(state: AgentState) -> str:
    return "followup" if (state.follow_up_question and state.final_report) else "extractor"


def route_after_extractor(state: AgentState) -> str:
    return "mapper" if state.behaviors else "no_results"


def route_after_validator(state: AgentState) -> str:
    if state.rejections and state.retry_count < state.max_retries:
        return "mapper"
    return "advisor"


# ---------------------------------------------------------------------------
# Stub nodes (canned partial updates)
# ---------------------------------------------------------------------------


def _stub_msg(agent: str, text: str) -> AIMessage:
    return AIMessage(content=f"[stub] {text}", name=agent)


def stub_extractor(state: AgentState) -> dict:
    behaviors = (
        [Behavior(index=0, sentence=state.report_text.strip()[:200])]
        if state.report_text.strip()
        else []
    )
    return {
        "behaviors": behaviors,
        "iocs": {},
        "messages": [_stub_msg("extractor", f"extracted {len(behaviors)} behavior(s)")],
    }


def stub_mapper(state: AgentState) -> dict:
    proposals = [
        Mapping(
            behavior_index=b.index,
            technique_id="T1566.001",
            technique_name="Spearphishing Attachment",
            evidence_quote=b.sentence,
            confidence=0.5,
            reasoning="stub mapping",
        )
        for b in state.behaviors
    ]
    return {
        "proposals": proposals,
        "messages": [_stub_msg("mapper", f"proposed {len(proposals)} mapping(s)")],
    }


def stub_validator(state: AgentState) -> dict:
    accepted = list(state.accepted_mappings) + list(state.proposals)
    return {
        "accepted_mappings": accepted,
        "rejections": [],
        "retry_count": state.retry_count,
        "messages": [_stub_msg("validator", f"accepted {len(state.proposals)} mapping(s)")],
    }


def stub_advisor(state: AgentState) -> dict:
    entries = [
        ReportEntry(
            technique_id=m.technique_id,
            technique_name=m.technique_name,
            evidence_quote=m.evidence_quote,
            confidence=m.confidence,
        )
        for m in state.accepted_mappings
    ]
    report = FinalReport(entries=entries, iocs=state.iocs, summary="stub summary")
    return {
        "final_report": report,
        "status": Status.COMPLETE,
        "messages": [_stub_msg("advisor", f"final report with {len(entries)} entr(y/ies)")],
    }


def stub_followup(state: AgentState) -> dict:
    return {
        "followup_answer": FollowupAnswer(answer=f"stub answer to: {state.follow_up_question}"),
        "follow_up_question": None,
        "messages": [_stub_msg("followup", "answered follow-up question")],
    }


def stub_no_results(state: AgentState) -> dict:
    return {
        "status": Status.NO_BEHAVIORS,
        "messages": [_stub_msg("no_results", "no attacker behaviors found")],
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Build and compile the graph. Uses an in-memory MemorySaver by default."""
    builder = StateGraph(AgentState)

    builder.add_node("extractor", stub_extractor)
    builder.add_node("mapper", stub_mapper)
    builder.add_node("validator", stub_validator)
    builder.add_node("advisor", stub_advisor)
    builder.add_node("followup", stub_followup)
    builder.add_node("no_results", stub_no_results)

    builder.add_conditional_edges(
        START, route_entry, {"followup": "followup", "extractor": "extractor"}
    )
    builder.add_conditional_edges(
        "extractor", route_after_extractor, {"mapper": "mapper", "no_results": "no_results"}
    )
    builder.add_edge("mapper", "validator")
    builder.add_conditional_edges(
        "validator", route_after_validator, {"mapper": "mapper", "advisor": "advisor"}
    )
    builder.add_edge("advisor", END)
    builder.add_edge("followup", END)
    builder.add_edge("no_results", END)

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())
