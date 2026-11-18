from attack_mapper.graph import (
    build_graph,
    route_after_extractor,
    route_after_validator,
    route_entry,
)
from attack_mapper.state import (
    AgentState,
    Behavior,
    FinalReport,
    Rejection,
    RejectionReason,
    Status,
)

REPORT = "The actor emailed staff a Word file containing a malicious macro to gain access."


def _rejection() -> Rejection:
    return Rejection(
        behavior_index=0,
        technique_id="T9999",
        reason=RejectionReason.UNKNOWN_ID,
        detail="ID does not exist",
    )


# route_entry -----------------------------------------------------------------


def test_route_entry_new_report_goes_to_extractor():
    assert route_entry(AgentState(report_text=REPORT)) == "extractor"


def test_route_entry_followup_with_existing_report():
    s = AgentState(follow_up_question="Q?", final_report=FinalReport(entries=[], summary="s"))
    assert route_entry(s) == "followup"


def test_route_entry_followup_without_report_goes_to_extractor():
    assert route_entry(AgentState(follow_up_question="Q?")) == "extractor"


# route_after_extractor ---------------------------------------------------------


def test_route_after_extractor_no_behaviors():
    assert route_after_extractor(AgentState()) == "no_results"


def test_route_after_extractor_with_behaviors():
    s = AgentState(behaviors=[Behavior(index=0, sentence="x")])
    assert route_after_extractor(s) == "mapper"


# route_after_validator ---------------------------------------------------------


def test_route_after_validator_all_accepted():
    assert route_after_validator(AgentState(rejections=[])) == "advisor"


def test_route_after_validator_rejections_under_cap_retries():
    s = AgentState(rejections=[_rejection()], retry_count=1, max_retries=2)
    assert route_after_validator(s) == "mapper"


def test_route_after_validator_retry_cap_reached():
    s = AgentState(rejections=[_rejection()], retry_count=2, max_retries=2)
    assert route_after_validator(s) == "advisor"


# end to end on the stub graph ------------------------------------------------


def test_stub_graph_end_to_end_with_thread_and_followup():
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-thread-1"}}

    visited = [
        node
        for update in graph.stream({"report_text": REPORT}, config, stream_mode="updates")
        for node in update
    ]
    assert visited == ["extractor", "mapper", "validator", "advisor"]

    values = graph.get_state(config).values
    assert values["status"] is Status.COMPLETE
    assert values["final_report"].entries[0].technique_id == "T1566.001"
    assert len(values["messages"]) == 4

    # Same thread: checkpoint still holds the report, route_entry goes to followup.
    visited = [
        node
        for update in graph.stream(
            {"follow_up_question": "What mitigations apply?"}, config, stream_mode="updates"
        )
        for node in update
    ]
    assert visited == ["followup"]
    values = graph.get_state(config).values
    assert values["report_text"] == REPORT
    assert values["follow_up_question"] is None
    assert "What mitigations apply?" in values["followup_answer"].answer
    assert len(values["messages"]) == 5


def test_stub_graph_no_behaviors_path():
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-thread-2"}}
    final = graph.invoke({"report_text": ""}, config)
    assert final["status"] is Status.NO_BEHAVIORS
