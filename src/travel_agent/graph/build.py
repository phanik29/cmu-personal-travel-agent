"""Graph wiring. One checkpoint thread per trip; the plan, the options, and cancellations each pause the graph
with `interrupt()` until the traveler decides."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from travel_agent.graph import execution as ex
from travel_agent.graph import nodes
from travel_agent.graph.deps import Deps
from travel_agent.graph.state import AgentTask, TravelState

AGENT_NODES = ["flight_agent", "hotel_agent", "car_agent", "transfer_agent", "context_agent"]
WAITING = ["await_plan_approval", "await_approval", "await_cancel_approval", END]


def build_graph(checkpointer: Any = None):
    builder = StateGraph(TravelState, context_schema=Deps)
    builder.add_node("route_intent", nodes.route_intent)
    builder.add_node("intake", nodes.intake)
    builder.add_node("planner", nodes.planner)
    builder.add_node("await_plan_approval", nodes.await_plan_approval)
    for agent in AGENT_NODES:
        builder.add_node(agent, getattr(nodes, agent), input_schema=AgentTask)
    builder.add_node("evaluate", nodes.evaluate)
    builder.add_node("planner_review", nodes.planner_review)
    builder.add_node("ask_user", nodes.ask_user)
    builder.add_node("prepare_proposal", ex.prepare_proposal)
    builder.add_node("await_approval", ex.await_approval)
    builder.add_node("record_approval", ex.record_approval)
    builder.add_node("execute", ex.execute)
    builder.add_node("verify", ex.verify)
    builder.add_node("reflect", ex.reflect)
    builder.add_node("handle_rejection", ex.handle_rejection)
    builder.add_node("prepare_cancellation", ex.prepare_cancellation)
    builder.add_node("await_cancel_approval", ex.await_cancel_approval)
    builder.add_node("execute_cancellation", ex.execute_cancellation)
    builder.add_node("cancel_declined", ex.cancel_declined)
    builder.add_node("qa_assistant", ex.qa_assistant)

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges("route_intent", nodes.route_after_intent, [
        "intake", "planner", "qa_assistant", "prepare_cancellation", "record_approval", "handle_rejection",
        "execute_cancellation", "cancel_declined", *AGENT_NODES, "evaluate", *WAITING])
    builder.add_conditional_edges("intake", nodes.route_after_intake, ["planner", *WAITING])
    builder.add_conditional_edges("planner", nodes.route_after_planner, [*AGENT_NODES, "evaluate",
                                                                         "await_plan_approval", END])
    builder.add_conditional_edges("await_plan_approval", nodes.route_after_plan_approval,
                                  [*AGENT_NODES, "evaluate", "route_intent", END])
    for agent in AGENT_NODES:
        builder.add_edge(agent, "evaluate")
    builder.add_edge("evaluate", "planner_review")
    builder.add_conditional_edges("planner_review", nodes.route_after_review,
                                  [*AGENT_NODES, "evaluate", "prepare_proposal", "ask_user"])
    builder.add_edge("ask_user", END)
    builder.add_edge("prepare_proposal", "await_approval")
    builder.add_conditional_edges("await_approval", ex.route_after_approval,
                                  ["record_approval", "handle_rejection", "intake", "route_intent", END])
    builder.add_conditional_edges("record_approval", ex.route_after_record, ["execute", END])
    builder.add_edge("execute", "verify")
    builder.add_edge("verify", "reflect")
    builder.add_edge("reflect", END)
    builder.add_edge("handle_rejection", END)
    builder.add_conditional_edges("prepare_cancellation", ex.route_after_prepare_cancellation,
                                  ["await_cancel_approval", END])
    builder.add_conditional_edges("await_cancel_approval", ex.route_after_cancel_approval,
                                  ["execute_cancellation", "cancel_declined", "route_intent"])
    builder.add_edge("execute_cancellation", END)
    builder.add_edge("cancel_declined", END)
    builder.add_conditional_edges("qa_assistant", ex.route_after_qa, WAITING)
    return builder.compile(checkpointer=checkpointer)
