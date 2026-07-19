"""Bounded product orchestration for local Cogni-OS agents.

The package root is intentionally lazy. Importing a lightweight child such as
``cogni_agent.tools`` must not preload the model service or ``torch`` in the
GPU5 control process before its launch authority is marked attempted.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ACTIVE_AGENT_STATUSES": ("manager", "ACTIVE_AGENT_STATUSES"),
    "AgentBusyError": ("manager", "AgentBusyError"),
    "AgentManager": ("manager", "AgentManager"),
    "BaseModelMutationError": ("model_service", "BaseModelMutationError"),
    "BoundedConversationStore": ("conversation", "BoundedConversationStore"),
    "ConversationError": ("conversation", "ConversationError"),
    "ConversationFastPath": ("conversation_fastpath", "ConversationFastPath"),
    "ConversationSnapshot": ("conversation", "ConversationSnapshot"),
    "ConversationTurn": ("conversation", "ConversationTurn"),
    "CorePipelineLimits": ("core_pipeline", "CorePipelineLimits"),
    "CoreTurnAuthorityError": ("core_pipeline", "CoreTurnAuthorityError"),
    "CoreTurnPipeline": ("core_pipeline", "CoreTurnPipeline"),
    "CoreTurnRequest": ("core_pipeline", "CoreTurnRequest"),
    "CoreTurnResult": ("core_pipeline", "CoreTurnResult"),
    "CoreTurnTelemetry": ("core_pipeline", "CoreTurnTelemetry"),
    "FastWeightActivation": ("core_pipeline", "FastWeightActivation"),
    "FastWeightCompilationPlan": ("core_pipeline", "FastWeightCompilationPlan"),
    "GenerationCancelled": ("model_service", "GenerationCancelled"),
    "GenerationChunk": ("model_service", "GenerationChunk"),
    "GenerationResult": ("model_service", "GenerationResult"),
    "LocalGemmaCorePipelineFactory": (
        "model_service",
        "LocalGemmaCorePipelineFactory",
    ),
    "LocalGemmaModelFactory": ("model_service", "LocalGemmaModelFactory"),
    "ModelService": ("model_service", "ModelService"),
    "ModelServiceError": ("model_service", "ModelServiceError"),
    "NoActiveAgentTurnError": ("manager", "NoActiveAgentTurnError"),
    "ToolPolicyError": ("tools", "ToolPolicyError"),
    "ToolRequest": ("tools", "ToolRequest"),
    "ToolResult": ("tools", "ToolResult"),
    "WorkspaceToolExecutor": ("tools", "WorkspaceToolExecutor"),
    "WorkerAuthorityError": ("model_service", "WorkerAuthorityError"),
    "parse_tool_request": ("tools", "parse_tool_request"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
