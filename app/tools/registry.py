from typing import Any, Callable, Coroutine
from app.core.logging import logger
from app.schemas.tools import ToolDefinition, ToolResult


class ToolRegistry:
    """Central registry of backend processing tools."""

    def __init__(self):
        self._tools: dict[str, Callable[..., Coroutine[Any, Any, ToolResult]]] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        supported_modalities: list[str] | None = None,
        supported_tasks: list[str] | None = None,
    ):
        """Decorator to register a tool function."""
        def decorator(func: Callable[..., Coroutine[Any, Any, ToolResult]]):
            self._tools[name] = func
            self._definitions[name] = ToolDefinition(
                name=name,
                description=description,
                supported_modalities=supported_modalities or [],
                supported_tasks=supported_tasks or [],
            )
            logger.info("tool_registered", tool_name=name)
            return func
        return decorator

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with keyword arguments."""
        if name not in self._tools:
            return ToolResult(
                tool=name,
                status="error",
                error=f"Tool '{name}' is not registered in ToolRegistry."
            )
        try:
            return await self._tools[name](**kwargs)
        except Exception as e:
            logger.error("tool_execution_failed", tool=name, error=str(e))
            return ToolResult(
                tool=name,
                status="error",
                error=f"Execution error in tool '{name}': {str(e)}"
            )

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions.values())


tool_registry = ToolRegistry()
