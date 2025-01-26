from .utils import create_simple_logger

from crewai.tools.base_tool import Tool
from smolagents import Tool as SmolagentsTool
from pydantic import BaseModel, Field, create_model
from typing import Any
from inspect import signature

logger = create_simple_logger(__name__)


def create_tool_from_smolagents(tool: SmolagentsTool) -> "Tool":
    """Create a Tool instance from a smolagents tool

    This method takes a smolagents tool object and converts it into a
    Tool instance. It ensures that the provided tool has a callable 'func'
    attribute and infers the argument schema and description.

    Args:
        tool (SmolagentsTool): The smolagents tool object to be converted.

    Returns:
        Tool: A new Tool instance created from the provided smolagents tool

    Raises:
        ValueError: If the provided tool does not have a callable 'forward' attribute.
    """
    # smolagents tools implement `forward` method which is called when
    # the tool is called
    if not hasattr(tool, "forward"):
        msg = "The provided tool must have a callable 'forward' attribute."
        logger.error(msg)
        raise ValueError(msg)

    annotations = signature(tool.forward).parameters
    inputs = tool.inputs
    args_fields = {}

    for name, param in annotations.items():
        if name == "self":
            continue

        param_annotation = param.annotation if param.annotation != param.empty else Any
        field_info = Field(default=..., description=inputs[name]["description"])
        args_fields[name] = (param_annotation, field_info)

    if args_fields:
        args_schema = create_model(f"{tool.name}Input", **args_fields)
    else:
        # Create a default schema with no fields if no parameters are found
        args_schema = create_model(f"{tool.name}Input", __base__=BaseModel)

    return Tool(
        name=getattr(tool, "name", "Unnamed Tool"),
        description=tool.description,
        func=tool.forward,
        args_schema=args_schema,
    )
