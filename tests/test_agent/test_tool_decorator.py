"""Tests for agent/tool_decorator.py — @tool decorator."""

from __future__ import annotations

from pydantic import BaseModel, Field

from anchor.agent.models import AgentTool
from anchor.agent.tool_decorator import tool


class TestBareDecorator:
    def test_creates_agent_tool(self) -> None:
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello {name}"

        assert isinstance(greet, AgentTool)

    def test_name_from_function(self) -> None:
        @tool
        def my_tool(x: str) -> str:
            """Do stuff."""
            return x

        assert my_tool.name == "my_tool"

    def test_description_from_docstring(self) -> None:
        @tool
        def save_fact(fact: str) -> str:
            """Save an important fact about the user."""
            return fact

        assert save_fact.description == "Save an important fact about the user."

    def test_schema_has_required_fields(self) -> None:
        @tool
        def search(query: str) -> str:
            """Search for something."""
            return query

        schema = search.input_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_schema_with_defaults(self) -> None:
        @tool
        def search(query: str, limit: int = 5) -> str:
            """Search."""
            return query

        schema = search.input_schema
        assert "query" in schema["required"]
        assert "limit" not in schema.get("required", [])
        assert schema["properties"]["limit"]["default"] == 5

    def test_fn_callable(self) -> None:
        @tool
        def add(a: str) -> str:
            """Add."""
            return f"added {a}"

        result = add.fn(a="hello")
        assert result == "added hello"

    def test_tool_schema(self) -> None:
        @tool
        def ping(msg: str) -> str:
            """Ping."""
            return msg

        schema = ping.to_tool_schema()
        assert schema.name == "ping"
        assert schema.description == "Ping."
        assert schema.input_schema is not None

    def test_descriptions_from_docstring_args(self) -> None:
        @tool
        def get_weather(city: str, units: str = "celsius") -> str:
            """Get weather for a city.

            Args:
                city: The city name to look up.
                units: Temperature units (celsius or fahrenheit).
            """
            return f"{city} {units}"

        schema = get_weather.input_schema
        assert schema["properties"]["city"]["description"] == "The city name to look up."
        assert (
            schema["properties"]["units"]["description"]
            == "Temperature units (celsius or fahrenheit)."
        )

    def test_no_docstring_uses_name(self) -> None:
        @tool
        def mystery(x: str) -> str:
            return x

        assert mystery.description == "mystery"


class TestParameterisedDecorator:
    def test_custom_name(self) -> None:
        @tool(name="custom_greet")
        def greet(name: str) -> str:
            """Greet."""
            return f"Hello {name}"

        assert greet.name == "custom_greet"

    def test_custom_description(self) -> None:
        @tool(description="A custom description")
        def greet(name: str) -> str:
            """Original doc."""
            return f"Hello {name}"

        assert greet.description == "A custom description"

    def test_custom_name_and_description(self) -> None:
        @tool(name="hi", description="Say hi")
        def greet(name: str) -> str:
            """Ignored."""
            return f"Hello {name}"

        assert greet.name == "hi"
        assert greet.description == "Say hi"


class TestWithInputModel:
    def test_explicit_pydantic_model(self) -> None:
        class WeatherInput(BaseModel):
            city: str = Field(description="City name")
            units: str = Field(default="celsius", description="Temperature units")

        @tool(input_model=WeatherInput)
        def get_weather(city: str, units: str = "celsius") -> str:
            """Get weather."""
            return f"{city} {units}"

        schema = get_weather.input_schema
        assert schema["properties"]["city"]["description"] == "City name"
        assert schema["properties"]["units"]["default"] == "celsius"

    def test_model_schema_used_over_function_hints(self) -> None:
        class MyInput(BaseModel):
            x: int = Field(description="Custom X")

        @tool(input_model=MyInput)
        def fn(x: str) -> str:
            """Fn."""
            return x

        schema = fn.input_schema
        # Should use MyInput's int type, not function's str
        assert schema["properties"]["x"]["type"] == "integer"


class TestEdgeCases:
    def test_no_params_tool(self) -> None:
        @tool
        def noop() -> str:
            """Do nothing."""
            return "done"

        assert noop.name == "noop"
        assert noop.input_schema["type"] == "object"

    def test_multiple_types(self) -> None:
        @tool
        def process(name: str, count: int, active: bool, ratio: float) -> str:
            """Process data."""
            return "ok"

        schema = process.input_schema
        assert len(schema["properties"]) == 4
        assert set(schema["required"]) == {"name", "count", "active", "ratio"}

    def test_optional_param(self) -> None:
        @tool
        def fn(name: str, tag: str | None = None) -> str:
            """Fn."""
            return name

        schema = fn.input_schema
        assert "name" in schema["required"]
        assert "tag" not in schema.get("required", [])
