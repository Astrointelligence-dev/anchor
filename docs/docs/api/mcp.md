# MCP API Reference

The `anchor.mcp` module provides bidirectional Model Context Protocol (MCP)
integration. It lets you consume external MCP servers from anchor's Agent
and expose anchor capabilities as MCP tools, resources, and prompts.

All classes are importable from `anchor.mcp`:

```python
from anchor.mcp import (
    FastMCPClientBridge, FastMCPServerBridge,
    MCPClientPool,
    MCPServerConfig, MCPResource, MCPPrompt, MCPPromptArgument,
    MCPClient, MCPServer,
    mcp_tool_to_agent_tool, parse_server_string,
    MCPError, MCPConnectionError, MCPTimeoutError,
    MCPConfigError, MCPToolError,
)
```

!!! note
    The MCP bridge requires the `fastmcp` package.
    Install with: `pip install astro-anchor[mcp]`

---

## FastMCPClientBridge

Connects to an external MCP server using FastMCP's Client with auto-inferred
transport. Implements the `MCPClient` protocol. Adapts MCP tools to anchor's
`AgentTool` format for seamless Agent integration.

### Constructor

```python
class FastMCPClientBridge:
    def __init__(self, config: MCPServerConfig) -> None
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `config` | `MCPServerConfig` | Server connection configuration |

### Methods

#### connect

```python
async def connect(self) -> None
```

Connect to the MCP server using auto-inferred transport (STDIO or HTTP based
on the config). Raises `MCPConnectionError` on failure.

#### disconnect

```python
async def disconnect(self) -> None
```

Close the connection. Logs errors but does not raise.

#### list_tools

```python
async def list_tools(self) -> list[ToolSchema]
```

List available tools from the MCP server. Results are cached if
`config.cache_tools` is `True`.

#### call_tool

```python
async def call_tool(self, name: str, arguments: dict[str, Any]) -> str
```

Execute a tool on the MCP server. Returns the result as a string.
Raises `MCPToolError` on failure.

#### list_resources

```python
async def list_resources(self) -> list[MCPResource]
```

List resources from the MCP server.

#### read_resource

```python
async def read_resource(self, uri: str) -> str
```

Read a resource by URI. Returns its content as a string.

#### list_prompts

```python
async def list_prompts(self) -> list[MCPPrompt]
```

List prompt templates from the MCP server.

#### get_prompt

```python
async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str
```

Get a rendered prompt template.

#### as_agent_tools

```python
async def as_agent_tools(self) -> list[AgentTool]
```

Convert all MCP tools to `AgentTool` instances for use with `Agent.with_tools()`.
Tool names are prefixed with the server name when `config.prefix_tools` is `True`.

### Async Context Manager

`FastMCPClientBridge` supports `async with` for automatic connect/disconnect:

```python
async with FastMCPClientBridge(config) as bridge:
    tools = await bridge.as_agent_tools()
```

---

## FastMCPServerBridge

Exposes anchor capabilities as an MCP server via FastMCP. Implements the
`MCPServer` protocol.

### Constructor

```python
class FastMCPServerBridge:
    def __init__(self, name: str = "anchor") -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"anchor"` | Server name |

### Methods

#### expose_tool

```python
def expose_tool(self, tool: AgentTool) -> None
```

Register a single `AgentTool` as an MCP tool.

#### expose_tools

```python
def expose_tools(self, tools: list[AgentTool]) -> None
```

Register multiple tools.

#### expose_resource

```python
def expose_resource(
    self,
    uri: str,
    handler: Callable[..., str] | Callable[..., Any],
) -> None
```

Register a resource at the given URI pattern.

#### expose_prompt

```python
def expose_prompt(
    self,
    name: str,
    handler: Callable[..., str] | Callable[..., Any],
) -> None
```

Register a prompt template.

#### from_agent (classmethod)

```python
@classmethod
def from_agent(cls, agent: Agent) -> FastMCPServerBridge
```

Create a server from an `Agent`, exposing all its tools. Also exposes:

- Resource `context://pipeline` -- pipeline configuration
- Resource `context://memory` -- memory state
- Prompt `chat` -- send a message through the agent

#### from_pipeline (classmethod)

```python
@classmethod
def from_pipeline(cls, pipeline: ContextPipeline) -> FastMCPServerBridge
```

Create a server exposing a pipeline as retrieval tools. Exposes:

- Tool `query` -- run the pipeline and return results
- Resource `context://result` -- last pipeline result

#### run

```python
async def run(self, transport: str = "stdio") -> None
```

Start the MCP server. Supported transports: `"stdio"`, `"http"`, `"sse"`.

---

## MCPClientPool

Manages connections to multiple MCP servers. Convenience class for connecting
to several servers at once and collecting all their tools as `AgentTool` instances.

### Constructor

```python
class MCPClientPool:
    def __init__(self, configs: list[MCPServerConfig]) -> None
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `configs` | `list[MCPServerConfig]` | List of server configurations |

### Methods

#### connect_all

```python
async def connect_all(self) -> None
```

Connect to all configured servers concurrently. If any connection fails,
already-connected bridges are cleaned up before re-raising the first error.

#### disconnect_all

```python
async def disconnect_all(self) -> None
```

Disconnect all servers.

#### all_agent_tools

```python
async def all_agent_tools(self) -> list[AgentTool]
```

Collect `AgentTool` instances from all connected servers.

### Async Context Manager

```python
async with MCPClientPool(configs) as pool:
    tools = await pool.all_agent_tools()
```

---

## MCPServerConfig

Frozen Pydantic model for configuring a connection to an external MCP server.
Requires either `command` (STDIO) or `url` (HTTP) -- not both.

```python
class MCPServerConfig(BaseModel, frozen=True):
    name: str
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    env: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    cache_tools: bool = True
    prefix_tools: bool = True
    timeout: float | None = 30.0
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique server name. Used for tool name prefixing. |
| `command` | `str \| None` | `None` | STDIO transport: command to spawn (e.g., `"npx"`, `"python"`) |
| `args` | `list[str] \| None` | `None` | STDIO transport: command arguments |
| `url` | `str \| None` | `None` | HTTP/SSE transport: server URL |
| `env` | `dict[str, str] \| None` | `None` | Environment variables passed to the server process |
| `headers` | `dict[str, str] \| None` | `None` | HTTP headers for authentication |
| `cache_tools` | `bool` | `True` | Cache the tool list after first discovery |
| `prefix_tools` | `bool` | `True` | Prefix tool names with server name to prevent collisions |
| `timeout` | `float \| None` | `30.0` | Timeout in seconds for MCP operations. `None` disables timeout. |

---

## MCPResource

Frozen Pydantic model describing an MCP resource.

```python
class MCPResource(BaseModel, frozen=True):
    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None
```

---

## MCPPrompt

Frozen Pydantic model describing an MCP prompt template.

```python
class MCPPrompt(BaseModel, frozen=True):
    name: str
    description: str | None = None
    arguments: list[MCPPromptArgument] | None = None
```

---

## MCPPromptArgument

Frozen Pydantic model for a single argument of an MCP prompt template.

```python
class MCPPromptArgument(BaseModel, frozen=True):
    name: str
    description: str | None = None
    required: bool = False
```

---

## MCPClient (Protocol)

Runtime-checkable protocol for consuming external MCP servers. Use this as
the type hint when accepting any MCP client implementation.

```python
@runtime_checkable
class MCPClient(Protocol):
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def list_tools(self) -> list[ToolSchema]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...
    async def list_resources(self) -> list[MCPResource]: ...
    async def read_resource(self, uri: str) -> str: ...
    async def list_prompts(self) -> list[MCPPrompt]: ...
    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...
```

---

## MCPServer (Protocol)

Runtime-checkable protocol for exposing anchor capabilities as an MCP server.

```python
@runtime_checkable
class MCPServer(Protocol):
    def expose_tool(self, tool: AgentTool) -> None: ...
    def expose_tools(self, tools: list[AgentTool]) -> None: ...
    def expose_resource(self, uri: str, handler: Callable[..., Any]) -> None: ...
    def expose_prompt(self, name: str, handler: Callable[..., Any]) -> None: ...
    async def run(self, transport: Literal["stdio", "sse", "http"] = "stdio") -> None: ...
```

---

## Utility Functions

### mcp_tool_to_agent_tool

```python
def mcp_tool_to_agent_tool(
    *,
    schema: ToolSchema,
    async_caller: Callable[[str, dict[str, Any]], Coroutine[Any, Any, str]],
    server_name: str,
    prefix: bool,
) -> AgentTool
```

Convert an MCP `ToolSchema` into an anchor `AgentTool`. The returned tool's
synchronous `fn` raises `MCPError` -- MCP tools require async execution via
`agent.achat()`. The async caller is attached as `_mcp_async_caller` for use
by `Agent._aexecute_tool()`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `schema` | `ToolSchema` | MCP tool schema to convert |
| `async_caller` | `Callable` | Async function that calls the tool on the MCP server |
| `server_name` | `str` | Server name for tool name prefixing |
| `prefix` | `bool` | Whether to prefix tool names with the server name |

### parse_server_string

```python
def parse_server_string(server: str) -> MCPServerConfig
```

Parse a convenience string into an `MCPServerConfig`:

- Strings starting with `http://` or `https://` become URL configs
- Other strings become command + args configs (split by whitespace)

Raises `MCPError` if the string is empty.

**Examples**

```python
from anchor.mcp import parse_server_string

# HTTP server
config = parse_server_string("http://localhost:8080/mcp")

# STDIO server
config = parse_server_string("npx -y @modelcontextprotocol/server-filesystem /tmp")
```

---

## Errors

All MCP errors inherit from `MCPError`, which itself inherits from
`AstroContextError`.

### MCPError

```python
class MCPError(AstroContextError)
```

Base error for all MCP operations.

### MCPConnectionError

```python
class MCPConnectionError(MCPError):
    server_name: str
    transport: str
```

Failed to connect to an MCP server. Carries the server name and transport type.

### MCPToolError

```python
class MCPToolError(MCPError):
    tool_name: str
    server_name: str
    cause: BaseException | None
```

Error executing an MCP tool. Carries the tool name, server name, and
underlying cause.

### MCPTimeoutError

```python
class MCPTimeoutError(MCPError):
    server_name: str
    operation: str
```

MCP server operation timed out.

### MCPConfigError

```python
class MCPConfigError(MCPError)
```

Invalid MCP server configuration.

---

## See Also

- [MCP Guide](../guides/mcp.md) -- usage guide with examples
- [Agent API Reference](agent.md) -- the Agent that consumes MCP tools
- [LLM API Reference](llm.md) -- provider protocol and tool models
- [Protocols Reference](protocols.md) -- extension point protocols
