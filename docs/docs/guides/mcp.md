# MCP Bridge Guide

The MCP (Model Context Protocol) bridge provides bidirectional integration
between anchor and the MCP ecosystem. You can consume tools from external MCP
servers and expose anchor capabilities as MCP tools, resources, and prompts.

## Overview

The bridge has two sides:

1. **Client bridge** -- connect to external MCP servers and use their tools
   within anchor's Agent
2. **Server bridge** -- expose anchor's tools, pipelines, and agents as MCP
   servers that other applications can consume

```
External MCP Servers          anchor Agent          External MCP Clients
        |                         |                         |
        v                         v                         v
  FastMCPClientBridge  -->  Agent.with_tools()    FastMCPServerBridge
        |                                                   |
        v                                                   v
  MCP tools become            anchor tools become
  AgentTool instances         MCP tools/resources/prompts
```

!!! note
    The MCP bridge requires the `fastmcp` package.
    Install with: `pip install astro-anchor[mcp]`

## Client Bridge

The `FastMCPClientBridge` connects to an external MCP server and adapts its
tools for use with anchor's Agent.

### Connecting to a Server

Configure the connection with `MCPServerConfig`, specifying either a command
(STDIO transport) or a URL (HTTP transport):

```python
import asyncio
from anchor.mcp import FastMCPClientBridge, MCPServerConfig

# STDIO transport (spawns a subprocess)
config = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

# HTTP transport
config = MCPServerConfig(
    name="my-api",
    url="http://localhost:8080/mcp",
    headers={"Authorization": "Bearer my-token"},
)

async def main():
    async with FastMCPClientBridge(config) as bridge:
        tools = await bridge.list_tools()
        print(f"Found {len(tools)} tools")

asyncio.run(main())
```

### Listing and Executing Tools

Once connected, you can discover and call tools directly:

```python
async with FastMCPClientBridge(config) as bridge:
    # List available tools
    tools = await bridge.list_tools()
    for tool in tools:
        print(f"  {tool.name}: {tool.description}")

    # Call a tool
    result = await bridge.call_tool("read_file", {"path": "/tmp/example.txt"})
    print(result)
```

### Resources and Prompts

The client bridge also supports MCP resources and prompt templates:

```python
async with FastMCPClientBridge(config) as bridge:
    # List and read resources
    resources = await bridge.list_resources()
    for r in resources:
        content = await bridge.read_resource(r.uri)
        print(f"{r.name}: {content[:100]}")

    # List and render prompts
    prompts = await bridge.list_prompts()
    rendered = await bridge.get_prompt("summarize", {"text": "Hello world"})
    print(rendered)
```

### Quick Config with parse_server_string

For convenience, you can parse a server string directly:

```python
from anchor.mcp import parse_server_string

# HTTP server
config = parse_server_string("http://localhost:8080/mcp")

# STDIO server (command + args)
config = parse_server_string("npx -y @modelcontextprotocol/server-filesystem /tmp")
```

## Server Bridge

The `FastMCPServerBridge` exposes anchor capabilities as an MCP server that
other applications (Claude Desktop, VS Code, etc.) can connect to.

### Exposing Tools

```python
import asyncio
from anchor import tool
from anchor.mcp import FastMCPServerBridge

@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

server = FastMCPServerBridge(name="my-tools")
server.expose_tools([greet])

asyncio.run(server.run(transport="stdio"))
```

### Exposing Resources and Prompts

```python
server = FastMCPServerBridge(name="my-server")

# Expose a resource
def get_status() -> str:
    return "System is running"

server.expose_resource("status://health", get_status)

# Expose a prompt template
def greeting_prompt(name: str) -> str:
    return f"Please greet {name} warmly."

server.expose_prompt("greet", greeting_prompt)
```

### From an Agent

Create a server that exposes all of an Agent's tools, plus pipeline and memory
resources:

```python
from anchor import Agent, tool
from anchor.mcp import FastMCPServerBridge

@tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return f"Results for: {query}"

agent = (
    Agent(model="claude-haiku-4-5-20251001")
    .with_system_prompt("You are a helpful assistant.")
    .with_tools([search])
)

server = FastMCPServerBridge.from_agent(agent)
# Exposes: search tool, context://pipeline, context://memory, chat prompt

import asyncio
asyncio.run(server.run(transport="stdio"))
```

### From a Pipeline

Create a server that exposes a pipeline as a query tool:

```python
from anchor.pipeline import ContextPipeline
from anchor.mcp import FastMCPServerBridge

pipeline = ContextPipeline(max_tokens=8192)
# ... configure pipeline ...

server = FastMCPServerBridge.from_pipeline(pipeline)
# Exposes: query tool, context://result resource

import asyncio
asyncio.run(server.run(transport="http"))
```

### Transport Options

The server supports three transports:

| Transport | Use Case |
|---|---|
| `"stdio"` | Local integrations (Claude Desktop, VS Code) |
| `"http"` | Remote HTTP connections (streamable HTTP) |
| `"sse"` | Server-Sent Events for real-time streaming |

## Client Pool

The `MCPClientPool` manages connections to multiple MCP servers and collects
all their tools into a single list.

```python
import asyncio
from anchor.mcp import MCPClientPool, MCPServerConfig

configs = [
    MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    ),
    MCPServerConfig(
        name="api",
        url="http://localhost:8080/mcp",
    ),
]

async def main():
    async with MCPClientPool(configs) as pool:
        tools = await pool.all_agent_tools()
        print(f"Collected {len(tools)} tools from {len(configs)} servers")

asyncio.run(main())
```

If any connection fails, the pool cleans up already-connected bridges before
raising the error.

## Agent Integration

The primary use case is feeding MCP tools into anchor's Agent. Since MCP tools
are async, you must use `agent.achat()` instead of `agent.chat()`.

```python
import asyncio
from anchor import Agent
from anchor.mcp import FastMCPClientBridge, MCPServerConfig

config = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

async def main():
    async with FastMCPClientBridge(config) as bridge:
        tools = await bridge.as_agent_tools()

        agent = (
            Agent(model="claude-haiku-4-5-20251001")
            .with_system_prompt("You can read and write files.")
            .with_tools(tools)
        )

        async for chunk in agent.achat("List the files in /tmp"):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### With Multiple Servers

Use `MCPClientPool` to combine tools from several servers:

```python
async def main():
    configs = [
        MCPServerConfig(name="files", command="npx",
                        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
        MCPServerConfig(name="api", url="http://localhost:8080/mcp"),
    ]

    async with MCPClientPool(configs) as pool:
        tools = await pool.all_agent_tools()

        agent = (
            Agent(model="claude-haiku-4-5-20251001")
            .with_system_prompt("You have access to files and an API.")
            .with_tools(tools)
        )

        async for chunk in agent.achat("What files are available?"):
            print(chunk, end="", flush=True)
```

### Tool Name Prefixing

By default, tool names are prefixed with the server name to prevent collisions
(e.g., `filesystem_read_file`). Disable this with `prefix_tools=False`:

```python
config = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    prefix_tools=False,  # Tools keep their original names
)
```

## Deferred Tools

`MCPServerConfig(defer_tools=True)` registers the server's tools with
`defer_loading=True`: their schemas stay out of the prompt until the
model loads them through the auto-registered `search_tools` meta-tool —
useful for servers with large tool surfaces.

## Prompts and Resources

MCP prompts are user-controlled and resources are application-controlled
(the reference-client pattern) — anchor exposes both to your
application and never injects them automatically:

```python
prompts = await agent.mcp_prompts()            # {server: [MCPPrompt]}
text = await agent.mcp_get_prompt("github", "pr_review", {"pr": "42"})
resources = await agent.mcp_resources()        # {server: [MCPResource]}
readme = await agent.mcp_read_resource("docs", "file://readme")
# your app decides: send `text` as a user message, attach `readme`, ...
```

The same surface exists on `MCPClientPool` directly.

## Error Handling

All MCP errors inherit from `MCPError`, which itself inherits from
`AstroContextError`. This lets you catch MCP errors specifically or handle
them alongside other anchor errors.

```python
from anchor.mcp import (
    FastMCPClientBridge, MCPServerConfig,
    MCPError, MCPConnectionError, MCPToolError,
    MCPTimeoutError, MCPConfigError,
)

config = MCPServerConfig(name="my-server", url="http://localhost:8080/mcp")

try:
    async with FastMCPClientBridge(config) as bridge:
        result = await bridge.call_tool("my_tool", {"arg": "value"})
except MCPConnectionError as e:
    print(f"Connection failed to {e.server_name}: {e}")
except MCPToolError as e:
    print(f"Tool {e.tool_name} failed on {e.server_name}: {e}")
except MCPTimeoutError as e:
    print(f"Timeout during {e.operation}: {e}")
except MCPConfigError as e:
    print(f"Configuration error: {e}")
except MCPError as e:
    print(f"General MCP error: {e}")
```

| Error | When Raised |
|---|---|
| `MCPConnectionError` | Failed to connect to an MCP server |
| `MCPToolError` | Error executing an MCP tool |
| `MCPTimeoutError` | MCP operation timed out |
| `MCPConfigError` | Invalid server configuration |

## See Also

- [MCP API Reference](../api/mcp.md) -- complete API signatures
- [Agent Guide](agent.md) -- using the Agent with tools
- [Agent API Reference](../api/agent.md) -- Agent class details
