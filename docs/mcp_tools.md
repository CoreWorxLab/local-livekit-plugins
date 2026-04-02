# MCP Server Usage in AgentSession

In the LiveKit Agents runtime, `mcp_servers` are used to provide external tools to a voice agent via the Model Context Protocol (MCP). Here's a breakdown of how they are integrated and used.

## 1. Initialization in `AgentSession`

`mcp_servers` is an optional argument in the `AgentSession` constructor. It is stored as a property and can be accessed by the session.

```python
# livekit/agents/voice/agent_session.py

class AgentSession(...):
    def __init__(
        self,
        ...,
        mcp_servers: NotGivenOr[list[mcp.MCPServer]] = NOT_GIVEN,
        ...
    ) -> None:
        ...
        self._mcp_servers = mcp_servers or None

    @property
    def mcp_servers(self) -> list[mcp.MCPServer] | None:
        return self._mcp_servers
```

## 2. Managing MCP Tools with `AgentActivity`

The `AgentSession` manages one or more `AgentActivity` instances. When an `AgentActivity` starts, it initializes the MCP servers and discovers their available tools.

```python
# livekit/agents/voice/agent_activity.py

class AgentActivity(...):
    async def _start_session(self) -> None:
        ...
        if self.mcp_servers:
            # Initialize each server and list tools
            gathered = await asyncio.gather(
                *(_list_mcp_tools_task(s) for s in self.mcp_servers),
                return_exceptions=True,
            )
            # Store discovered MCP tools
            self._mcp_tools = tools
```

`AgentActivity` prioritizes the `mcp_servers` defined on the `Agent` itself, falling back to the session's servers if none are provided by the agent.

```python
@property
def mcp_servers(self) -> list[mcp.MCPServer] | None:
    return (
        self._agent.mcp_servers
        if is_given(self._agent.mcp_servers)
        else self._session.mcp_servers
    )
```

## 3. Tool Discovery and Implementation

Each discovered MCP tool is wrapped as a `RawFunctionTool`. This allows them to be treated interchangeably with other function tools.

```python
# livekit/agents/llm/mcp.py

MCPTool = RawFunctionTool

class MCPServer(ABC):
    def _make_function_tool(self, ...) -> MCPTool:
        async def _tool_called(raw_arguments: dict[str, Any]) -> Any:
            # Forward the call to the actual MCP server
            tool_result = await self._client.call_tool(name, raw_arguments)
            return tool_result.content[0].model_dump_json()

        return function_tool(_tool_called, raw_schema=raw_schema)
```

## 4. Integration with LLM

The `AgentActivity` combines all available tools (session tools, agent tools, and MCP tools) into a single list that is provided to the LLM during inference.

```python
@property
def tools(self) -> list[llm.FunctionTool | llm.RawFunctionTool | mcp.MCPTool]:
    return self._session.tools + self._agent.tools + self._mcp_tools
```

## 5. Tool Layers in `AgentActivity`

`AgentActivity` combines tools from multiple sources to provide the LLM with a complete set of capabilities.

### 1. `self._session.tools`
These are **global tools** defined at the session level.
- **Where they come from**: They are passed into the `AgentSession` constructor.
- **Scope**: They are shared across all agents that might run within that session (e.g., if you hand off from one agent to another, these tools remain available).
- **Implementation**: In `AgentSession`, these are stored in `self._tools`.

### 2. `self._agent.tools`
These are **agent-specific tools**.
- **Where they come from**: 
    1. Tools passed directly into the `Agent` constructor.
    2. Methods within the `Agent` class (or its subclasses) that are decorated with `@llm.function_tool`. The `Agent` class automatically discovers these using `find_function_tools(self)`.
- **Scope**: They are only available while that specific agent is active.
- **Implementation**: In `Agent`, these are stored in `self._tools`.

### How they are used together
In `AgentActivity.tools`, these are combined with any discovered MCP tools to create the final list of tools provided to the LLM:

## 6. LLM Tool Integration: Prompting and Decision-Making

When the agent interacts with an LLM, the tools are not hidden in a text prompt; they are provided as a core part of the model's interface.

### How Tools are "Prompted"
Tools are not added to the text prompt (like the system instructions). Instead, they are converted into a structured format (JSON Schema) and sent as a specialized `tools` parameter in the API call to the LLM provider (e.g., OpenAI).

```python
# livekit/agents/inference/llm.py

def to_fnc_ctx(fnc_ctx: list[llm.FunctionTool | llm.RawFunctionTool]) -> list[ChatCompletionToolParam]:
    # Converts tools into OpenAI-compatible JSON Schema
    return [
        {
            "type": "function",
            "function": fnc.raw_schema if is_raw(fnc) else build_openai_schema(fnc)
        } for fnc in fnc_ctx
    ]
```

### When the LLM Decides to Use Tools
The decision-making happens entirely within the LLM model itself. 
1. **Model Reasoning**: Based on the user's request and the tool descriptions provided in the schema, the model determines if a tool is needed.
2. **Tool Call Generation**: If needed, the model returns a special `tool_calls` message instead of (or in addition to) regular text.
3. **Detection**: The `LLMStream` in the LiveKit runtime parses these tool calls from the API response:

```python
# livekit/agents/inference/llm.py

def _parse_choice(self, choice: Choice) -> llm.ChatChunk | None:
    if choice.delta.tool_calls:
        # Detected that the LLM wants to call a tool
        return llm.ChatChunk(tool_calls=...)
```

## 7. Deep Dive: `_execute_tools_task` Workflow

The `_execute_tools_task` function in `livekit/agents/voice/generation.py` is the core engine for tool execution. It handles everything from argument validation to parallel task management.

### Detailed Workflow Stages

#### 1. Setup and Event Listening
The function initializes a list of `tasks` and defines a `_tool_completed` helper. It starts an `async for` loop over `function_stream`, which yields tool calls as they are parsed from the LLM's response. This allow the agent to start executing the first tool even while the LLM is still generating the second one.

#### 2. Discovery and Type Safety
For each incoming call:
- It looks up the function name in the `ToolContext`.
- It verifies that the tool choice allows for execution (e.g., not `tool_choice="none"`).
- It validates that the tool is a supported type (`FunctionTool` or `RawFunctionTool`).

#### 3. Intelligent Argument Preparation (Dependency Injection)
`llm_utils.prepare_function_arguments` is called to:
- **Parse the LLM's JSON**: Converts the string arguments into Python types.
- **Dependency Injection**: If the tool's signature includes special types like `RunContext`, `AgentSession`, or `SpeechHandle`, the utility automatically injects the current instances of these objects. This makes it easy for developers to access the current session state inside their tools.

#### 4. Asynchronous Task Spawning
Instead of executing tools sequentially, the function creates a sub-task for each call using `asyncio.create_task(_traceable_fnc_tool(...))`. 
- **Parallelism**: This allows multiple tools to run simultaneously if requested by the LLM.
- **Monitoring**: It uses `_set_activity_task_info` to mark these tasks as "inline tasks," allowing the runtime to track tool-driven state changes.

#### 5. Observability and Tracing
Each tool execution is wrapped in an **OpenTelemetry span** via `tracer.start_as_current_span("function_tool")`. This records:
- The tool's name and arguments.
- The return value or any exception that occurred.
- Performance metrics (how long the execution took).

#### 6. Graceful Interruption and Shielding
When the function finishes its loop, it waits for all spawned tasks with `asyncio.gather`. 
- **Shielding**: It uses `asyncio.shield` to ensure that if the agent session is cancelled (e.g., due to user interruption), currently running tool calls are allowed to finish. This prevents leaving the system in an inconsistent state.

#### 7. Result Reporting
The result of each tool is captured and sent back to the `AgentActivity` via callbacks. These results are ultimately fed back into the LLM's conversation history as `FunctionCallOutput` messages.



