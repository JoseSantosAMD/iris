# GitHub Agents

This directory contains agent configurations for GitHub Copilot to provide specialized assistance for this repository.

## Available Agents

### ROCminfo Validator (`rocminfo-validator.md`)

An agent that validates commits by checking if rocminfo works correctly in CI workflows.

**Use it by asking:**
- "Is this commit good?"
- "Did rocminfo work in the latest CI run?"
- "What GPUs were detected?"
- "Check if the MCP server is working"

**What it checks:**
- ✅ ROCm installation verification
- ✅ MCP server startup
- ✅ rocminfo function execution
- ✅ GPU detection

**How it works:**
The agent reads the workflow logs from the "ROCminfo MCP Server" workflow and looks for success/failure indicators to determine if the commit properly supports ROCm functionality.

## How to Use Agents

1. **In a PR or commit**: Ask GitHub Copilot to reference the agent
   - "Check if this commit is good using the rocminfo validator"
   - "@workspace is this commit good?"

2. **In GitHub Copilot Chat**: The agent instructions are automatically available as context when you ask relevant questions

3. **In code reviews**: Use the agent to quickly validate ROCm functionality without manually checking logs

## Creating New Agents

To add a new agent:
1. Create a new `.md` file in this directory
2. Define the agent's purpose and instructions
3. Specify what it should check and how to respond
4. Update this README with the new agent

## Tips

- Agents work best with specific, targeted questions
- Reference the agent explicitly if needed: "Use the rocminfo validator to check this PR"
- Agents can read workflow logs, PR comments, and repository files
- Keep agent instructions clear and actionable
