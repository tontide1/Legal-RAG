---
name: debugging-strategies
description: Systematic problem-solving for elusive bugs and performance issues. Use this skill when tracking down bugs, investigating production incidents, or analyzing stack traces.
---

# Debugging Strategies

## Overview

Transform debugging from guesswork into systematic problem-solving using proven strategies, powerful tools, and methodical approaches. Focus on reproducing issues, forming hypotheses, and controlled experimentation.

## Debugging Workflow

- Reproduce the issue consistently and capture logs, traces, and environment details.
- Form testable hypotheses and design controlled experiments to isolate root causes.
- Narrow the search scope using binary search and targeted instrumentation.
- Document findings and verify fixes across different environments.
- Use `references/implementation-playbook.md` for detailed debugging patterns and checklists.

## Key Techniques

- **Isolate the Problem**: Create minimal reproductions and remove unrelated code.
- **Gather Information**: Capture full stack traces, error codes, and environment variables.
- **Binary Search Debugging**: Use `git bisect` or comment out code sections to find regressions.
- **Differential Debugging**: Compare working vs broken environments/users/browsers.
- **Logging & Instrumentation**: Use strategic logging and tracing to track variable values and execution flow.

## Resources

- `references/implementation-playbook.md`: Detailed patterns, checklists, and code samples for various programming languages (JS/TS, Python, Go).
