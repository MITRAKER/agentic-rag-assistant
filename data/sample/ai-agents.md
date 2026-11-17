# AI Agents

An AI agent is a language model that can decide which action to take next,
call a tool, look at the result and repeat until the task is done.

## Tools

Tools are functions the agent may call, such as searching a vector database,
searching the web or calling a weather API. Each tool has a name, a description
and a schema for its arguments, so the model knows when and how to use it.

## Routing

A router is a simple form of agent. Given a question, it chooses the best
source: internal documents for company knowledge, web search for recent news,
or an API for live data such as prices.

## Self-checking

A verification step asks the model whether every claim in the draft answer is
supported by a retrieved source. Unsupported answers are retried with a better
search or returned with a warning.
