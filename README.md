# 🧠 GenAI Project – LangChain & LangGraph Applications

This repository contains **Generative AI applications built using LangChain and LangGraph**, showcasing both **lightweight (mini)** and **scalable (large)** architectures.  
The project focuses on **LLM orchestration, agent workflows, long-term memory, tool usage, and MCP server integration**.

---

## 🚀 Overview

The goal of this project is to explore how modern GenAI systems can be built using:
- **LangChain** for LLM interactions and tool calling
- **LangGraph** for structured, stateful agent workflows
- **Databases** for long-term memory as well as for short term memory
- **MCP (Model Context Protocol) Server** for standardized tool and context management

The repository demonstrates how the same AI system can scale from a **simple prototype** to a **production-style architecture**.

---

## 🧩 Project Structure


---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **LangGraph**
- **LLMs (OpenAI / compatible providers)**
- **SQLite** – lightweight long-term memory
- **PostgreSQL** – scalable persistent memory
- **MCP (Model Context Protocol) Server**
- **REST / Tool-based APIs**

---

## 🔹 Mini App

The **mini app** is designed as a **proof-of-concept**:

- Uses **LangChain + LangGraph**
- Stateless or short-term memory
- Simple agent flow
- Fast to run and easy to understand

**Use case:**  
Ideal for learning, experimentation, and rapid prototyping.

---

## 🔹 Large App

The **large app** demonstrates a **scalable GenAI system**:

- Complex **LangGraph-based agent workflows**
- **Long-term memory** using:
  - SQLite (local development)
  - PostgreSQL (production-ready)
- Tool-augmented reasoning
- MCP server for structured context and tool access

**Use case:**  
Designed for real-world, persistent, and multi-step AI applications.

---

## 🧠 Long-Term Memory

The system supports **persistent conversational and contextual memory**:

- **SQLite**  
  - Lightweight
  - Easy local setup
  - Ideal for development & testing

- **PostgreSQL**  
  - Scalable and durable
  - Suitable for production workloads
  - Supports long-running agent memory

Memory is used to store:
- Conversation history
- Agent state
- User context
- Tool outputs (when required)

---

## 🔧 Tools Integration

Agents can dynamically use tools for enhanced reasoning, such as:
- Database queries
- External API calls
- Search & retrieval
- Custom utility functions

Tools are integrated via **LangChain tool interfaces** and orchestrated through **LangGraph**.

---

## 🔌 MCP Server Integration

This project includes **MCP (Model Context Protocol) Server** support to:
- Standardize context passing
- Centralize tool management
- Improve modularity and scalability
- Enable clean separation between agents and external capabilities

MCP helps make the system **more maintainable and extensible**.

---
