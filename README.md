# Complete Agentic AI 🤖

Learning and building **Agentic AI applications** using Python, LangChain, LangGraph, Pydantic, and other AI tools.

## 🚀 Getting Started

### Activate Environment

**CMD:**
```cmd
.venv\Scripts\activate
```

### Sync Dependencies

```bash
uv sync
```

### Start Jupyter

```bash
jupyter notebook
```

## 📦 uv Commands

Add package:
```bash
uv add pydantic
```

Remove package:
```bash
uv remove pydantic
```

Check packages:
```bash
uv pip list
```

## 📁 Important Files

- `pyproject.toml` → Dependencies
- `uv.lock` → Locked versions
- `.venv` → Virtual environment

## 🌐 Run Streamlit App

```bash
uv run streamlit run 5-Single-AI-Agent/4-app.py
```

## 🌐 To show new Kernel in VSCODE

```bash
uv run python -m ipykernel install --user --name 6-multi-ai-agent --display-name "Python 3.12 (6-Multi-AI-Agent)"
```

## 🌐 Run Streamlit App for Multi Agent System

```bash
uv run streamlit run app.py
```

## 🌐 Change directory for multi agent kernel

```bash
cd "C:\Robin\AI course\complete-agentic-ai\6-Multi-AI-Agent"
```


## 🌐 Change directory for running streamlit

```bash
cd 6-Multi-AI-Agent
```

```bash
cd 13-End-To-End-Agent
```

```bash
streamlit run app_db.py
```

The overall flow is:

                    app_db.py
                       │
                       ▼
                 ┌───────────┐
                 │  backend  │
                 └───────────┘
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
     config           llm             state
       │               │                │
       │               ▼                │
       │          OpenAI/Cohere         │
       │                                │
       └───────────────┬────────────────┘
                       ▼
                    graph.py
                       │
                       ▼
                  LangGraph
                       │
                       ▼
                  database.py
                       │
                       ▼
                   chatbot.db