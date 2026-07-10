# PharmaGuard AI

A multi-agent, RAG-powered compliance co-pilot for pharmaceutical IT teams. Describe a task in plain English — get back applicable regulatory frameworks, cited requirements, deviation flags, and a structured Go/No-Go compliance brief.

## Problem

Pharma IT teams manually cross-reference multiple regulatory frameworks (GxP, 21 CFR Part 11, GAMP 5, ISO 13485, IEC 62304) every time a release, validation, or audit task begins. This is slow, inconsistent, and depends on specialist knowledge that doesn't scale.

## Solution

PharmaGuard AI automates this end to end using a 5-agent LangGraph pipeline grounded in a RAG knowledge base — every output is traceable to a specific regulatory source, not LLM guesswork.

## Architecture
Task description
|
Intake Agent -> classifies task type and scope
|
Framework Selector -> identifies applicable regulations
|
Requirements Agent -> RAG retrieval from ChromaDB per framework
|
Deviation Checker -> compares task against requirements, flags gaps
|
Report Generator -> structured brief: summary, dos, don'ts, recommendation

## Tech Stack

- **Orchestration:** LangGraph (StateGraph, conditional edges, MemorySaver)
- **LLM:** LangChain + Groq (llama-3.3-70b-versatile)
- **RAG:** ChromaDB + HuggingFace embeddings (all-MiniLM-L6-v2)
- **Language:** Python 3.10

## Setup

```bash
git clone <your-repo-url>
cd agentic-ai-learning
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with:

GROQ_API_KEY=your_key_here

Build the knowledge base:
```bash
python day09_chromadb.py
```

Run PharmaGuard AI:
```bash
python pharma_guard.py
```

## Example Output

Input: *"Releasing a new LIMS module v3.1 that handles electronic batch records with electronic signatures and audit trail capabilities"*

Output: Applicable frameworks identified, 7 requirements retrieved, risk level assessed, structured compliance brief with actionable recommendations.

## Author

Aditya — building AI systems that apply domain expertise to compliance automation.