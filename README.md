# Search-Engine-using-tools-and-agents
## Overview 
The agent retrieves real-time information before answering rather than relying on model memory.
## How It Works 
User Query → Agent decides tool → Tool fetches data → LLM synthesizes answer → Response shown with tool steps visible.
## Tools Used 
Wikipedia for factual lookups, Arxiv for academic/research queries, DuckDuckGo for general web search.
## Tech Stack 
LangChain, Groq API, Qwen3-32B, Streamlit, Python.
## Features 
Real-time tool use, visible intermediate steps, streaming responses, configurable via sidebar API key input, handles parsing errors gracefully.
## Setup & Installation 
Clone repo, create .env with GROQ_API_KEY, pip install -r requirements.txt, streamlit run app.py.
## Demo 
https://search-engine2.streamlit.app/
