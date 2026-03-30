import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search = DuckDuckGoSearchRun()

st.title("🔎 Search Engine with LangChain and Groq")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input(placeholder='Ask me anything...'):

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="qwen/qwen3-32b", streaming=True)
    tools = [wikipedia, arxiv, search]

    # ✅ Strict prompt that forces tool usage
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant.

        IMPORTANT RULES:
        - You MUST ALWAYS use maximum tools before answering.
        - NEVER answer from your own knowledge directly."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt_template)

        # ✅ return_intermediate_steps enabled
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        response = agent_executor.invoke(
            {"input": prompt, "chat_history": st.session_state.messages},
            callbacks=[st_cb]
        )

        # ✅ Display tool usage info
        steps = response.get("intermediate_steps", [])
        if steps:
            st.success(f"✅ Agent used {len(steps)} tool(s)!")
            for i, (action, result) in enumerate(steps):
                with st.expander(f"🔧 Tool Call {i+1}: `{action.tool}`"):
                    st.write("**Input:**", action.tool_input)
                    st.write("**Output:**", result)
        else:
            st.warning("⚠️ No tools were used — LLM answered from memory.")

        final_response = response["output"]
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.write(final_response)