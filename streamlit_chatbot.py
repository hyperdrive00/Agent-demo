import streamlit as st
import pandas as pd
import time
import traceback
from typing import Generator, Any
import uuid

# Import AEM Agent
from AEMAgent import AEMAgent


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "aem_agent" not in st.session_state:
        st.session_state.aem_agent = None
    
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False


def get_agent_instance(model_name: str, api_mode: str, user_api_key: str = None, user_base_url: str = None) -> AEMAgent:
    """Get or create AEM Agent instance."""
    try:
        if api_mode == "free":
            agent = AEMAgent(
                llm_model_name=model_name,
                session_id=st.session_state.session_id,
                api_mode="free"
            )
        else:  # user mode
            if not user_api_key:
                raise ValueError("API key is required for user mode")
            agent = AEMAgent(
                llm_model_name=model_name,
                session_id=st.session_state.session_id,
                api_mode="user",
                user_api_key=user_api_key,
                user_base_url=user_base_url
            )
        return agent
    except Exception as e:
        st.error(f"Failed to initialize AEM Agent: {str(e)}")
        return None


def stream_agent_response(agent: AEMAgent, question: str) -> Generator[str, None, None]:
    """Stream responses from AEM Agent."""
    try:
        for response in agent.task_execution(question):
            if isinstance(response, pd.DataFrame):
                # Convert DataFrame to formatted string
                if not response.empty:
                    yield f"\n**Query Results:**\n\n{response.to_string(index=False)}\n\n"
                else:
                    yield "\n**No results found.**\n\n"
            else:
                yield str(response)
    except Exception as e:
        yield f"\n**Error:** {str(e)}\n"


def main():
    st.set_page_config(
        page_title="AEM Agent Chatbot",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 AEM Agent Chatbot")
    st.markdown("Ask questions about Anionic Exchange Membranes (AEMs) and their properties!")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Mode Selection
        api_mode = st.selectbox(
            "API Mode",
            options=["free", "user"],
            format_func=lambda x: "Free (OpenRouter)" if x == "free" else "User API Key",
            help="Choose between free OpenRouter models or your own API key"
        )
        
        # Model Selection
        if api_mode == "free":
            st.subheader("Free Models")
            # Use get_models_by_provider() which returns the grouped structure we need
            model_groups = AEMAgent.get_models_by_provider()
            
            # Model selection with grouping
            selected_model = None
            for provider, models in model_groups.items():
                with st.expander(f"📚 {provider.title()} Models"):
                    for model_id, description in models:
                        if st.button(f"Select {model_id.split('/')[-1]}", key=model_id):
                            selected_model = model_id
                            st.session_state.selected_model = model_id
                            break
            
            # Use session state to remember selection
            if "selected_model" not in st.session_state:
                st.session_state.selected_model = "deepseek/deepseek-chat-v3-0324:free"
            
            model_name = st.session_state.selected_model
            
            # Show current selection
            st.info(f"**Current Model:** {model_name}")
            
            user_api_key = None
            user_base_url = None
            
        else:  # user mode
            st.subheader("User API Settings")
            user_api_key = st.text_input(
                "API Key",
                type="password",
                help="Enter your OpenAI API key or compatible API key"
            )
            user_base_url = st.text_input(
                "Base URL (Optional)",
                placeholder="https://api.openai.com/v1",
                help="Leave empty for OpenAI, or enter custom base URL"
            )
            model_name = st.text_input(
                "Model Name",
                value="gpt-4o",
                help="Enter the model name (e.g., gpt-4o, gpt-3.5-turbo)"
            )
        
        # Initialize Agent button
        if st.button("🚀 Initialize Agent", type="primary"):
            with st.spinner("Initializing AEM Agent..."):
                agent = get_agent_instance(model_name, api_mode, user_api_key, user_base_url)
                if agent:
                    st.session_state.aem_agent = agent
                    st.session_state.agent_initialized = True
                    st.success("✅ Agent initialized successfully!")
                    st.rerun()
        
        # Show agent status
        if st.session_state.agent_initialized:
            st.success("✅ Agent Ready")
            st.info(f"**Model:** {model_name}")
            st.info(f"**Session ID:** {st.session_state.session_id[:8]}...")
        else:
            st.warning("⚠️ Agent not initialized")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.aem_agent = None
            st.session_state.agent_initialized = False
            st.rerun()
    
    # Main chat interface
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # For assistant messages, check if content contains DataFrame info
                content = message["content"]
                if "**Query Results:**" in content:
                    # Split content to show formatted results
                    parts = content.split("**Query Results:**")
                    if len(parts) > 1:
                        st.markdown(parts[0])
                        st.markdown("**Query Results:**")
                        st.text(parts[1])
                    else:
                        st.markdown(content)
                else:
                    st.markdown(content)
    
    # Chat input
    if prompt := st.chat_input("Ask about AEMs (e.g., 'What are the conductivity values for AEMs tested at 25°C?')"):
        if not st.session_state.agent_initialized:
            st.error("Please initialize the agent first using the sidebar.")
            st.stop()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream response from AEM Agent
            try:
                for response_chunk in stream_agent_response(st.session_state.aem_agent, prompt):
                    full_response += response_chunk
                    response_placeholder.markdown(full_response)
            except Exception as e:
                error_msg = f"Error processing your question: {str(e)}"
                st.error(error_msg)
                full_response = error_msg
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Footer with information
    st.markdown("---")
    st.markdown("""
    **About AEM Agent Chatbot:**
    - 🧪 Specialized in Anionic Exchange Membrane (AEM) research
    - 📊 Queries database with AEM properties and testing conditions
    - 🤖 Powered by advanced language models
    - 💡 Ask about conductivity, water uptake, swelling ratio, tensile strength, etc.
    
    **Sample Questions:**
    - "What are the conductivity values for AEMs tested at 25°C?"
    - "Show me AEMs with high tensile strength"
    - "Compare water uptake values across different membranes"
    """)


if __name__ == "__main__":
    main() 