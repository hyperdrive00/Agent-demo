# DesAgent.py
import time
import os
import sqlite3
import pandas as pd
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

import re
import json
from json_repair import repair_json
from typing import List, Dict, Any
import streamlit as st

from prompts import (
    SQL_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    FEEWSHOT_EXAMPLES,
    EXAMPLE_OUTPUT_PROMPT,
    CONVERT_SQL_SYSTEM_PROMPT
)

# Database configuration
DB_PATH = "extracted_data.db"  # Path to your SQLite database

# OpenRouter configuration for free tier
OPENROUTER_FREE_API_KEY = st.secrets.get('OPENROUTER_FREE_API_KEY', None)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter Free Tier Models (updated list as of 2025)
OPENROUTER_FREE_MODELS = {
    # DeepSeek Models
    "deepseek/deepseek-chat-v3-0324:free": "DeepSeek Chat V3 - Excellent for coding and general tasks",
    "deepseek/deepseek-r1:free": "DeepSeek R1 - Advanced reasoning model with open reasoning tokens",
    "deepseek/deepseek-r1-zero:free": "DeepSeek R1 Zero - RL-trained model without SFT",
    "deepseek/deepseek-r1-0528-qwen3-8b:free": "DeepSeek R1 0528 Qwen3 8B - Latest reasoning model",
    "deepseek/deepseek-r1-distill-llama-70b:free": "DeepSeek R1 Distill Llama 70B - High performance distilled model",
    "deepseek/deepseek-r1-distill-qwen-32b:free": "DeepSeek R1 Distill Qwen 32B - Strong reasoning capabilities",
    "deepseek/deepseek-r1-distill-qwen-14b:free": "DeepSeek R1 Distill Qwen 14B - Balanced performance",
    "deepseek/deepseek-r1-distill-qwen-7b:free": "DeepSeek R1 Distill Qwen 7B - Efficient reasoning model",
    "deepseek/deepseek-r1-distill-qwen-1.5b:free": "DeepSeek R1 Distill Qwen 1.5B - Lightweight yet capable",
    "deepseek/deepseek-r1-distill-llama-8b:free": "DeepSeek R1 Distill Llama 8B - Good balance of size and performance",
    "deepseek/deepseek-v3-base:free": "DeepSeek V3 Base - 671B parameter base model",
    
    # Meta Llama Models
    "meta-llama/llama-4-scout:free": "Llama 4 Scout - Multimodal MoE model with 200K context",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct - High-quality instruction following",
    
    # Google Models
    "google/gemini-2.5-pro-exp-03-25:free": "Gemini 2.5 Pro Experimental - Latest Google model",
    "google/gemini-2.0-flash-thinking-exp:free": "Gemini 2.0 Flash Thinking - Fast reasoning model",
    "google/gemini-2.0-flash-exp:free": "Gemini 2.0 Flash Experimental - Quick responses",
    "google/gemma-3-27b-it:free": "Gemma 3 27B IT - Google's open model",
    
    # Qwen Models
    "qwen/qwen3-4b:free": "Qwen3 4B - Dual-mode architecture with 128K context",
    "qwen/qwen3-0.6b-04-28:free": "Qwen3 0.6B - Lightweight model with 32K context",
    "qwen/qwq-32b:free": "QwQ 32B - Specialized reasoning model",
    
    # NVIDIA Models
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": "Llama 3.1 Nemotron Ultra 253B - NVIDIA's flagship model"
}

# Default fallback model
DEFAULT_FREE_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Default configuration for user API
DEFAULT_BASE_URL = None
DEFAULT_MODEL = "gpt-4o"

# Database schema information
DB_SCHEMA = """
Table: extracted_data
Columns:
- document_title (TEXT): The title/name of the document containing the AEM data
- aem_name (TEXT): The name of the Anionic Exchange Membrane
- oc_oh_conductivity (TEXT): OH⁻ conductivity with units (e.g., "123 (mS cm-1)")
- oc_testing_temperature (TEXT): Testing temperature for conductivity (e.g., "25 (°C)")
- sr_swelling_ratio (TEXT): Swelling ratio with units (e.g., "15 (%)")
- sr_testing_temperature (TEXT): Testing temperature for swelling ratio (e.g., "80 (°C)")
- wu_water_uptake (TEXT): Water uptake with units (e.g., "25 (%)")
- wu_testing_temperature (TEXT): Testing temperature for water uptake (e.g., "60 (°C)")
- ts_tensile_strength (TEXT): Tensile strength with units (e.g., "42 (MPa)")

Note: Property values are stored as text strings that include both the numeric value and units.
"""

class FlexibleJsonOutputParser(BaseOutputParser):
    """Custom JSON parser that handles malformed JSON by attempting repair."""
    
    def parse(self, text: str) -> dict:
        """Parse text to JSON, attempting repair if needed."""
        try:
            # Try direct JSON parsing first
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Attempt to repair malformed JSON
                repaired = repair_json(text)
                return json.loads(repaired)
            except Exception as e:
                # If all else fails, try to extract JSON from text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except:
                        pass
                
                # Return a default error structure
                return {
                    "thought_process": f"Failed to parse JSON: {text}",
                    "use_sql": "no",
                    "error": str(e)
                }

class AEMAgent:
    """AEM Agent for querying SQLite database using LangChain."""
    
    def __init__(self, llm_model_name=None, session_id=None, api_mode="free", user_api_key=None, user_base_url=None, db_path=None):
        """
        Initialize the AEMAgent with LLM model and session details.

        Args:
            llm_model_name (str, optional): Name of the language model to use. 
                                          For free mode: choose from OPENROUTER_FREE_MODELS.keys() or None for default.
                                          For user mode: any model supported by the API.
            session_id (str, optional): Session identifier. Defaults to "global".
            api_mode (str): Either "free" for OpenRouter free tier or "user" for user-provided API key.
            user_api_key (str, optional): User's API key when api_mode is "user".
            user_base_url (str, optional): User's base URL when api_mode is "user".
            db_path (str, optional): Path to SQLite database file. Defaults to "extracted_data.db".
        """
        self.api_mode = api_mode
        self.user_api_key = user_api_key
        self.user_base_url = user_base_url
        self.db_path = db_path or DB_PATH
        
        # Configure API settings based on mode
        if api_mode == "free":
            # Validate model selection for free mode
            if llm_model_name:
                # Check if model is in our known free models list
                if llm_model_name not in OPENROUTER_FREE_MODELS:
                    # If not in our list, check if it's at least formatted correctly as a free model
                    if not llm_model_name.endswith(":free"):
                        raise ValueError(f"Invalid free model '{llm_model_name}'. Model must end with ':free' for free tier usage.")
                    elif "/" not in llm_model_name:
                        raise ValueError(f"Invalid model format '{llm_model_name}'. Expected format: 'provider/model-name:free'")
                    else:
                        # Model format looks correct but not in our list - warn but allow
                        print(f"Warning: Model '{llm_model_name}' not in known free models list. Attempting to use anyway...")
            
            self.llm_model_name = llm_model_name or DEFAULT_FREE_MODEL
            self.base_url = OPENROUTER_BASE_URL
            self.api_key = OPENROUTER_FREE_API_KEY
            if not self.api_key:
                raise ValueError("OpenRouter free API key not found in secrets. Please configure OPENROUTER_FREE_API_KEY.")
        elif api_mode == "user":
            self.llm_model_name = llm_model_name or DEFAULT_MODEL
            self.base_url = user_base_url or DEFAULT_BASE_URL
            self.api_key = user_api_key
            if not self.api_key:
                raise ValueError("User API key is required when api_mode is 'user'.")
        else:
            raise ValueError("api_mode must be either 'free' or 'user'")
            
        self.session_id = session_id or "global"  # fallback
        self.log_dir = "chat_logs"
        self.log_file = f"./{self.log_dir}/chat_log_{self.session_id}.txt"
        
        # Make directory if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.CHAT_HISTORY = ChatMessageHistory()
        self.CHAT_HISTORY_FILE_PATH = "chat_history/chat_history.txt"
        
        # Test database connection
        self.test_db_connection()
        
        self.fewshot_examples = FEEWSHOT_EXAMPLES
        self.example_output_prompt = EXAMPLE_OUTPUT_PROMPT
        self.sql_system_prompt = SQL_SYSTEM_PROMPT
        self.answer_system_prompt = ANSWER_SYSTEM_PROMPT
        self.convert_sql_system_prompt = CONVERT_SQL_SYSTEM_PROMPT
        # Escape curly braces in schema to prevent template variable interpretation
        self.schema = DB_SCHEMA.replace("{", "{{").replace("}", "}}")
        
        self.sql_agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.sql_system_prompt),
                ("system", "{fewshot_examples}"),
                ("system", self.example_output_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", self.schema),
                ("human", "{question}"),
            ]
        )
        
        # Configure LLM clients based on API mode
        llm_kwargs = {
            "model": self.llm_model_name,
            "temperature": 0,
            "api_key": self.api_key,
        }
        
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
            
        # For OpenRouter, we need to handle JSON mode differently
        if api_mode == "free":
            # OpenRouter free tier may not support JSON mode reliably
            try:
                self.sql_llm = ChatOpenAI(**llm_kwargs)
                self.answer_llm = ChatOpenAI(**llm_kwargs)
                # Test with a simple query
                self.sql_llm.invoke("test")
            except Exception as e:
                print(f"Warning: Error with OpenRouter setup: {e}")
                # Fallback to basic configuration
                self.sql_llm = ChatOpenAI(**llm_kwargs)
                self.answer_llm = ChatOpenAI(**llm_kwargs)
        else:
            # User API - assume it supports standard OpenAI features
            self.sql_llm = ChatOpenAI(**llm_kwargs)
            self.answer_llm = ChatOpenAI(**llm_kwargs)
        
        # Create chains
        self.json_parser = FlexibleJsonOutputParser()
        
        # SQL generation chain
        self.sql_chain = (
            self.sql_agent_prompt
            | self.sql_llm
            | self.json_parser
        )
        
        # SQL optimization chain  
        self.convert_sql_chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.convert_sql_system_prompt),
                ("human", "{sql_query}")
            ])
            | self.sql_llm
            | self.json_parser
        )
        
        # Start session logging
        self.start_session_log()

    def test_db_connection(self):
        """Test the database connection and verify table exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if the table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='extracted_data'
            """)
            
            if not cursor.fetchone():
                raise Exception("Table 'extracted_data' not found in database")
            
            # Get table info
            cursor.execute("PRAGMA table_info(extracted_data)")
            columns = cursor.fetchall()
            print(f"Database connected successfully. Table has {len(columns)} columns.")
            
            conn.close()
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

    @classmethod
    def get_available_free_models(cls):
        """Get list of available free models."""
        return list(OPENROUTER_FREE_MODELS.keys())

    @classmethod
    def list_free_models(cls):
        """Print available free models with descriptions."""
        print("Available Free Models:")
        print("-" * 80)
        for model_id, description in OPENROUTER_FREE_MODELS.items():
            provider = model_id.split('/')[0]
            print(f"[{provider.upper()}] {model_id}")
            print(f"  Description: {description}")
            print()

    @classmethod
    def get_models_by_provider(cls, provider=None):
        """Get models filtered by provider."""
        if provider:
            return {k: v for k, v in OPENROUTER_FREE_MODELS.items() 
                   if k.startswith(provider.lower())}
        else:
            # Group by provider
            providers = {}
            for model_id, description in OPENROUTER_FREE_MODELS.items():
                prov = model_id.split('/')[0]
                if prov not in providers:
                    providers[prov] = []
                providers[prov].append((model_id, description))
            return providers

    @classmethod
    def get_recommended_models(cls):
        """Get recommended models for different use cases."""
        return {
            "general": "deepseek/deepseek-chat-v3-0324:free",
            "reasoning": "deepseek/deepseek-r1:free", 
            "coding": "deepseek/deepseek-chat-v3-0324:free",
            "fast": "google/gemini-2.0-flash-exp:free",
            "multimodal": "meta-llama/llama-4-scout:free",
            "lightweight": "qwen/qwen3-0.6b-04-28:free"
        }

    def start_session_log(self):
        """Initialize session logging."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== AEM Agent Session Log - {self.session_id} ===\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.llm_model_name}\n")
            f.write(f"API Mode: {self.api_mode}\n")
            f.write(f"Database: {self.db_path}\n")
            f.write("=" * 50 + "\n\n")

    def log_message(self, role, content):
        """Log a message to the session log file."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {role.upper()}: {content}\n")
                f.write("-" * 30 + "\n")
        except Exception as e:
            print(f"Logging error: {e}")

    def save_session_log(self, log_filepath=None):
        """Save the current session log to a specified file."""
        if log_filepath is None:
            log_filepath = f"session_log_{self.session_id}_{int(time.time())}.txt"
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as source:
                content = source.read()
            with open(log_filepath, 'w', encoding='utf-8') as target:
                target.write(content)
            print(f"Session log saved to: {log_filepath}")
        except Exception as e:
            print(f"Error saving session log: {e}")

    def fix_sql_query(self, sql_query, error_message, question):
        """Fix a SQL query that caused an error."""
        fix_prompt = f"""
        The following SQL query caused an error:
        Query: {sql_query}
        Error: {error_message}
        Original question: {question}
        
        Please fix the SQL query based on the error message and the database schema.
        Return only the corrected SQL query without any additional text.
        
        Database schema:
        {self.schema}
        """
        
        try:
            fixed_query = self.sql_llm.invoke(fix_prompt).content
            return fixed_query.strip()
        except Exception as e:
            print(f"Error fixing SQL query: {e}")
            return sql_query

    def execute_sql_query(self, sql_query, retry_count=3, question=None):
        """
        Execute a SQL query against the database with retry mechanism.

        Args:
            sql_query (str): The SQL query to execute.
            retry_count (int, optional): Number of retry attempts. Defaults to 3.
            question (str, optional): The user's original question.

        Returns:
            pd.DataFrame or None: The query result as a DataFrame or None if unsuccessful.
        """
        for i in range(retry_count):
            try:
                self.log_message("ai", f"SQL query: {sql_query}")
                self.CHAT_HISTORY.add_ai_message(f"SQL query: {sql_query}")
                
                # Execute the query
                conn = sqlite3.connect(self.db_path)
                result = pd.read_sql_query(sql_query, conn)
                conn.close()
                
                return result
                
            except Exception as e:
                print(f"Error: {e}, please fix the SQL query and try again.")
                self.log_message("ai", f"Error: {e}, please fix the SQL query and try again.")
                self.CHAT_HISTORY.add_ai_message(f"Error: {e}, please fix the SQL query and try again.")
                sql_query = self.fix_sql_query(sql_query, error_message=e, question=question)
                if i == retry_count - 1:
                    return None
        return None

    def summarize_dataframe(self, result: pd.DataFrame) -> str:
        """
        Summarize the result of a SQL query.
        """
        if isinstance(result, pd.DataFrame):
            # analyze the size, column names, data type of each column, randomly sample 10 rows
            summary = f"Size: {result.shape}\n"
            summary += f"Columns: {result.columns.tolist()}\n"
            # Escape curly braces in data types to prevent template variable interpretation
            data_types_str = str(result.dtypes.to_dict()).replace("{", "{{").replace("}", "}}")
            summary += f"Data types: {data_types_str}\n"
            # Handle case where dataframe has fewer than 10 rows
            sample_size = min(10, len(result))
            sample_markdown = result.sample(sample_size).to_markdown() if sample_size > 0 else 'No rows available'
            # Escape curly braces in sample data as well, just in case
            sample_markdown = sample_markdown.replace("{", "{{").replace("}", "}}")
            summary += f"Sample rows: {sample_markdown}\n"
            return summary
        else:
            return str(result)[:20000] # limit the length of the result to 20000 characters

    def create_final_result_prompt_template(self, use_sql, result):
        """Create the final result prompt template."""
        if use_sql == "yes":
            if result is not None:
                # Escape curly braces in the result to prevent template variable interpretation
                escaped_result = str(result).replace("{", "{{").replace("}", "}}")
                return f"SQL query result needed, use the following result to answer the question:\n{escaped_result}"
            else:
                return "SQL query was executed but no results were found."
        else:
            return "No SQL query result needed, answer the question directly."

    def task_execution(self, question):
        """
        Execute the task based on the user's question.

        Args:
            question (str): The user's question.

        Yields:
            str: Responses or intermediate steps.
        """
        try:
            result_summary = None
            result = None
            # Log the user's message
            self.log_message("user", question)
            self.CHAT_HISTORY.add_user_message(question)

            # Run the chain to decide if a SQL query is needed
            sql_response = self.sql_chain.invoke({
                "question": question,
                "chat_history": self.CHAT_HISTORY.messages,
                "fewshot_examples": self.fewshot_examples
            })
            use_sql = sql_response.get("use_sql", "no")

            # Handle SQL query execution
            if use_sql == "yes":
                if "sql_query" in sql_response:
                    thought_process = sql_response.get("thought_process", "")
                    # Escape curly braces in thought process to prevent template variable interpretation
                    escaped_thought_process = str(thought_process).replace("{", "{{").replace("}", "}}")
                    # Format thought process to avoid markdown parsing issues
                    msg = f"[Thought Process]\n{escaped_thought_process}\n\n"
                    yield msg
                    self.CHAT_HISTORY.add_ai_message(msg)
                    sql_query = sql_response["sql_query"]

                    msg = f"[Generated SQL Query]\n{sql_query}\n\n"
                    yield msg
                    self.CHAT_HISTORY.add_ai_message(msg)
                    self.log_message("ai", f"Generated SQL Query: {sql_query}")

                    # Execute the SQL query
                    result = self.execute_sql_query(sql_query, retry_count=3, question=question)
                    if result is None or result.empty:
                        msg = "Error: No results found. Please try another query."
                        self.log_message("ai", msg)
                        self.CHAT_HISTORY.add_ai_message(msg)
                        yield msg + "\n\n"
                    else:
                        result_summary = self.summarize_dataframe(result)
                        msg = f"[Results found]\n{result_summary}"
                        self.log_message("ai", msg)
                        self.CHAT_HISTORY.add_ai_message(msg)
                        yield result
                else:
                    msg = f"Error: No SQL query found in the response.\n{sql_response}"
                    self.log_message("ai", msg)
                    self.CHAT_HISTORY.add_ai_message(msg)
                    yield msg + "\n\n"
                    result = None
            else:
                thought_process = sql_response.get("thought_process", "")
                # Escape curly braces in thought process to prevent template variable interpretation
                escaped_thought_process = str(thought_process).replace("{", "{{").replace("}", "}}")
                yield f"[Thought Process]\n{escaped_thought_process}\n\n"
                self.CHAT_HISTORY.add_ai_message(f"[Thought Process]\n{escaped_thought_process}\n\n")
                self.log_message("ai", f"Thought Process: {escaped_thought_process}")
                result = None
                yield f"No SQL query result needed, just answer directly.\n\n"
                self.CHAT_HISTORY.add_ai_message("No SQL query result needed, just answer directly.")
                self.log_message("ai", "No SQL query result needed, just answer directly.")

            # Prepare final response
            final_result_prompt_template = self.create_final_result_prompt_template(use_sql, result_summary)
            answer_agent_prompt = ChatPromptTemplate.from_messages([
                ("system", self.answer_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", self.schema),
                ("human", final_result_prompt_template),
                ("human", "{question}"),
            ])

            current_chain = {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "result_summary": itemgetter("result_summary")
            }
            answer_agent = (
                current_chain
                | answer_agent_prompt
                | self.answer_llm
                | StrOutputParser()
            )
            chain_with_message_history = RunnableWithMessageHistory(
                answer_agent,
                lambda session_id: self.CHAT_HISTORY,
                input_messages_key='question',
                history_messages_key="chat_history",
            )

            respond_string = ""
            yield f"[Answer]\n"
            for chunk in chain_with_message_history.stream(
                {"question": question, "result_summary": result_summary},
                config={"configurable": {"session_id": self.session_id}}
            ):
                respond_string += chunk
                yield chunk

            # Log AI response
            self.log_message("ai", respond_string)
            self.CHAT_HISTORY.add_ai_message(respond_string)

        except Exception as e:
            error_msg = f"Error in task execution: {str(e)}"
            self.log_message("ai", error_msg)
            self.CHAT_HISTORY.add_ai_message(error_msg)
            yield error_msg

    def run(self):
        """
        Run the agent interactively.
        """
        print("AEM Agent is ready! Type 'quit' to exit.")
        print(f"Using model: {self.llm_model_name}")
        print(f"Connected to database: {self.db_path}")
        print("-" * 50)
        
        while True:
            question = input("\nYour question: ")
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            print("\nProcessing...")
            for response in self.task_execution(question):
                if isinstance(response, pd.DataFrame):
                    print("\n[Query Results]")
                    print(response.to_string(index=False))
                else:
                    print(response, end='')
            print("\n" + "="*50)

    def get_latest_processed_result(self):
        """Get the latest processed result."""
        # This method can be used to get the last query result
        # Implementation depends on how you want to store/retrieve results
        return None

