"""
Streamlit Web Interface for Natural Language to SQL Application
Main entry point for the web-based NL2SQL application
"""

# SQLite compatibility fix for Streamlit Cloud
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import logging
import os
import json
import pandas as pd
import time
from typing import Dict, Any, Optional
import traceback
from dotenv import load_dotenv

from database_manager import DatabaseManager
from crew_setup import NL2SQLCrew
from agents import NL2SQLAgents
from tasks import NL2SQLTasks

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NL2SQL CrewAI Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .sql-query {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_sql_from_text(text: str) -> Optional[str]:
    """Extract SQL query from text using improved regex patterns"""
    import re
    
    # Pattern to match SQL queries - comprehensive patterns
    sql_patterns = [
        # Code block patterns
        r'```sql\s*(.*?)\s*```',
        r'```SQL\s*(.*?)\s*```',
        r'```\s*(SELECT.*?)\s*```',
        r'```\s*(select.*?)\s*```',
        
        # Query with labels
        r'(?:SQL[:\s]+|Query[:\s]+|Generated Query[:\s]+)?\s*```\s*(SELECT.*?)\s*```',
        r'(?:SQL[:\s]+|Query[:\s]+|Generated Query[:\s]+)?\s*```\s*(select.*?)\s*```',
        
        # Standalone SQL patterns
        r'(?:^|\n)(?:SQL[:\s]+|Query[:\s]+|Generated Query[:\s]+)\s*(SELECT.*?)(?=\n\n|\n[A-Z]|\n#|\nExplanation|\nThe|\nThis|\Z)',
        r'(?:^|\n)(?:SQL[:\s]+|Query[:\s]+|Generated Query[:\s]+)\s*(select.*?)(?=\n\n|\n[A-Z]|\n#|\nExplanation|\nThe|\nThis|\Z)',
        
        # Generic SELECT patterns
        r'(?:^|\n)\s*(SELECT\s+(?:DISTINCT\s+)?.*?FROM\s+\w+.*?)(?=\n\n|\n[A-Z]|\n#|\nExplanation|\nThe|\nThis|\Z)',
        r'(?:^|\n)\s*(select\s+(?:distinct\s+)?.*?from\s+\w+.*?)(?=\n\n|\n[A-Z]|\n#|\nExplanation|\nThe|\nThis|\Z)',
        
        # SQL with semicolon
        r'(SELECT.*?;)',
        r'(select.*?;)',
        
        # Last resort - any SELECT statement
        r'\b(SELECT\s+.*?FROM\s+\w+.*?)(?:\s|$)',
        r'\b(select\s+.*?from\s+\w+.*?)(?:\s|$)',
    ]
    
    for pattern in sql_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            sql_query = match.strip()
            # Clean up the query
            sql_query = re.sub(r'\s+', ' ', sql_query)  # Normalize whitespace
            sql_query = sql_query.replace('```', '').strip()
            
            # Basic validation - must contain SELECT and have reasonable length
            if (sql_query.upper().startswith('SELECT') and 
                len(sql_query) > 10 and 
                'FROM' in sql_query.upper()):
                return sql_query
    
    return None

def extract_interpretation_from_text(text: str, sql_query: Optional[str] = None) -> str:
    """Extract interpretation part from text, removing SQL code blocks"""
    import re
    
    if not text:
        return ""
    
    # If we have SQL query, try to remove it from the text
    if sql_query:
        # Remove SQL code blocks
        text = re.sub(r'```sql.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'```SQL.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove the actual SQL query if it appears in text
        text = text.replace(sql_query, '')
    
    # Clean up the text
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
    text = text.strip()
    
    return text

def format_sql_query(sql_query: str) -> str:
    """Format SQL query with proper line breaks and indentation"""
    import re
    
    if not sql_query:
        return sql_query
    
    # Remove extra whitespace and normalize
    sql_query = re.sub(r'\s+', ' ', sql_query.strip())
    
    # Start formatting
    formatted_sql = sql_query
    
    # Handle SELECT clause
    formatted_sql = re.sub(r'\bSELECT\b', '\nSELECT', formatted_sql, flags=re.IGNORECASE)
    
    # Handle FROM clause
    formatted_sql = re.sub(r'\bFROM\b', '\nFROM', formatted_sql, flags=re.IGNORECASE)
    
    # Handle WHERE clause
    formatted_sql = re.sub(r'\bWHERE\b', '\nWHERE', formatted_sql, flags=re.IGNORECASE)
    
    # Handle JOIN clauses
    formatted_sql = re.sub(r'\b(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|JOIN)\b', r'\n\1', formatted_sql, flags=re.IGNORECASE)
    
    # Handle ON clause (for JOINs)
    formatted_sql = re.sub(r'\bON\b', '\n    ON', formatted_sql, flags=re.IGNORECASE)
    
    # Handle GROUP BY clause
    formatted_sql = re.sub(r'\bGROUP\s+BY\b', '\nGROUP BY', formatted_sql, flags=re.IGNORECASE)
    
    # Handle HAVING clause
    formatted_sql = re.sub(r'\bHAVING\b', '\nHAVING', formatted_sql, flags=re.IGNORECASE)
    
    # Handle ORDER BY clause
    formatted_sql = re.sub(r'\bORDER\s+BY\b', '\nORDER BY', formatted_sql, flags=re.IGNORECASE)
    
    # Handle LIMIT clause
    formatted_sql = re.sub(r'\bLIMIT\b', '\nLIMIT', formatted_sql, flags=re.IGNORECASE)
    
    # Handle UNION clauses
    formatted_sql = re.sub(r'\bUNION(\s+ALL)?\b', r'\nUNION\1', formatted_sql, flags=re.IGNORECASE)
    
    # Handle AND/OR in WHERE clauses (add indentation)
    formatted_sql = re.sub(r'\b(AND|OR)\b(?=.*)', r'\n    \1', formatted_sql, flags=re.IGNORECASE)
    
    # Clean up multiple newlines and leading/trailing whitespace
    formatted_sql = re.sub(r'\n\s*\n', '\n', formatted_sql)
    formatted_sql = formatted_sql.strip()
    
    # Remove leading newline if present
    if formatted_sql.startswith('\n'):
        formatted_sql = formatted_sql[1:]
    
    return formatted_sql

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'crew' not in st.session_state:
        st.session_state.crew = None
    
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = False
    
    if 'schema_cached' not in st.session_state:
        st.session_state.schema_cached = False
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'current_schema' not in st.session_state:
        st.session_state.current_schema = None
    
    if 'current_ai_result' not in st.session_state:
        st.session_state.current_ai_result = None
    
    if 'feedback_examples' not in st.session_state:
        st.session_state.feedback_examples = []
    
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = {}

def sidebar_database_config():
    """Render database configuration sidebar"""
    st.sidebar.header("🗄️ Database Configuration")
    
    # Database type selection
    db_type = st.sidebar.selectbox(
        "Database Type",
        ["SQLite", "PostgreSQL", "MySQL"],
        help="Select the type of database to connect to"
    )
    
    # Database connection parameters
    connection_params = {}
    
    if db_type == "SQLite":
        uploaded_file = st.sidebar.file_uploader(
            "Upload SQLite Database",
            type=['sqlite', 'db', 'sqlite3'],
            help="Upload your SQLite database file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            connection_params = {
                'db_type': 'sqlite',
                'file_path': temp_path
            }
            
            st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        
        # Alternative: direct file path input
        file_path = st.sidebar.text_input(
            "Or enter SQLite file path",
            help="Enter the full path to your SQLite database file"
        )
        
        if file_path and not uploaded_file:
            connection_params = {
                'db_type': 'sqlite',
                'file_path': file_path
            }
    
    elif db_type in ["PostgreSQL", "MySQL"]:
        host = st.sidebar.text_input("Host", value="localhost")
        port = st.sidebar.number_input(
            "Port", 
            value=5432 if db_type == "PostgreSQL" else 3306,
            min_value=1,
            max_value=65535
        )
        database = st.sidebar.text_input("Database Name")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if all([host, database, username, password]):
            connection_params = {
                'db_type': db_type.lower(),
                'host': host,
                'port': port,
                'database': database,
                'username': username,
                'password': password
            }
    
    # Connection button
    if st.sidebar.button("Connect to Database"):
        if connection_params:
            connect_to_database(connection_params)
        else:
            st.sidebar.error("Please fill in all required connection parameters")
    
    # Connection status
    if st.session_state.connection_status:
        st.sidebar.success("✅ Database Connected")
        
        # Show table count
        try:
            tables = st.session_state.db_manager.get_table_names()
            st.sidebar.info(f"📊 {len(tables)} tables found")
        except Exception as e:
            st.sidebar.warning(f"Could not retrieve table count: {str(e)}")
    else:
        st.sidebar.warning("❌ Database Not Connected")
    
    return connection_params

def connect_to_database(params: Dict[str, Any]):
    """Connect to database with given parameters"""
    try:
        with st.spinner("Connecting to database..."):
            success = st.session_state.db_manager.connect(**params)
            
            if success:
                st.session_state.connection_status = True
                st.session_state.crew = NL2SQLCrew(
                    st.session_state.db_manager,
                    model_name='gpt-4o'
                )
                st.sidebar.success("Database connected successfully!")
                
                # Clear previous schema cache
                st.session_state.schema_cached = False
                st.session_state.current_schema = None
                
                # Trigger schema analysis
                analyze_schema()
                
            else:
                st.sidebar.error("Failed to connect to database")
                st.session_state.connection_status = False
                
    except Exception as e:
        st.sidebar.error(f"Database connection error: {str(e)}")
        st.session_state.connection_status = False

def analyze_schema():
    """Analyze database schema"""
    if not st.session_state.connection_status or not st.session_state.crew:
        return
    
    try:
        with st.spinner("Analyzing database schema..."):
            db_type = st.session_state.db_manager.database_type or "Unknown"
            db_path = "connected_database"
            
            schema_analysis = st.session_state.crew.analyze_schema(db_type, db_path)
            st.session_state.current_schema = schema_analysis
            st.session_state.schema_cached = True
            
            st.success("Schema analysis completed!")
            
    except Exception as e:
        st.error(f"Schema analysis failed: {str(e)}")

def check_api_configuration():
    """Check API configuration without showing UI elements"""
    # Check for OpenAI API Key in environment (silently)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None, None
    
    # Always use gpt-4o as the default model
    selected_model = "gpt-4o"
    st.session_state.selected_model = selected_model
    
    return api_key, selected_model

def render_schema_viewer():
    """Render database schema viewer"""
    st.subheader("📊 Database Schema")
    
    if not st.session_state.connection_status:
        st.warning("Please connect to a database first")
        return
    
    if not st.session_state.schema_cached:
        if st.button("Analyze Schema"):
            analyze_schema()
        return
    
    # Display schema information
    try:
        tables = st.session_state.db_manager.get_table_names()
        
        # Table selection
        selected_table = st.selectbox(
            "Select Table to View",
            ["All Tables"] + tables,
            help="Choose a table to view detailed information"
        )
        
        if selected_table == "All Tables":
            # Show all tables summary
            st.write("### Database Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Tables", len(tables))
            
            with col2:
                try:
                    total_rows = 0
                    for table in tables:
                        stats = st.session_state.db_manager.get_table_stats(table)
                        total_rows += stats.get('row_count', 0)
                    st.metric("Total Rows", f"{total_rows:,}")
                except:
                    st.metric("Total Rows", "N/A")
            
            with col3:
                st.metric("Database Type", st.session_state.db_manager.database_type.upper())
            
            # Tables list
            st.write("### Tables")
            for table in tables:
                with st.expander(f"📋 {table}"):
                    try:
                        stats = st.session_state.db_manager.get_table_stats(table)
                        schema = st.session_state.db_manager.get_table_schema(table)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Rows:** {stats.get('row_count', 'N/A'):,}")
                            st.write(f"**Columns:** {len(schema.get('columns', []))}")
                        
                        with col2:
                            if schema.get('primary_keys'):
                                st.write(f"**Primary Keys:** {', '.join(schema['primary_keys'])}")
                            if schema.get('foreign_keys'):
                                st.write(f"**Foreign Keys:** {len(schema['foreign_keys'])}")
                        
                        # Show columns
                        if schema.get('columns'):
                            st.write("**Columns:**")
                            for col in schema['columns'][:5]:  # Show first 5 columns
                                st.write(f"- {col['name']}: {col['type']}")
                            if len(schema['columns']) > 5:
                                st.write(f"... and {len(schema['columns']) - 5} more columns")
                    
                    except Exception as e:
                        st.error(f"Error loading table info: {str(e)}")
        
        else:
            # Show detailed table information
            st.write(f"### Table: {selected_table}")
            
            try:
                schema = st.session_state.db_manager.get_table_schema(selected_table)
                stats = st.session_state.db_manager.get_table_stats(selected_table)
                sample_data = st.session_state.db_manager.get_sample_data(selected_table, limit=5)
                
                # Table statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Rows", f"{stats.get('row_count', 'N/A'):,}")
                
                with col2:
                    st.metric("Columns", len(schema.get('columns', [])))
                
                with col3:
                    st.metric("Primary Keys", len(schema.get('primary_keys', [])))
                
                # Column information
                st.write("#### Columns")
                if schema.get('columns'):
                    columns_df = pd.DataFrame([
                        {
                            'Column': col['name'],
                            'Type': col['type'],
                            'Nullable': 'Yes' if col['nullable'] else 'No',
                            'Default': col.get('default', 'None')
                        }
                        for col in schema['columns']
                    ])
                    st.dataframe(columns_df, use_container_width=True)
                
                # Constraints
                if schema.get('primary_keys') or schema.get('foreign_keys'):
                    st.write("#### Constraints")
                    
                    if schema.get('primary_keys'):
                        st.write(f"**Primary Keys:** {', '.join(schema['primary_keys'])}")
                    
                    if schema.get('foreign_keys'):
                        st.write("**Foreign Keys:**")
                        for fk in schema['foreign_keys']:
                            st.write(f"- {fk['column']} → {fk['referenced_table']}.{fk['referenced_column']}")
                
                # Sample data
                st.write("#### Sample Data")
                if sample_data['success'] and sample_data['data']:
                    df = pd.DataFrame(sample_data['data'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No sample data available")
            
            except Exception as e:
                st.error(f"Error loading table details: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading schema: {str(e)}")

def render_query_interface():
    """Render natural language query interface"""
    st.subheader("💬 Natural Language Query")
    
    if not st.session_state.connection_status:
        st.warning("Please connect to a database first")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Service is not available. Please contact the administrator.")
        return
    
    # Query input
    question = st.text_area(
        "Enter your question",
        placeholder="e.g., How many teams are in the NBA?",
        height=100,
        help="Enter your question in natural language"
    )
    
    # Submit button
    if st.button("🚀 Process Query", type="primary"):
        if not question.strip():
            st.error("Please enter a question")
            return
        
        process_natural_language_query(question, use_full_workflow=True, show_metrics=False)
    
    # Display AI results if they exist
    if 'current_ai_result' in st.session_state and st.session_state.current_ai_result:
        ai_result = st.session_state.current_ai_result
        display_query_results(
            ai_result['result'], 
            ai_result['question'], 
            ai_result['processing_time'], 
            ai_result['show_metrics']
        )
    

def process_natural_language_query(question: str, use_full_workflow: bool, show_metrics: bool):
    """Process natural language query through CrewAI"""
    
    if not st.session_state.crew:
        st.error("Crew not initialized. Please check database connection.")
        return
    
    try:
        # Show processing status
        with st.spinner("Processing your query..."):
            start_time = time.time()
            
            # Get database info for workflow
            db_type = st.session_state.db_manager.database_type or "Unknown"
            db_path = "connected_database"
            
            # Process query
            result = st.session_state.crew.process_query(
                natural_language_question=question,
                use_full_workflow=use_full_workflow,
                db_type=db_type,
                db_path=db_path
            )
            
            processing_time = time.time() - start_time
        
        # Store current results in session state for persistent display
        st.session_state.current_ai_result = {
            'question': question,
            'result': result,
            'processing_time': processing_time,
            'show_metrics': show_metrics,
            'timestamp': time.time()
        }
        
        # Add to history
        st.session_state.query_history.append({
            'question': question,
            'timestamp': time.time(),
            'result': result,
            'processing_time': processing_time
        })
        
    except Exception as e:
        st.error(f"Query processing failed: {str(e)}")
        st.expander("Error Details", expanded=False).code(traceback.format_exc())

def display_query_results(result: Dict[str, Any], question: str, processing_time: float, show_metrics: bool):
    """Display query processing results"""
    
    st.subheader("📊 Query Results")
    
    # Create unique key for this query result (moved to top)
    import hashlib
    result_hash = hashlib.md5(f"{question}_{processing_time}".encode()).hexdigest()[:8]
    result_id = f"query_{result_hash}"
    
    # Success/failure status
    if result.get('success', False):
        if result.get('workflow_type') == 'feedback_retry':
            st.success(f"✅ Query revised and processed successfully in {processing_time:.2f} seconds")
            if result.get('original_sql'):
                st.info(f"🔄 **Note**: Original query returned no results. AI generated a revised query.")
        else:
            st.success(f"✅ Query processed successfully in {processing_time:.2f} seconds")
    else:
        st.error(f"❌ Query processing failed: {result.get('error', 'Unknown error')}")
        return
    
    # Try to extract SQL query from result (multiple approaches)
    sql_query = result.get('sql_query')
    raw_output = result.get('raw_output', '')
    
    # If no SQL query extracted, try to find it in raw output
    if not sql_query and raw_output:
        sql_query = extract_sql_from_text(raw_output)
    
    # Display SQL Query Section
    if sql_query:
        # Format the SQL query for better readability
        formatted_sql = format_sql_query(sql_query)
        
        if result.get('workflow_type') == 'feedback_retry' and result.get('original_sql'):
            st.write("#### 🔍 Generated SQL Query (Revised)")
            st.code(formatted_sql, language='sql')
            
            # Show original query in expander
            with st.expander("🔍 View Original Query (No Results)"):
                formatted_original = format_sql_query(result.get('original_sql'))
                st.code(formatted_original, language='sql')
                st.caption("This query executed successfully but returned no data, so it was revised.")
        else:
            st.write("#### 🔍 Generated SQL Query")
            st.code(formatted_sql, language='sql')
        
        # Automatically execute the query and show results
        st.write("#### 📊 Query Results")
        execution_result = execute_sql_query(sql_query, return_results=True)
        # Store execution results in the main result for history
        if execution_result:
            result['execution_result'] = execution_result
        
        # Custom SQL section - moved here right after AI results
        st.markdown("---")
        st.subheader("🔍 Check ground truth by running your own query")
        
        # Initialize custom query history in session state
        if 'custom_query_history' not in st.session_state:
            st.session_state.custom_query_history = []
        
        # Custom SQL input
        custom_sql = st.text_area(
            "Enter your SQL query",
            placeholder="SELECT column1, column2\nFROM table_name\nWHERE condition\nLIMIT 10;",
            height=120,
            help="Enter any valid SQL query to execute on the connected database",
            key=f"custom_sql_{result_id}"
        )
        
        # Execute custom SQL button
        if st.button("🚀 Execute SQL Query", key=f"execute_custom_{result_id}"):
            if not custom_sql.strip():
                st.error("Please enter a SQL query")
            else:
                execute_custom_sql_query(custom_sql)
        
        # Feedback section
        st.markdown("---")
        st.write("#### 👍 Was this query helpful?")
        st.caption("Your feedback helps improve the AI's query generation. Positive examples become few-shot learning examples for better future queries.")
        
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("👍 Good", key=f"thumbs_up_{result_id}"):
                save_positive_feedback(question, sql_query, result)
                st.success("Thanks! This example will help improve future queries.")
                st.rerun()
        
        with col2:
            if st.button("👎 Needs Improvement", key=f"thumbs_down_{result_id}"):
                save_negative_feedback(question, sql_query, result)
                st.warning("Thanks for the feedback! We'll work on improving similar queries.")
                st.rerun()
        
        with col3:
            # Show current feedback status
            if result_id in st.session_state.user_feedback:
                feedback_type = st.session_state.user_feedback[result_id]
                if feedback_type == "positive":
                    st.success("✅ You marked this query as helpful")
                else:
                    st.warning("❌ You marked this query as needing improvement")
        
        # Copy query button
        if st.button("📋 Copy Query"):
            st.write("Query copied to clipboard!")
    else:
        st.warning("⚠️ Could not extract SQL query from the response")
        
        # Show raw output in expandable section for debugging
        with st.expander("🔍 View Raw Response"):
            st.text(raw_output)
    
    # Show AI interpretation
    if raw_output:
        st.write("#### 🤖 AI Analysis & Interpretation")
        
        # Try to extract just the interpretation part (remove SQL)
        interpretation = extract_interpretation_from_text(raw_output, sql_query)
        
        if interpretation:
            st.write(interpretation)
        else:
            st.write(raw_output)
    
    # Show metrics if requested
    if show_metrics:
        show_performance_metrics(result, processing_time)
    
    # Debug section - show processing details
    with st.expander("🔧 Debug: View Processing Details"):
        if result.get('sql_query'):
            st.write("**Extracted SQL Query:**")
            formatted_debug_sql = format_sql_query(result['sql_query'])
            st.code(formatted_debug_sql, language='sql')
        
        st.write("**Processing Details:**")
        st.json({
            "success": result.get('success', False),
            "processing_time": processing_time,
            "workflow_type": result.get('workflow_type', 'unknown'),
            "model_used": result.get('model_used', 'unknown')
        })

def execute_sql_query(sql_query: str, return_results: bool = False):
    """Execute SQL query and display results"""
    try:
        with st.spinner("Executing SQL query..."):
            query_result = st.session_state.db_manager.execute_query(sql_query)
        
        if query_result['success']:
            row_count = query_result.get('row_count', 0)
            st.success(f"✅ Query executed successfully - {row_count} rows returned")
            
            if query_result['data'] and len(query_result['data']) > 0:
                # Display results as dataframe
                df = pd.DataFrame(query_result['data'])
                st.dataframe(df, use_container_width=True)
                
                # Show summary information
                st.info(f"📋 Showing {len(df)} rows × {len(df.columns)} columns")
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
                
                if return_results:
                    return query_result
            else:
                st.info("✨ Query executed successfully but returned no data")
                if return_results:
                    return query_result
        else:
            st.error(f"❌ Query execution failed: {query_result.get('error', 'Unknown error')}")
            if return_results:
                return query_result
            
    except Exception as e:
        st.error(f"❌ Error executing query: {str(e)}")
        # Show more detailed error information
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        if return_results:
            return {"success": False, "error": str(e), "data": None, "row_count": 0}

def show_performance_metrics(result: Dict[str, Any], processing_time: float):
    """Display performance metrics"""
    st.write("#### Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col2:
        st.metric("Workflow Type", result.get('workflow_type', 'Unknown'))
    
    with col3:
        # Count positive feedback from user_feedback session state
        positive_count = 0
        if 'user_feedback' in st.session_state:
            positive_count = sum(1 for feedback in st.session_state.user_feedback.values() if feedback == "positive")
        st.metric("Queries Liked", positive_count)
    
    with col4:
        # Count negative feedback from user_feedback session state
        negative_count = 0
        if 'user_feedback' in st.session_state:
            negative_count = sum(1 for feedback in st.session_state.user_feedback.values() if feedback == "negative")
        st.metric("Queries Disliked", negative_count)

def render_query_history():
    """Render query history"""
    st.subheader("📜 Query History")
    
    if not st.session_state.query_history:
        st.info("No queries processed yet")
        return
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.rerun()
    
    # Display history
    for i, entry in enumerate(reversed(st.session_state.query_history)):
        with st.expander(f"Query {len(st.session_state.query_history) - i}: {entry['question'][:50]}..."):
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")
            st.write(f"**Processing Time:** {entry['processing_time']:.2f}s")
            
            # Show SQL Query
            if entry['result'].get('sql_query'):
                st.write("**SQL Query:**")
                formatted_history_sql = format_sql_query(entry['result']['sql_query'])
                st.code(formatted_history_sql, language='sql')
            
            # Show Query Results
            execution_result = entry['result'].get('execution_result')
            if execution_result and execution_result.get('success'):
                st.write("**Query Results:**")
                if execution_result.get('data') and len(execution_result['data']) > 0:
                    df = pd.DataFrame(execution_result['data'])
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"📋 {len(df)} rows × {len(df.columns)} columns")
                else:
                    st.info("Query executed but returned no data")
            elif execution_result:
                st.error(f"Query execution failed: {execution_result.get('error', 'Unknown error')}")
            
            # Show Overall Status
            if entry['result'].get('success'):
                st.success("✅ Query processed successfully")
            else:
                st.error(f"❌ Query processing failed: {entry['result'].get('error', 'Unknown error')}")

def execute_custom_sql_query(sql_query: str):
    """Execute custom SQL query and display results"""
    
    try:
        # Clean and prepare query
        cleaned_query = sql_query.strip()
        
        # Add LIMIT 10 if not already present
        if cleaned_query.upper().find('LIMIT') == -1:
            if cleaned_query.endswith(';'):
                cleaned_query = cleaned_query[:-1] + ' LIMIT 10;'
            else:
                cleaned_query += ' LIMIT 10'
        
        # Execute query
        start_time = time.time()
        
        with st.spinner("Executing SQL query..."):
            query_result = st.session_state.db_manager.execute_query(cleaned_query)
        
        execution_time = time.time() - start_time
        
        # Store in custom query history
        st.session_state.custom_query_history.append({
            'query': cleaned_query,
            'timestamp': time.time(),
            'result': query_result,
            'execution_time': execution_time
        })
        
        # Display results in a separate section
        st.write("#### 📊 Ground Truth Query Results")
        if query_result['success']:
            row_count = query_result.get('row_count', 0)
            st.success(f"✅ Query executed successfully in {execution_time:.2f}s - {row_count} rows returned")
            
            if query_result.get('data') and len(query_result['data']) > 0:
                # Display results as dataframe
                df = pd.DataFrame(query_result['data'])
                st.dataframe(df, use_container_width=True)
                st.info(f"📋 Showing {len(df)} rows × {len(df.columns)} columns")
                        
            else:
                st.info("✨ Query executed successfully but returned no data")
                
        else:
            st.error(f"❌ Query execution failed: {query_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")

def save_positive_feedback(question: str, sql_query: str, result: Dict[str, Any]):
    """Save positive feedback as a few-shot example"""
    try:
        # Create example entry
        example = {
            'question': question,
            'sql_query': sql_query,
            'timestamp': time.time(),
            'execution_successful': result.get('execution_result', {}).get('success', False),
            'row_count': result.get('execution_result', {}).get('row_count', 0),
            'workflow_type': result.get('workflow_type', 'normal')
        }
        
        # Add to feedback examples (limit to 10 most recent)
        st.session_state.feedback_examples.append(example)
        if len(st.session_state.feedback_examples) > 10:
            st.session_state.feedback_examples = st.session_state.feedback_examples[-10:]
        
        # Mark as positive feedback
        result_id = f"{question}_{sql_query}_{result.get('processing_time', 0)}"
        st.session_state.user_feedback[result_id] = "positive"
        
        # Save to persistent storage
        save_feedback_to_file()
        
    except Exception as e:
        st.error(f"Error saving positive feedback: {str(e)}")

def save_negative_feedback(question: str, sql_query: str, result: Dict[str, Any]):
    """Save negative feedback for future improvement"""
    try:
        # Mark as negative feedback
        result_id = f"{question}_{sql_query}_{result.get('processing_time', 0)}"
        st.session_state.user_feedback[result_id] = "negative"
        
        # Could store negative examples separately for analysis
        negative_example = {
            'question': question,
            'sql_query': sql_query,
            'timestamp': time.time(),
            'feedback_type': 'negative',
            'workflow_type': result.get('workflow_type', 'normal')
        }
        
        # For now, just log it (could be enhanced to store for analysis)
        logger.info(f"Negative feedback received for question: {question}")
        
    except Exception as e:
        st.error(f"Error saving negative feedback: {str(e)}")

def save_feedback_to_file():
    """Save feedback examples to a persistent file"""
    try:
        import json
        import os
        
        feedback_file = "feedback_examples.json"
        
        # Prepare data for saving
        feedback_data = {
            'examples': st.session_state.feedback_examples,
            'last_updated': time.time()
        }
        
        # Save to file
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving feedback to file: {str(e)}")

def load_feedback_from_file():
    """Load feedback examples from persistent file"""
    try:
        import json
        import os
        
        feedback_file = "feedback_examples.json"
        
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
                st.session_state.feedback_examples = feedback_data.get('examples', [])
                
    except Exception as e:
        logger.error(f"Error loading feedback from file: {str(e)}")

def get_few_shot_examples() -> str:
    """Generate few-shot examples from positive feedback"""
    try:
        if not st.session_state.feedback_examples:
            return ""
        
        # Get up to 5 most recent positive examples
        recent_examples = st.session_state.feedback_examples[-5:]
        
        few_shot_text = "\n## Few-Shot Examples (from successful queries):\n\n"
        
        for i, example in enumerate(recent_examples, 1):
            few_shot_text += f"**Example {i}:**\n"
            few_shot_text += f"Question: \"{example['question']}\"\n"
            few_shot_text += f"SQL Query:\n```sql\n{example['sql_query']}\n```\n"
            if example.get('row_count', 0) > 0:
                few_shot_text += f"Result: Successfully returned {example['row_count']} rows\n\n"
            else:
                few_shot_text += f"Result: Query executed successfully\n\n"
        
        few_shot_text += "Use these examples as reference for similar types of questions.\n\n"
        
        return few_shot_text
        
    except Exception as e:
        logger.error(f"Error generating few-shot examples: {str(e)}")
        return ""

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Load feedback examples from file
    load_feedback_from_file()
    
    # Header
    st.title("🤖 Natural Language to SQL")
    st.markdown("Convert natural language to SQL using LLMs")
    
    # Check API configuration (no UI)
    api_key, model = check_api_configuration()
    
    # Sidebar
    with st.sidebar:
        # Database configuration
        connection_params = sidebar_database_config()
        
        st.markdown("---")
        
        # Application info
        st.header("ℹ️ Application Info")
        st.info("""
        This application uses CrewAI with specialized agents:
        - **Schema Analyst**: Analyzes database structure
        - **SQL Generator**: Converts NL to SQL
        - **SQL Evaluator**: Validates and executes queries, sends feedback to SQL generator in case of errors.
        - **Result Interpreter**: Explains reasoning behind generated queries
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "📊 Schema", "📜 History", "⚡ Performance"])
    
    with tab1:
        render_query_interface()
    
    with tab2:
        render_schema_viewer()
    
    with tab3:
        render_query_history()
    
    with tab4:
        st.subheader("⚡ Performance Dashboard")
        
        if st.session_state.crew:
            metrics = st.session_state.crew.get_performance_metrics()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", metrics.get('total_queries', 0))
            
            with col2:
                st.metric("Avg. Processing Time", f"{metrics.get('average_execution_time', 0):.2f}s")
            
            with col3:
                # Count positive feedback from user_feedback session state
                positive_count = 0
                if 'user_feedback' in st.session_state:
                    positive_count = sum(1 for feedback in st.session_state.user_feedback.values() if feedback == "positive")
                st.metric("Queries Liked", positive_count)
            
            with col4:
                # Count negative feedback from user_feedback session state
                negative_count = 0
                if 'user_feedback' in st.session_state:
                    negative_count = sum(1 for feedback in st.session_state.user_feedback.values() if feedback == "negative")
                st.metric("Queries Disliked", negative_count)
            
            # Reset metrics button
            if st.button("🔄 Reset Metrics"):
                st.session_state.crew.reset_metrics()
                st.success("Metrics reset successfully")
                st.rerun()
        
        else:
            st.info("Connect to a database to view performance metrics")

if __name__ == "__main__":
    main()