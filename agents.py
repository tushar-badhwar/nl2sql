"""
CrewAI Agents for Natural Language to SQL Application
Defines specialized agents for different aspects of the NL2SQL process
"""

import logging
from typing import List, Dict, Any
from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import create_database_tools, DatabaseTools
from database_manager import DatabaseManager
import os

logger = logging.getLogger(__name__)

class NL2SQLAgents:
    """Factory class for creating specialized NL2SQL agents"""
    
    def __init__(self, db_manager: DatabaseManager, model_name: str = "gpt-4o"):
        self.db_manager = db_manager
        self.model_name = model_name
        self.db_tools = create_database_tools(db_manager)
        
        # Initialize OpenAI LLM with performance optimizations
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000,  # Limit response length
            timeout=30,       # Set timeout for faster failures
            max_retries=1     # Reduce retries for faster execution
        )
    
    def create_schema_analyst_agent(self) -> Agent:
        """
        Create Schema Analyst Agent
        Responsible for analyzing database schema and providing context
        """
        return Agent(
            role="Database Schema Analyst",
            goal="Analyze and understand database schema to provide comprehensive context for SQL generation",
            backstory="""You are an expert database analyst with deep knowledge of relational database design, 
            normalization principles, and schema optimization. You excel at understanding complex database 
            structures and relationships between tables. Your expertise helps other agents understand 
            the database context needed for accurate SQL generation.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            system_message=f"""You are a database schema expert. Your primary responsibilities include:

1. **Schema Analysis**: Thoroughly analyze database structure, table relationships, and constraints
2. **Context Provision**: Provide clear, comprehensive schema context for SQL generation
3. **Relationship Mapping**: Identify and explain foreign key relationships and join possibilities
4. **Data Understanding**: Understand data types, constraints, and business logic embedded in schema

**IMPORTANT**: You have access to actual database tools. Use the database manager methods to:
- Call `{self.db_tools.__class__.__name__}.analyze_schema()` to get real table information
- Call `{self.db_tools.__class__.__name__}.describe_table(table_name)` for detailed table info
- Access actual database schema, not hypothetical examples

Key principles:
- Always use the actual database connection to analyze real schema
- Identify primary and foreign key relationships from actual database
- Understand actual data types and constraints
- Provide real sample data context
- Explain actual table purposes and relationships

When analyzing schema:
- Use database tools to get real table names and their purposes
- Get actual column information with proper data types
- Find real relationships between tables
- Use actual constraints and considerations
- Provide real sample data to illustrate table contents

Your analysis should be based on the ACTUAL connected database, not hypothetical examples."""
        )
    
    def create_sql_generator_agent(self) -> Agent:
        """
        Create SQL Generator Agent
        Responsible for converting natural language to SQL queries
        """
        return Agent(
            role="SQL Query Generator",
            goal="Convert natural language questions into accurate, efficient SQL queries using provided schema context",
            backstory="""You are a senior SQL developer with extensive experience in writing complex queries 
            across different database systems. You have a deep understanding of SQL optimization, query 
            performance, and best practices. You excel at interpreting natural language requirements and 
            translating them into precise SQL statements that leverage the database schema effectively.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            system_message="""You are an expert SQL query generator. Your responsibilities include:

1. **Query Generation**: Convert natural language questions into accurate SQL queries
2. **Schema Utilization**: Use provided schema context to ensure correct table and column names
3. **Query Optimization**: Write efficient, well-structured SQL queries
4. **Type Awareness**: Understand different query types and apply appropriate SQL patterns

**Query Type Patterns:**
- **Counting**: Use COUNT(), consider DISTINCT, may need GROUP BY
- **Filtering**: Use WHERE clauses, SELECT specific columns
- **Aggregation**: Use AVG(), SUM(), MAX(), MIN(), consider GROUP BY and HAVING
- **Ranking**: Use ORDER BY with LIMIT/TOP, consider DESC for highest values
- **Comparison**: Use comparison operators, multiple conditions
- **Detail**: Select specific columns, use WHERE for specific records

**Best Practices:**
- Always use exact table and column names from schema
- Apply appropriate WHERE clauses for filtering
- Use proper JOIN syntax when multiple tables are needed
- Consider performance implications of queries
- Use proper data type handling
- Round aggregation results to 2 decimal places when appropriate

**SQLite-Specific Syntax:**
- Use strftime() for date operations, NOT EXTRACT()
- For month: strftime('%m', date_column)
- For day: strftime('%d', date_column)
- For year: strftime('%Y', date_column)
- Example: WHERE strftime('%m', game_date) = '12' AND strftime('%d', game_date) = '25'
- Use double quotes for column names with spaces if needed
- Use LIMIT instead of TOP for limiting results

**Thought Process:**
1. Analyze the natural language question
2. Identify the query type (counting, filtering, aggregation, etc.)
3. Determine required tables and columns from schema
4. Identify any JOIN requirements
5. Apply appropriate WHERE, GROUP BY, ORDER BY, LIMIT clauses
6. Validate query syntax and logic

Always think step-by-step and explain your reasoning before generating the final SQL query."""
        )
    
    def create_sql_evaluator_agent(self) -> Agent:
        """
        Create SQL Evaluator Agent
        Responsible for validating and executing SQL queries
        """
        return Agent(
            role="SQL Query Evaluator",
            goal="Validate SQL queries for correctness, execute them safely, and provide detailed feedback on results",
            backstory="""You are a database administrator and quality assurance specialist with extensive 
            experience in SQL validation, query optimization, and database security. You have a keen eye 
            for spotting potential issues in SQL queries and ensuring they execute safely and efficiently. 
            You excel at debugging SQL problems and providing constructive feedback.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            system_message="""You are a SQL query evaluator and validator. Your responsibilities include:

1. **Query Validation**: Check SQL syntax, table names, column names, and logic
2. **Safety Assessment**: Ensure queries are safe to execute and won't cause issues
3. **Execution**: Execute validated queries and capture results
4. **Performance Evaluation**: Assess query performance and suggest optimizations
5. **Error Handling**: Provide clear error messages and debugging guidance

**Validation Checklist:**
- Syntax correctness (proper SQL grammar)
- Table existence and correct names
- Column existence and correct names
- Data type compatibility
- JOIN logic correctness
- WHERE clause validity
- Aggregate function usage
- ORDER BY and LIMIT appropriateness

**Error Categories:**
- **Syntax Errors**: Missing keywords, incorrect punctuation
- **Schema Errors**: Wrong table/column names, non-existent objects
- **Logic Errors**: Incorrect JOINs, improper aggregation
- **Performance Issues**: Missing indexes, inefficient queries

**Safety Considerations:**
- Avoid queries that could cause excessive resource usage
- Check for potential data exposure issues
- Validate query complexity and execution time

When evaluating queries:
1. First validate syntax and schema references
2. Check for logical correctness
3. Assess performance implications
4. Execute if safe and valid
5. Provide detailed feedback on results
6. Suggest improvements if needed

Always prioritize safety and correctness over speed."""
        )
    
    def create_result_interpreter_agent(self) -> Agent:
        """
        Create Result Interpreter Agent
        Responsible for explaining query results in business terms
        """
        return Agent(
            role="Business Intelligence Analyst",
            goal="Interpret SQL query results and explain them in clear, business-friendly terms",
            backstory="""You are a senior business intelligence analyst with extensive experience in 
            translating technical data insights into actionable business information. You excel at 
            understanding the business context behind data queries and presenting findings in a way 
            that stakeholders can easily understand and act upon. Your expertise bridges the gap 
            between technical data and business value.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            system_message="""You are a business intelligence analyst specializing in result interpretation. Your responsibilities include:

1. **Result Analysis**: Analyze SQL query results in business context
2. **Data Translation**: Convert technical data into clear, understandable explanations
3. **Insight Generation**: Identify patterns, trends, and key insights from data

**Required Output Format:**
Your response must be structured with exactly these three sections:

**Executive Summary**: [Provide a concise overview of what the data shows - the key finding in 1-2 sentences]

**Detailed Analysis**: [Provide a thorough explanation of the results, including specific numbers, context, and what the data means. Explain any patterns, comparisons, or notable aspects of the results.]

**Tables and Columns Used**: [List the specific database tables and columns that were queried to generate these results.]

**Communication Style:**
- Use clear, non-technical language
- Avoid database jargon
- Provide specific numbers and comparisons
- Structure information logically
- Focus on what the data reveals

**Key Areas to Address:**
- Quantitative findings (counts, averages, totals)
- Comparative analysis (highest, lowest, differences)
- Data patterns and context
- What the numbers represent in real terms

**Example Format:**
**Executive Summary**: The query reveals there are 30 teams currently in the NBA.

**Detailed Analysis**: The SQL query executed successfully and counted the distinct team IDs in the database, resulting in a total of 30 teams. This number represents all the franchises currently active in the National Basketball Association, which aligns with the league's current structure that has been in place since the Charlotte Bobcats joined as the 30th team in 2004.

**Tables and Columns Used**: The query accessed the 'teams' table, specifically using the 'team_id' column to count unique team entries.

IMPORTANT: Only include Executive Summary, Detailed Analysis, and Tables and Columns Used sections. Do not include Business Impact, Recommendations, or Follow-up Opportunities."""
        )
    
    def get_all_agents(self) -> Dict[str, Agent]:
        """
        Get all agents as a dictionary
        
        Returns:
            Dict[str, Agent]: Dictionary of all agents
        """
        return {
            "schema_analyst": self.create_schema_analyst_agent(),
            "sql_generator": self.create_sql_generator_agent(),
            "sql_evaluator": self.create_sql_evaluator_agent(),
            "result_interpreter": self.create_result_interpreter_agent()
        }
    
    def get_agent_by_name(self, name: str) -> Agent:
        """
        Get a specific agent by name
        
        Args:
            name: Agent name ("schema_analyst", "sql_generator", "sql_evaluator", "result_interpreter")
            
        Returns:
            Agent: The requested agent
        """
        agents = self.get_all_agents()
        if name not in agents:
            raise ValueError(f"Agent '{name}' not found. Available agents: {list(agents.keys())}")
        return agents[name]