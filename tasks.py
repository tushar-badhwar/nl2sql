"""
CrewAI Tasks for Natural Language to SQL Application
Defines the workflow tasks that agents execute in sequence
"""

import logging
from typing import Dict, Any, List
from crewai import Task
from agents import NL2SQLAgents
from database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class NL2SQLTasks:
    """Factory class for creating NL2SQL workflow tasks"""
    
    def __init__(self, agents: NL2SQLAgents):
        self.agents = agents.get_all_agents()
    
    def create_schema_analysis_task(self, db_type: str = "Unknown", db_path: str = "Unknown") -> Task:
        """
        Create schema analysis task
        
        Args:
            db_type: Database type
            db_path: Database path or name
            
        Returns:
            Task: Schema analysis task
        """
        return Task(
            description=f"""
            Analyze the database schema to understand the structure and relationships between tables.
            
            Your task is to:
            1. Connect to the database if not already connected
            2. Analyze the complete database schema
            3. Identify all tables, their columns, data types, and constraints
            4. Map relationships between tables (foreign keys, primary keys)
            5. Understand the business domain and purpose of each table
            6. Provide sample data from key tables to understand data patterns
            7. Create a comprehensive schema summary that will help with SQL generation
            
            Focus on:
            - Table names and their purposes
            - Column names and data types
            - Primary and foreign key relationships
            - Data patterns and constraints
            - Business context of the data
            
            Database connection details:
            - Database type: {db_type}
            - Database path/name: {db_path}
            
            Provide a detailed analysis that will serve as the foundation for accurate SQL query generation.
            """,
            expected_output="""
            A comprehensive database schema analysis including:
            1. List of all tables with their purposes
            2. Detailed column information for each table (name, type, constraints)
            3. Primary and foreign key relationships
            4. Sample data from key tables
            5. Business context and domain understanding
            6. Table relationship diagram (textual description)
            7. Key insights about the data structure
            
            Format the output as a structured report that can be easily referenced by other agents.
            """,
            agent=self.agents["schema_analyst"]
        )
    
    def create_query_generation_task(self, natural_language_question: str, schema_context: str = "") -> Task:
        """
        Create SQL query generation task
        
        Args:
            natural_language_question: The natural language question to convert to SQL
            schema_context: Database schema context from schema analysis
            
        Returns:
            Task: SQL query generation task
        """
        return Task(
            description=f"""
            Convert the following natural language question into an accurate SQL query using the provided schema context.
            
            Natural Language Question: "{natural_language_question}"
            
            Schema Context: {schema_context}
            
            Your task is to:
            1. Analyze the natural language question to understand the intent
            2. Identify the query type (counting, filtering, aggregation, ranking, etc.)
            3. Determine which tables and columns are needed based on the schema
            4. Identify any JOIN requirements between tables
            5. Apply appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses
            6. Generate an accurate, efficient SQL query
            7. Validate the query syntax and logic
            
            Key considerations:
            - Use exact table and column names from the schema
            - Apply proper SQL syntax and best practices (SQLite syntax)
            - Consider performance implications
            - Handle data types appropriately
            - Round aggregation results to 2 decimal places when appropriate
            - Use proper JOIN syntax when multiple tables are needed
            
            SQLite-specific syntax rules:
            - Use strftime() for date operations, NOT EXTRACT()
            - For month: strftime('%m', date_column)
            - For day: strftime('%d', date_column)
            - For year: strftime('%Y', date_column)
            - Example: WHERE strftime('%m', game_date) = '12' AND strftime('%d', game_date) = '25'
            
            Think step-by-step:
            1. What is the user asking for?
            2. What type of query is needed?
            3. Which tables contain the required data?
            4. What columns are needed?
            5. Are any JOINs required?
            6. What filters or conditions are needed?
            7. How should the results be ordered or limited?
            
            Generate a clean, efficient SQL query that accurately answers the question.
            """,
            expected_output="""
            A complete SQL query that:
            1. Correctly answers the natural language question
            2. Uses proper table and column names from the schema
            3. Applies appropriate SQL syntax and structure
            4. Includes necessary JOINs, WHERE clauses, and other components
            5. Is optimized for performance
            6. Handles data types correctly
            
            Format: Provide the SQL query in a clean, readable format with proper indentation.
            Also include a brief explanation of what the query does and why specific components were chosen.
            """,
            agent=self.agents["sql_generator"]
        )
    
    def create_query_evaluation_task(self, sql_query: str) -> Task:
        """
        Create SQL query evaluation task
        
        Args:
            sql_query: The SQL query to evaluate and execute
            
        Returns:
            Task: SQL query evaluation task
        """
        return Task(
            description=f"""
            Evaluate and execute the following SQL query to ensure it's correct and safe.
            
            SQL Query to evaluate:
            {sql_query}
            
            Your task is to:
            1. Validate the SQL syntax and structure
            2. Check that all table and column names exist in the database
            3. Verify that the query logic is correct
            4. Assess potential performance implications
            5. Execute the query if it's safe and valid
            6. Capture and analyze the results
            7. Identify any errors or issues
            8. Provide feedback on query quality and optimization opportunities
            
            Validation checklist:
            - Syntax correctness (proper SQL grammar)
            - Table existence and correct names
            - Column existence and correct names  
            - Data type compatibility
            - JOIN logic correctness
            - WHERE clause validity
            - Aggregate function usage
            - ORDER BY and LIMIT appropriateness
            
            Safety considerations:
            - Avoid queries that could cause excessive resource usage
            - Check for potential data exposure issues
            - Validate query complexity and execution time
            
            If the query has issues:
            - Provide clear error messages
            - Suggest specific fixes
            - Explain why the issues occurred
            
            If the query is successful:
            - Execute it and capture results
            - Provide result summary
            - Assess query performance
            - Suggest optimizations if applicable
            """,
            expected_output="""
            A comprehensive query evaluation report including:
            1. Validation status (valid/invalid)
            2. Any syntax or logic errors found
            3. Query execution results (if successful)
            4. Performance assessment
            5. Result summary (number of rows, columns, data types)
            6. Any optimization suggestions
            7. Error messages and fixes (if applicable)
            
            Format:
            - Status: SUCCESS/FAILURE
            - Errors: List of any issues found
            - Results: Query execution results
            - Performance: Execution time and resource usage
            - Recommendations: Optimization suggestions
            """,
            agent=self.agents["sql_evaluator"]
        )
    
    def create_result_interpretation_task(self, query_results: Dict[str, Any], original_question: str) -> Task:
        """
        Create result interpretation task
        
        Args:
            query_results: Results from SQL query execution
            original_question: The original natural language question
            
        Returns:
            Task: Result interpretation task
        """
        return Task(
            description=f"""
            Interpret the SQL query results and explain them in clear, business-friendly terms.
            
            Original Question: "{original_question}"
            
            Query Results: {query_results}
            
            Your task is to:
            1. Analyze the query results in the context of the original question
            2. Translate technical data into business-friendly explanations
            3. Identify key patterns, trends, and insights
            4. Provide context about what the results mean
            5. Identify the specific database tables and columns that were used
            
            Interpretation framework:
            - What: Clearly state what the data shows
            - Context: Provide relevant context for understanding the results
            - Tables/Columns: Identify the database sources used to generate the results
            
            Communication principles:
            - Use clear, non-technical language
            - Avoid database jargon
            - Provide specific numbers and comparisons
            - Structure information logically
            
            Key areas to address:
            - Quantitative findings (counts, averages, totals)
            - Comparative analysis (highest, lowest, differences)
            - Trends and patterns
            - Database sources (tables and columns used)
            
            Make the technical data accessible and understandable for users.
            """,
            expected_output="""
            A business-friendly interpretation of the query results including:
            1. Clear summary of findings in plain language
            2. Comprehensive explanation of the results and context
            3. Identification of specific database tables and columns used
            
            Format:
            - Executive Summary: Key findings in 1-2 sentences
            - Detailed Analysis: Comprehensive explanation of results, patterns, and context
            - Tables and Columns Used: Specific database tables and columns that were queried
            """,
            agent=self.agents["result_interpreter"]
        )
    
    def create_complete_workflow_tasks(self, 
                                     natural_language_question: str, 
                                     db_type: str = "Unknown",
                                     db_path: str = "Unknown") -> List[Task]:
        """
        Create complete workflow tasks for NL2SQL processing
        
        Args:
            natural_language_question: The natural language question to process
            db_type: Database type
            db_path: Database path or name
            
        Returns:
            List[Task]: Complete list of tasks for the workflow
        """
        # Create schema analysis task
        schema_task = self.create_schema_analysis_task(db_type, db_path)
        
        # Create query generation task
        query_task = self.create_query_generation_task(natural_language_question)
        
        # Create query evaluation task
        evaluation_task = self.create_query_evaluation_task("")
        
        # Removed result interpretation task for better performance
        return [schema_task, query_task, evaluation_task]
    
    def create_quick_query_tasks(self, 
                                natural_language_question: str, 
                                schema_context: str = "") -> List[Task]:
        """
        Create a simplified workflow for quick queries when schema is already known
        
        Args:
            natural_language_question: The natural language question to process
            schema_context: Pre-analyzed schema context
            
        Returns:
            List[Task]: Simplified list of tasks for quick processing
        """
        # Create query generation task
        query_task = self.create_query_generation_task(natural_language_question, schema_context)
        
        # Create query evaluation task
        evaluation_task = self.create_query_evaluation_task("")
        
        # Removed result interpretation task for better performance
        
        return [query_task, evaluation_task]
    
    def create_sql_generation_task_with_feedback(self, question: str, feedback_prompt: str, schema_context: str) -> Task:
        """
        Create a SQL generation task with feedback from previous failed attempt
        
        Args:
            question: Original natural language question
            feedback_prompt: Feedback about why previous query failed
            schema_context: Database schema context
            
        Returns:
            Task: SQL generation task with feedback
        """
        return Task(
            description=f"""
            QUERY REVISION TASK: Generate a revised SQL query based on feedback.
            
            {feedback_prompt}
            
            Schema Context:
            {schema_context}
            
            Original Question: "{question}"
            
            Your task is to:
            1. Analyze the feedback about why the previous query failed
            2. Review the database schema carefully
            3. Identify potential issues with the original query
            4. Generate a revised SQL query that addresses the issues
            5. Ensure the new query uses correct table and column names
            6. Consider alternative approaches (JOINs, WHERE clauses, etc.)
            7. Validate the logic matches the natural language question
            
            IMPORTANT: Your response must include the revised SQL query clearly formatted like this:

            ```sql
            [YOUR REVISED SQL QUERY HERE]
            ```
            
            Also explain what changes you made and why.
            """,
            expected_output="""
            A revised SQL query with explanation of changes made to address the feedback.
            The query should be properly formatted in SQL code blocks and should address
            the issues identified in the feedback.
            """,
            agent=self.agents["sql_generator"]
        )