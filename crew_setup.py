"""
CrewAI Crew Setup and Orchestration for Natural Language to SQL Application
Main orchestrator that coordinates all agents and tasks
"""

import logging
import os
from typing import Dict, Any, List, Optional
from crewai import Crew, Process
from agents import NL2SQLAgents
from tasks import NL2SQLTasks
from database_manager import DatabaseManager
import time
import json

logger = logging.getLogger(__name__)

class NL2SQLCrew:
    """
    Main crew orchestrator for Natural Language to SQL conversion
    """
    
    def __init__(self, db_manager: DatabaseManager, model_name: str = "gpt-4o"):
        self.db_manager = db_manager
        self.model_name = model_name
        
        # Initialize agents and tasks
        self.agents_factory = NL2SQLAgents(db_manager, model_name)
        self.tasks_factory = NL2SQLTasks(self.agents_factory)
        
        # Cache for schema analysis to avoid repeated analysis
        self._schema_cache: Optional[str] = None
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_execution_time": 0.0,
            "execution_times": []
        }
    
    def analyze_schema(self, db_type: str = "Unknown", db_path: str = "Unknown") -> str:
        """
        Analyze database schema and cache results
        
        Args:
            db_type: Database type
            db_path: Database path or name
            
        Returns:
            str: Schema analysis results
        """
        if self._schema_cache is not None:
            logger.info("Using cached schema analysis")
            return self._schema_cache
            
        logger.info("Performing new schema analysis")
        
        try:
            # Get actual schema information from database manager
            actual_schema = self._get_actual_database_schema()
            
            # Create schema analysis task with real schema data
            schema_task = self.tasks_factory.create_schema_analysis_task(db_type, db_path)
            
            # Override the task description with actual schema
            schema_task.description = f"""
            Analyze the following ACTUAL database schema and provide a comprehensive summary:
            
            {actual_schema}
            
            Your task is to:
            1. Summarize the tables and their purposes based on the actual schema above
            2. Identify relationships between tables using the foreign key information
            3. Provide business context for what this database represents
            4. Create a structured summary that will help with SQL generation
            
            Focus on the ACTUAL table and column names provided above, not hypothetical examples.
            """
            
            # Create crew with just schema analyst
            schema_crew = Crew(
                agents=[self.agents_factory.get_agent_by_name("schema_analyst")],
                tasks=[schema_task],
                process=Process.sequential,
                verbose=False  # Reduce verbosity for faster execution
            )
            
            # Execute schema analysis
            result = schema_crew.kickoff()
            
            # Cache the result
            self._schema_cache = str(result)
            
            logger.info("Schema analysis completed successfully")
            return self._schema_cache
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {str(e)}")
            raise
    
    def _get_actual_database_schema(self) -> str:
        """
        Get actual database schema from the database manager
        
        Returns:
            str: Formatted schema information
        """
        try:
            tables = self.db_manager.get_table_names()
            schema_info = "=== DATABASE SCHEMA ===\n\n"
            
            for table in tables:
                schema = self.db_manager.get_table_schema(table)
                stats = self.db_manager.get_table_stats(table)
                
                schema_info += f"Table: {table}\n"
                schema_info += f"Rows: {stats.get('row_count', 'Unknown')}\n"
                schema_info += f"Columns:\n"
                
                for col in schema.get('columns', []):
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    schema_info += f"  - {col['name']}: {col['type']} ({nullable})\n"
                
                if schema.get('primary_keys'):
                    schema_info += f"Primary Keys: {', '.join(schema['primary_keys'])}\n"
                
                if schema.get('foreign_keys'):
                    schema_info += "Foreign Keys:\n"
                    for fk in schema['foreign_keys']:
                        schema_info += f"  - {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}\n"
                
                # Get sample data (reduced for performance)
                sample_data = self.db_manager.get_sample_data(table, limit=1)
                if sample_data['success'] and sample_data['data']:
                    schema_info += f"Sample data (first 1 row):\n"
                    for i, row in enumerate(sample_data['data'][:1]):
                        schema_info += f"  Row {i+1}: {dict(row)}\n"
                
                schema_info += "\n"
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting actual schema: {str(e)}")
            return "ERROR: Could not retrieve actual database schema"
    
    def process_query(self, 
                     natural_language_question: str, 
                     use_full_workflow: bool = True,
                     db_type: str = "Unknown",
                     db_path: str = "Unknown") -> Dict[str, Any]:
        """
        Process a natural language query through the complete workflow
        
        Args:
            natural_language_question: The natural language question to process
            use_full_workflow: Whether to use full workflow or quick processing
            db_type: Database type
            db_path: Database path or name
            
        Returns:
            Dict[str, Any]: Complete processing results
        """
        start_time = time.time()
        
        try:
            self.metrics["total_queries"] += 1
            
            if use_full_workflow:
                return self._process_full_workflow(natural_language_question, db_type, db_path)
            else:
                return self._process_quick_workflow(natural_language_question)
                
        except Exception as e:
            self.metrics["failed_queries"] += 1
            logger.error(f"Query processing failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "question": natural_language_question,
                "processing_time": time.time() - start_time
            }
        
        finally:
            execution_time = time.time() - start_time
            self.metrics["execution_times"].append(execution_time)
            self.metrics["average_execution_time"] = sum(self.metrics["execution_times"]) / len(self.metrics["execution_times"])
    
    def _process_full_workflow(self, 
                              natural_language_question: str, 
                              db_type: str,
                              db_path: str) -> Dict[str, Any]:
        """
        Process query using the full workflow with all agents
        
        Args:
            natural_language_question: The natural language question
            db_type: Database type
            db_path: Database path or name
            
        Returns:
            Dict[str, Any]: Complete processing results
        """
        logger.info(f"Processing full workflow for question: {natural_language_question}")
        
        try:
            # Step 1: Schema Analysis (cached if available)
            schema_context = self.analyze_schema(db_type, db_path)
            
            # Step 2: Create workflow tasks - but we'll override descriptions with fresh context
            tasks = self.tasks_factory.create_complete_workflow_tasks(
                natural_language_question, 
                db_type,
                db_path
            )
            
            # Step 2.5: Force fresh schema context for ALL tasks, even when using cached analysis
            actual_schema = self._get_actual_database_schema()
            
            # Update schema analysis task (first task)
            if len(tasks) >= 1:
                schema_task = tasks[0]
                schema_task.description = f"""
                Analyze the following ACTUAL database schema and provide a comprehensive summary:
                
                {actual_schema}
                
                Your task is to:
                1. Summarize the tables and their purposes based on the actual schema above
                2. Identify relationships between tables using the foreign key information
                3. Provide business context for what this database represents
                4. Create a structured summary that will help with SQL generation
                
                Focus on the ACTUAL table and column names provided above, not hypothetical examples.
                """
            
            # Update SQL generation task (second task) with fresh schema context and few-shot examples
            if len(tasks) >= 2:
                sql_task = tasks[1]
                
                # Get few-shot examples from user feedback
                few_shot_examples = self._get_few_shot_examples()
                
                sql_task.description = f"""
                Convert the following natural language question into an accurate SQL query using the provided schema context.
                
                Natural Language Question: "{natural_language_question}"
                
                Current Database Schema:
                {actual_schema}
                
                Schema Analysis Context:
                {schema_context}
                
                {few_shot_examples}
                
                Your task is to:
                1. Analyze the natural language question to understand the intent
                2. Identify the query type (counting, filtering, aggregation, ranking, etc.)
                3. Determine which tables and columns are needed based on the schema above
                4. Identify any JOIN requirements between tables
                5. Apply appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses
                6. Generate an accurate, efficient SQL query
                7. Validate the query syntax and logic
                8. Use the few-shot examples above as reference for similar question patterns
                
                Key considerations:
                - Use exact table and column names from the current database schema above
                - Apply proper SQL syntax and best practices (SQLite syntax)
                - Consider performance implications
                - Handle data types appropriately
                - Round aggregation results to 2 decimal places when appropriate
                - Use proper JOIN syntax when multiple tables are needed
                - Learn from the successful examples provided above
                
                SQLite-specific syntax rules:
                - Use strftime() for date operations, NOT EXTRACT()
                - For month: strftime('%m', date_column)
                - For day: strftime('%d', date_column)
                - For year: strftime('%Y', date_column)
                - Example: WHERE strftime('%m', game_date) = '12' AND strftime('%d', game_date) = '25'
                
                IMPORTANT: Your response must include the SQL query clearly formatted like this:

                ```sql
                [YOUR SQL QUERY HERE]
                ```
                
                Generate a clean, efficient SQL query that accurately answers the question.
                """
            
            # Update SQL evaluation task (third task) with context
            if len(tasks) >= 3:
                eval_task = tasks[2]
                eval_task.description = f"""
                Validate and execute the SQL query generated for the question: "{natural_language_question}"
                
                Current Database Schema:
                {actual_schema}
                
                Your task is to:
                1. Validate the SQL query syntax and logic
                2. Check that all table and column names exist in the current schema
                3. Execute the query safely if valid
                4. Provide detailed feedback on results or errors
                5. Suggest improvements if needed
                
                Ensure the query is safe to execute and uses correct table/column names from the schema above.
                """
            
            # Removed result interpretation task for better performance - now only 3 tasks total
            
            # Step 3: Create crew with all agents
            crew = Crew(
                agents=list(self.agents_factory.get_all_agents().values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=False,  # Reduced verbosity for better performance
                max_execution_time=300  # 5 minute timeout
            )
            
            # Step 4: Execute workflow with potential feedback loop
            result = crew.kickoff()
            
            # Step 5: Parse and structure results, including individual task outputs
            structured_result = self._parse_crew_results(result, natural_language_question)
            
            # Try to extract SQL from individual task outputs
            logger.info(f"Crew attributes: {dir(crew)}")
            if hasattr(crew, 'tasks_outputs') and crew.tasks_outputs:
                logger.info(f"Found {len(crew.tasks_outputs)} task outputs")
                for i, task_output in enumerate(crew.tasks_outputs):
                    logger.info(f"Task {i} output preview: {str(task_output)[:200]}...")
                    if i == 1:  # SQL generation task (second task)
                        sql_from_task = self._extract_sql_from_result(str(task_output))
                        if sql_from_task and not structured_result.get("sql_query"):
                            structured_result["sql_query"] = sql_from_task
                            logger.info(f"Extracted SQL from task output: {sql_from_task}")
                            break
            else:
                logger.warning("No tasks_outputs attribute found or empty")
                # Try to access tasks and their results differently
                if hasattr(crew, 'tasks') and crew.tasks:
                    logger.info(f"Found {len(crew.tasks)} tasks")
                    for i, task in enumerate(crew.tasks):
                        if hasattr(task, 'output') and task.output:
                            logger.info(f"Task {i} has output: {str(task.output)[:200]}...")
                            if i == 1:  # SQL generation task
                                sql_from_task = self._extract_sql_from_result(str(task.output))
                                if sql_from_task:
                                    structured_result["sql_query"] = sql_from_task
                                    logger.info(f"Extracted SQL from task.output: {sql_from_task}")
                                    break
            
            # Step 6: Check if query needs revision based on results
            if self._should_retry_query(structured_result, natural_language_question):
                logger.info("Query returned no results - attempting to revise SQL")
                revised_result = self._retry_with_feedback(natural_language_question, structured_result, schema_context, actual_schema)
                if revised_result and revised_result.get('success'):
                    structured_result = revised_result
            
            self.metrics["successful_queries"] += 1
            logger.info("Full workflow completed successfully")
            
            return structured_result
            
        except Exception as e:
            logger.error(f"Full workflow failed: {str(e)}")
            raise
    
    def _process_quick_workflow(self, natural_language_question: str) -> Dict[str, Any]:
        """
        Process query using quick workflow (assumes schema is already analyzed)
        
        Args:
            natural_language_question: The natural language question
            
        Returns:
            Dict[str, Any]: Processing results
        """
        logger.info(f"Processing quick workflow for question: {natural_language_question}")
        
        try:
            # Use cached schema if available
            schema_context = self._schema_cache or ""
            
            # Create quick workflow tasks
            tasks = self.tasks_factory.create_quick_query_tasks(
                natural_language_question, 
                schema_context
            )
            
            # Create crew with relevant agents
            crew = Crew(
                agents=[
                    self.agents_factory.get_agent_by_name("sql_generator"),
                    self.agents_factory.get_agent_by_name("sql_evaluator"),
                    self.agents_factory.get_agent_by_name("result_interpreter")
                ],
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute workflow
            result = crew.kickoff()
            
            # Parse and structure results
            structured_result = self._parse_crew_results(result, natural_language_question)
            
            self.metrics["successful_queries"] += 1
            logger.info("Quick workflow completed successfully")
            
            return structured_result
            
        except Exception as e:
            logger.error(f"Quick workflow failed: {str(e)}")
            raise
    
    def _parse_crew_results(self, crew_result: Any, original_question: str) -> Dict[str, Any]:
        """
        Parse and structure crew execution results
        
        Args:
            crew_result: Raw crew execution result
            original_question: Original natural language question
            
        Returns:
            Dict[str, Any]: Structured results
        """
        try:
            # Convert crew result to string if needed
            result_str = str(crew_result)
            
            # Try to extract structured information
            # This is a simplified parsing - in production, you might want more sophisticated parsing
            
            structured_result = {
                "success": True,
                "question": original_question,
                "raw_output": result_str,
                "processing_time": self.metrics["execution_times"][-1] if self.metrics["execution_times"] else 0,
                "model_used": self.model_name,
                "workflow_type": "full" if self._schema_cache else "quick"
            }
            
            # Try to extract SQL query from result
            sql_query = self._extract_sql_from_result(result_str)
            if sql_query:
                structured_result["sql_query"] = sql_query
            
            # Try to extract key insights
            insights = self._extract_insights_from_result(result_str)
            if insights:
                structured_result["insights"] = insights
            
            return structured_result
            
        except Exception as e:
            logger.error(f"Error parsing crew results: {str(e)}")
            return {
                "success": False,
                "error": f"Result parsing failed: {str(e)}",
                "question": original_question,
                "raw_output": str(crew_result)
            }
    
    def _extract_sql_from_result(self, result_str: str) -> Optional[str]:
        """
        Extract SQL query from crew result string
        
        Args:
            result_str: Raw result string
            
        Returns:
            Optional[str]: Extracted SQL query
        """
        try:
            # Look for SQL patterns
            import re
            
            logger.info(f"Attempting SQL extraction from text of length {len(result_str)}")
            logger.info(f"First 500 chars: {result_str[:500]}")
            
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
                matches = re.findall(pattern, result_str, re.DOTALL | re.IGNORECASE)
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
            
        except Exception as e:
            logger.error(f"Error extracting SQL: {str(e)}")
            return None
    
    def _extract_insights_from_result(self, result_str: str) -> Optional[str]:
        """
        Extract key insights from crew result string
        
        Args:
            result_str: Raw result string
            
        Returns:
            Optional[str]: Extracted insights
        """
        try:
            import re
            
            # Look for common insight patterns
            insight_patterns = [
                r'(?:insights?|findings?|results?|analysis)[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)',
                r'(?:summary|conclusion)[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)',
                r'(?:business impact|implications)[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)',
            ]
            
            for pattern in insight_patterns:
                match = re.search(pattern, result_str, re.DOTALL | re.IGNORECASE)
                if match:
                    insight = match.group(1).strip()
                    if insight and len(insight) > 20:  # Basic validation
                        return insight
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return None
    
    def _should_retry_query(self, result: Dict[str, Any], question: str) -> bool:
        """
        Determine if a query should be retried based on the results
        
        Args:
            result: The query execution result
            question: The original natural language question
            
        Returns:
            bool: True if query should be retried
        """
        try:
            # Check if we have a SQL query
            sql_query = result.get("sql_query")
            if not sql_query:
                return False
            
            # Try to execute the SQL query to check results
            if hasattr(self, 'db_manager') and self.db_manager:
                try:
                    execution_result = self.db_manager.execute_query(sql_query)
                    
                    # If query executed successfully but returned no data
                    if (execution_result.get('success') and 
                        (not execution_result.get('data') or len(execution_result.get('data', [])) == 0)):
                        
                        # Check if this is a counting/aggregation query that should return something
                        sql_upper = sql_query.upper()
                        has_count = 'COUNT(' in sql_upper
                        has_sum = 'SUM(' in sql_upper
                        has_avg = 'AVG(' in sql_upper
                        has_max_min = 'MAX(' in sql_upper or 'MIN(' in sql_upper
                        
                        # Questions that typically expect results
                        question_lower = question.lower()
                        expects_data = any([
                            'how many' in question_lower,
                            'count' in question_lower,
                            'total' in question_lower,
                            'average' in question_lower,
                            'sum' in question_lower,
                            'list' in question_lower,
                            'show' in question_lower,
                            'what are' in question_lower,
                            'which' in question_lower
                        ])
                        
                        # If it's an aggregation query or expects data, consider retrying
                        if (has_count or has_sum or has_avg or has_max_min) or expects_data:
                            logger.info(f"Query returned no results but question expects data: {question}")
                            return True
                            
                except Exception as e:
                    logger.error(f"Error checking query results: {str(e)}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error in _should_retry_query: {str(e)}")
            return False
    
    def _get_few_shot_examples(self) -> str:
        """
        Get few-shot examples from user feedback for inclusion in SQL generation prompts
        
        Returns:
            str: Formatted few-shot examples or empty string if none available
        """
        try:
            # Try to import streamlit to access session state
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and 'feedback_examples' in st.session_state:
                    feedback_examples = st.session_state.feedback_examples
                    if feedback_examples:
                        examples_text = "\n\nFEW-SHOT EXAMPLES FROM USER FEEDBACK:\n"
                        examples_text += "Use these successful examples as reference for similar question patterns:\n\n"
                        
                        for i, example in enumerate(feedback_examples, 1):
                            examples_text += f"Example {i}:\n"
                            examples_text += f"Question: \"{example.get('question', '')}\"\n"
                            examples_text += f"SQL Query: {example.get('sql_query', '')}\n"
                            row_count = example.get('row_count', 0)
                            if row_count > 0:
                                examples_text += f"Result: Successfully returned {row_count} rows\n\n"
                            else:
                                examples_text += f"Result: Query executed successfully\n\n"
                        
                        return examples_text
            except ImportError:
                # Streamlit not available, try to load from file
                pass
            
            # Fallback: try to load feedback from file if it exists
            feedback_file = "feedback_examples.json"
            if os.path.exists(feedback_file):
                try:
                    with open(feedback_file, 'r') as f:
                        feedback_data = json.load(f)
                        positive_examples = feedback_data.get('examples', [])
                        
                        if positive_examples:
                            examples_text = "\n\nFEW-SHOT EXAMPLES FROM USER FEEDBACK:\n"
                            examples_text += "Use these successful examples as reference for similar question patterns:\n\n"
                            
                            for i, example in enumerate(positive_examples, 1):
                                examples_text += f"Example {i}:\n"
                                examples_text += f"Question: \"{example.get('question', '')}\"\n"
                                examples_text += f"SQL Query: {example.get('sql_query', '')}\n"
                                row_count = example.get('row_count', 0)
                                if row_count > 0:
                                    examples_text += f"Result: Successfully returned {row_count} rows\n\n"
                                else:
                                    examples_text += f"Result: Query executed successfully\n\n"
                            
                            return examples_text
                except Exception as e:
                    logger.warning(f"Could not load feedback from file: {str(e)}")
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting few-shot examples: {str(e)}")
            return ""
    
    def _retry_with_feedback(self, question: str, original_result: Dict[str, Any], 
                           schema_context: str, actual_schema: str) -> Dict[str, Any]:
        """
        Retry query generation with feedback about the failed result
        
        Args:
            question: Original natural language question
            original_result: The original result that needs improvement
            schema_context: Schema analysis context
            actual_schema: Actual database schema
            
        Returns:
            Dict[str, Any]: Revised query result
        """
        try:
            original_sql = original_result.get("sql_query", "")
            
            # Create feedback message
            feedback_prompt = f"""
            FEEDBACK REQUIRED: The previous SQL query returned no results, but the question suggests there should be data.

            Original Question: "{question}"
            Previous SQL Query: {original_sql}
            Issue: Query executed successfully but returned 0 rows

            Current Database Schema:
            {actual_schema}

            Please analyze why the query might be returning no results and generate a revised SQL query.
            
            Common issues to check:
            1. Are table names and column names correct?
            2. Are JOIN conditions appropriate?
            3. Are WHERE clauses too restrictive?
            4. Are there case sensitivity issues?
            5. Should LIKE operators be used instead of exact matches?
            6. Are date/time formats correct?
            7. Should the query be structured differently?

            Generate a revised SQL query that addresses these potential issues.
            """

            # Create a new SQL generation task with feedback
            revised_sql_task = self.tasks_factory.create_sql_generation_task_with_feedback(
                question, feedback_prompt, schema_context
            )
            
            # Create a simple crew with just SQL generator for retry
            retry_crew = Crew(
                agents=[self.agents_factory.get_agent_by_name("sql_generator")],
                tasks=[revised_sql_task],
                process=Process.sequential,
                verbose=True,
                max_execution_time=120
            )
            
            # Execute retry
            retry_result = retry_crew.kickoff()
            
            # Extract SQL from retry result
            revised_sql = self._extract_sql_from_result(str(retry_result))
            
            if revised_sql and revised_sql != original_sql:
                logger.info(f"Generated revised SQL: {revised_sql}")
                
                # Test the revised SQL
                test_result = self.db_manager.execute_query(revised_sql)
                
                if test_result.get('success'):
                    # Create new result structure
                    revised_result = {
                        "success": True,
                        "question": question,
                        "sql_query": revised_sql,
                        "raw_output": f"Revised query generated after feedback:\n\n{str(retry_result)}",
                        "processing_time": original_result.get("processing_time", 0),
                        "model_used": self.model_name,
                        "workflow_type": "feedback_retry",
                        "original_sql": original_sql,
                        "revision_reason": "Previous query returned no results"
                    }
                    
                    logger.info("Successfully generated revised query with results")
                    return revised_result
                else:
                    logger.warning(f"Revised SQL also failed: {test_result.get('error')}")
            else:
                logger.warning("Could not extract revised SQL from retry result")
                
        except Exception as e:
            logger.error(f"Error in _retry_with_feedback: {str(e)}")
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the crew
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        return {
            "total_queries": self.metrics["total_queries"],
            "successful_queries": self.metrics["successful_queries"],
            "failed_queries": self.metrics["failed_queries"],
            "success_rate": (self.metrics["successful_queries"] / max(1, self.metrics["total_queries"])) * 100,
            "average_execution_time": self.metrics["average_execution_time"],
            "model_used": self.model_name,
            "schema_cached": self._schema_cache is not None
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_execution_time": 0.0,
            "execution_times": []
        }
    
    def clear_schema_cache(self):
        """Clear cached schema analysis"""
        self._schema_cache = None
        logger.info("Schema cache cleared")
    
    def get_schema_summary(self) -> Optional[str]:
        """
        Get current schema analysis summary
        
        Returns:
            Optional[str]: Schema summary if available
        """
        return self._schema_cache
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate that the crew setup is correct
        
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            "database_connected": False,
            "agents_created": False,
            "tasks_factory_ready": False,
            "openai_api_configured": False,
            "overall_status": "unknown"
        }
        
        try:
            # Check database connection
            if self.db_manager.engine is not None:
                validation_results["database_connected"] = True
            
            # Check agents
            agents = self.agents_factory.get_all_agents()
            if len(agents) == 4:  # Should have 4 agents
                validation_results["agents_created"] = True
            
            # Check tasks factory
            if self.tasks_factory is not None:
                validation_results["tasks_factory_ready"] = True
            
            # Check OpenAI API
            import os
            if os.getenv("OPENAI_API_KEY"):
                validation_results["openai_api_configured"] = True
            
            # Overall status
            if all(validation_results.values()):
                validation_results["overall_status"] = "ready"
            else:
                validation_results["overall_status"] = "issues_found"
                
        except Exception as e:
            validation_results["overall_status"] = f"error: {str(e)}"
        
        return validation_results