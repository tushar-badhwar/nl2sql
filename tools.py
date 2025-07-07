"""
Custom Tools for Natural Language to SQL Application
Provides specialized tools for database operations and schema analysis
"""

import logging
from typing import Dict, List, Optional, Any
from database_manager import DatabaseManager
import json
import re

logger = logging.getLogger(__name__)

class DatabaseTools:
    """Collection of database-related tools for CrewAI agents"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def connect_database(self, 
                        db_type: str, 
                        host: str = None, 
                        port: int = None, 
                        database: str = None, 
                        username: str = None, 
                        password: str = None, 
                        file_path: str = None) -> str:
        """
        Connect to database and return connection status
        
        Args:
            db_type: Database type ('sqlite', 'postgresql', 'mysql')
            host: Database host (for PostgreSQL/MySQL)
            port: Database port (for PostgreSQL/MySQL)
            database: Database name
            username: Database username (for PostgreSQL/MySQL)
            password: Database password (for PostgreSQL/MySQL)
            file_path: File path for SQLite database
            
        Returns:
            str: Connection status and basic database information
        """
        try:
            success = self.db_manager.connect(
                db_type=db_type,
                host=host,
                port=port,
                database=database,
                username=username,
                password=password,
                file_path=file_path
            )
            
            if success:
                tables = self.db_manager.get_table_names()
                return f"Successfully connected to {db_type} database. Found {len(tables)} tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}"
            else:
                return f"Failed to connect to {db_type} database"
                
        except Exception as e:
            return f"Database connection error: {str(e)}"
    
    def analyze_schema(self, table_name: str = None) -> str:
        """
        Analyze database schema for a specific table or all tables
        
        Args:
            table_name: Optional specific table name to analyze
            
        Returns:
            str: Detailed schema information in a structured format
        """
        try:
            if table_name:
                schema = self.db_manager.get_table_schema(table_name)
                if not schema:
                    return f"Table '{table_name}' not found or error retrieving schema"
                
                # Format schema information
                result = f"Schema for table '{table_name}':\n"
                result += f"Columns ({len(schema['columns'])}):\n"
                
                for col in schema['columns']:
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    result += f"  - {col['name']}: {col['type']} ({nullable})\n"
                
                if schema['primary_keys']:
                    result += f"Primary Keys: {', '.join(schema['primary_keys'])}\n"
                
                if schema['foreign_keys']:
                    result += "Foreign Keys:\n"
                    for fk in schema['foreign_keys']:
                        result += f"  - {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}\n"
                
                # Get sample data
                sample_data = self.db_manager.get_sample_data(table_name, limit=3)
                if sample_data['success'] and sample_data['data']:
                    result += f"\nSample data (first 3 rows):\n"
                    for i, row in enumerate(sample_data['data'][:3]):
                        result += f"  Row {i+1}: {json.dumps(row, default=str)}\n"
                
                return result
                
            else:
                # Get schema for all tables
                tables = self.db_manager.get_table_names()
                result = f"Database contains {len(tables)} tables:\n\n"
                
                for table in tables:
                    schema = self.db_manager.get_table_schema(table)
                    stats = self.db_manager.get_table_stats(table)
                    
                    result += f"Table: {table}\n"
                    result += f"  Rows: {stats.get('row_count', 'Unknown')}\n"
                    result += f"  Columns: {', '.join([col['name'] for col in schema.get('columns', [])])}\n"
                    
                    if schema.get('primary_keys'):
                        result += f"  Primary Keys: {', '.join(schema['primary_keys'])}\n"
                    
                    if schema.get('foreign_keys'):
                        fk_info = [f"{fk['column']}->{fk['referenced_table']}.{fk['referenced_column']}" 
                                  for fk in schema['foreign_keys']]
                        result += f"  Foreign Keys: {', '.join(fk_info)}\n"
                    
                    result += "\n"
                
                return result
                
        except Exception as e:
            return f"Schema analysis error: {str(e)}"
    
    def execute_query(self, query: str, validate_only: bool = False) -> str:
        """
        Execute or validate a SQL query
        
        Args:
            query: SQL query string
            validate_only: If True, only validate the query without executing
            
        Returns:
            str: Query results or validation information
        """
        try:
            if validate_only:
                validation = self.db_manager.validate_query(query)
                if validation['valid']:
                    return f"Query is valid:\n{query}\n\nExecution plan available."
                else:
                    return f"Query validation failed:\n{query}\n\nError: {validation['error']}"
            
            else:
                result = self.db_manager.execute_query(query)
                
                if result['success']:
                    if result['data']:
                        response = f"Query executed successfully. Returned {result['row_count']} rows.\n"
                        response += f"Columns: {', '.join(result['columns'])}\n\n"
                        
                        # Show first few rows
                        for i, row in enumerate(result['data'][:5]):
                            response += f"Row {i+1}: {json.dumps(row, default=str)}\n"
                        
                        if len(result['data']) > 5:
                            response += f"... and {len(result['data']) - 5} more rows\n"
                        
                        # Also provide flat data for single column results
                        if len(result['columns']) == 1:
                            response += f"\nFlat results: {result['flat_data'][:10]}{'...' if len(result['flat_data']) > 10 else ''}\n"
                        
                        return response
                    else:
                        return f"Query executed successfully. No rows returned or it was a non-SELECT query."
                else:
                    return f"Query execution failed:\n{query}\n\nError: {result['error']}"
                    
        except Exception as e:
            return f"Query execution error: {str(e)}"
    
    def validate_query(self, query: str) -> str:
        """
        Validate SQL query syntax and structure
        
        Args:
            query: SQL query string
            
        Returns:
            str: Validation results and feedback
        """
        try:
            # Basic syntax checks
            query = query.strip()
            if not query:
                return "Error: Empty query provided"
            
            if not query.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP')):
                return "Error: Query must start with a valid SQL statement"
            
            # Use database validation
            validation = self.db_manager.validate_query(query)
            
            if validation['valid']:
                return f"Query validation successful:\n{query}\n\nThe query syntax is correct and can be executed."
            else:
                return f"Query validation failed:\n{query}\n\nError: {validation['error']}\n\nSuggestion: Check table names, column names, and SQL syntax."
                
        except Exception as e:
            return f"Query validation error: {str(e)}"
    
    def describe_table(self, table_name: str) -> str:
        """
        Provide detailed description of a table
        
        Args:
            table_name: Name of the table to describe
            
        Returns:
            str: Comprehensive table description
        """
        try:
            # Get table schema
            schema = self.db_manager.get_table_schema(table_name)
            if not schema:
                return f"Table '{table_name}' not found in database"
            
            # Get table statistics
            stats = self.db_manager.get_table_stats(table_name)
            
            # Get sample data
            sample_data = self.db_manager.get_sample_data(table_name, limit=5)
            
            # Build comprehensive description
            description = f"Table: {table_name}\n"
            description += "=" * (len(table_name) + 7) + "\n\n"
            
            # Basic statistics
            description += f"Statistics:\n"
            description += f"  - Total rows: {stats.get('row_count', 'Unknown')}\n"
            description += f"  - Total columns: {len(schema['columns'])}\n\n"
            
            # Column information
            description += "Columns:\n"
            for col in schema['columns']:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                default_val = f", Default: {col['default']}" if col['default'] else ""
                description += f"  - {col['name']}: {col['type']} ({nullable}{default_val})\n"
            
            # Keys and constraints
            if schema['primary_keys']:
                description += f"\nPrimary Keys: {', '.join(schema['primary_keys'])}\n"
            
            if schema['foreign_keys']:
                description += "\nForeign Keys:\n"
                for fk in schema['foreign_keys']:
                    description += f"  - {fk['column']} references {fk['referenced_table']}.{fk['referenced_column']}\n"
            
            # Sample data
            if sample_data['success'] and sample_data['data']:
                description += f"\nSample Data (first 5 rows):\n"
                for i, row in enumerate(sample_data['data']):
                    description += f"  Row {i+1}: {json.dumps(row, default=str)}\n"
            
            # Usage suggestions
            description += f"\nUsage suggestions:\n"
            description += f"  - Use SELECT * FROM {table_name} to view all data\n"
            description += f"  - Use SELECT COUNT(*) FROM {table_name} to count rows\n"
            
            if schema['foreign_keys']:
                description += f"  - This table can be joined with other tables using foreign keys\n"
            
            return description
            
        except Exception as e:
            return f"Table description error: {str(e)}"
    
    def analyze_query_type(self, question: str) -> str:
        """
        Analyze natural language question to determine query type
        
        Args:
            question: Natural language question
            
        Returns:
            str: Query type analysis and recommendations
        """
        try:
            question_lower = question.lower()
            
            # Query type patterns
            query_types = {
                'counting': ['how many', 'count', 'number of', 'total number'],
                'filtering': ['list', 'show', 'find', 'which', 'what are', 'from'],
                'aggregation': ['average', 'avg', 'sum', 'total', 'maximum', 'max', 'minimum', 'min'],
                'ranking': ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'first', 'last'],
                'comparison': ['compare', 'versus', 'vs', 'difference', 'more than', 'less than'],
                'detail': ['what is', 'tell me about', 'details', 'information about'],
                'history': ['when', 'history', 'over time', 'timeline', 'trend']
            }
            
            detected_types = []
            for query_type, keywords in query_types.items():
                if any(keyword in question_lower for keyword in keywords):
                    detected_types.append(query_type)
            
            if not detected_types:
                detected_types = ['filtering']  # Default type
            
            primary_type = detected_types[0]
            
            # Generate recommendations based on query type
            recommendations = {
                'counting': 'Use COUNT() function, consider DISTINCT if needed, may need GROUP BY',
                'filtering': 'Use WHERE clause for conditions, SELECT specific columns',
                'aggregation': 'Use aggregate functions (AVG, SUM, MAX, MIN), consider GROUP BY',
                'ranking': 'Use ORDER BY with LIMIT, consider DESC for highest values',
                'comparison': 'Use WHERE with comparison operators, may need multiple conditions',
                'detail': 'Select specific columns, use WHERE for specific records',
                'history': 'Order by date/time columns, consider date ranges'
            }
            
            analysis = f"Question: {question}\n\n"
            analysis += f"Detected query type(s): {', '.join(detected_types)}\n"
            analysis += f"Primary type: {primary_type}\n\n"
            analysis += f"Recommendations for {primary_type} queries:\n"
            analysis += f"  - {recommendations.get(primary_type, 'General SQL query')}\n\n"
            
            # Additional suggestions based on type
            if primary_type == 'counting':
                analysis += "SQL patterns: SELECT COUNT(*) FROM table WHERE condition\n"
            elif primary_type == 'aggregation':
                analysis += "SQL patterns: SELECT AVG(column) FROM table GROUP BY category\n"
            elif primary_type == 'ranking':
                analysis += "SQL patterns: SELECT column FROM table ORDER BY ranking_column DESC LIMIT n\n"
            elif primary_type == 'filtering':
                analysis += "SQL patterns: SELECT columns FROM table WHERE condition\n"
            
            return analysis
            
        except Exception as e:
            return f"Query type analysis error: {str(e)}"


def create_database_tools(db_manager: DatabaseManager) -> DatabaseTools:
    """
    Create database tools instance with the given database manager
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        DatabaseTools: Database tools instance
    """
    return DatabaseTools(db_manager)