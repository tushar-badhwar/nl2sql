"""
Database Manager for Natural Language to SQL Application
Provides database abstraction layer with SQLAlchemy support for multiple database types
"""

import logging
import sqlite3
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages database connections and operations for SQLite, PostgreSQL, and MySQL
    """
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.connection_string: Optional[str] = None
        self.database_type: Optional[str] = None
        self.metadata = MetaData()
        self._schema_cache: Dict[str, Dict] = {}
        
    def connect(self, 
                db_type: str, 
                host: str = None, 
                port: int = None, 
                database: str = None, 
                username: str = None, 
                password: str = None,
                file_path: str = None) -> bool:
        """
        Connect to database based on type and parameters
        
        Args:
            db_type: Database type ('sqlite', 'postgresql', 'mysql')
            host: Database host (for PostgreSQL/MySQL)
            port: Database port (for PostgreSQL/MySQL)
            database: Database name
            username: Database username (for PostgreSQL/MySQL)
            password: Database password (for PostgreSQL/MySQL)
            file_path: File path for SQLite database
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if db_type.lower() == 'sqlite':
                if not file_path:
                    raise ValueError("SQLite requires file_path parameter")
                self.connection_string = f"sqlite:///{file_path}"
                
            elif db_type.lower() == 'postgresql':
                if not all([host, database, username, password]):
                    raise ValueError("PostgreSQL requires host, database, username, and password")
                port = port or 5432
                self.connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                
            elif db_type.lower() == 'mysql':
                if not all([host, database, username, password]):
                    raise ValueError("MySQL requires host, database, username, and password")
                port = port or 3306
                self.connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
                
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
            self.engine = create_engine(self.connection_string, echo=False)
            self.database_type = db_type.lower()
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            logger.info(f"Successfully connected to {db_type} database")
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in the database
        
        Returns:
            List[str]: List of table names
        """
        if not self.engine:
            raise RuntimeError("Database not connected")
            
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            logger.error(f"Error getting table names: {str(e)}")
            return []
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a specific table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict containing table schema information
        """
        if not self.engine:
            raise RuntimeError("Database not connected")
            
        # Check cache first
        if table_name in self._schema_cache:
            return self._schema_cache[table_name]
            
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            primary_keys = inspector.get_pk_constraint(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            schema_info = {
                'table_name': table_name,
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable'],
                        'default': col.get('default')
                    }
                    for col in columns
                ],
                'primary_keys': primary_keys['constrained_columns'] if primary_keys else [],
                'foreign_keys': [
                    {
                        'column': fk['constrained_columns'][0],
                        'referenced_table': fk['referred_table'],
                        'referenced_column': fk['referred_columns'][0]
                    }
                    for fk in foreign_keys
                ]
            }
            
            # Cache the result
            self._schema_cache[table_name] = schema_info
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            return {}
    
    def get_database_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Get complete database schema information
        
        Returns:
            Dict containing schema information for all tables
        """
        tables = self.get_table_names()
        schema = {}
        
        for table in tables:
            schema[table] = self.get_table_schema(table)
            
        return schema
    
    def execute_query(self, query: str, params: Dict = None) -> Dict[str, Any]:
        """
        Execute a SQL query and return results
        
        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries
            
        Returns:
            Dict containing query results and metadata
        """
        if not self.engine:
            raise RuntimeError("Database not connected")
            
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                
                # Handle different result types
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = list(result.keys())
                    
                    # Convert to list of dictionaries
                    data = [dict(zip(columns, row)) for row in rows]
                    
                    # Also provide flattened results for single column queries
                    if len(columns) == 1:
                        flat_data = [row[0] for row in rows]
                    else:
                        flat_data = [tuple(row) for row in rows]
                    
                    return {
                        'success': True,
                        'data': data,
                        'flat_data': flat_data,
                        'columns': columns,
                        'row_count': len(rows),
                        'query': query
                    }
                else:
                    return {
                        'success': True,
                        'data': [],
                        'flat_data': [],
                        'columns': [],
                        'row_count': result.rowcount,
                        'query': query
                    }
                    
        except SQLAlchemyError as e:
            error_msg = str(e)
            logger.error(f"SQL execution error: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'query': query
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'query': query
            }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a SQL query without executing it
        
        Args:
            query: SQL query string
            
        Returns:
            Dict containing validation results
        """
        if not self.engine:
            raise RuntimeError("Database not connected")
            
        try:
            with self.engine.connect() as conn:
                # Use EXPLAIN to validate query without executing
                if self.database_type == 'sqlite':
                    explain_query = f"EXPLAIN QUERY PLAN {query}"
                else:
                    explain_query = f"EXPLAIN {query}"
                
                result = conn.execute(text(explain_query))
                
                return {
                    'valid': True,
                    'query': query,
                    'explanation': [dict(row) for row in result.fetchall()]
                }
                
        except Exception as e:
            return {
                'valid': False,
                'query': query,
                'error': str(e)
            }
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get sample data from a table
        
        Args:
            table_name: Name of the table
            limit: Number of rows to return
            
        Returns:
            Dict containing sample data
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get basic statistics for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict containing table statistics
        """
        try:
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            result = self.execute_query(count_query)
            
            if result['success']:
                row_count = result['flat_data'][0]
                schema = self.get_table_schema(table_name)
                
                return {
                    'table_name': table_name,
                    'row_count': row_count,
                    'column_count': len(schema.get('columns', [])),
                    'columns': [col['name'] for col in schema.get('columns', [])],
                    'primary_keys': schema.get('primary_keys', []),
                    'foreign_keys': schema.get('foreign_keys', [])
                }
            else:
                return {'error': result['error']}
                
        except Exception as e:
            return {'error': str(e)}
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed")
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed"""
        self.close()