#!/usr/bin/env python3
"""
Test script to verify the NL2SQL application setup
"""

import os
import sys
from database_manager import DatabaseManager
from crew_setup import NL2SQLCrew
from agents import NL2SQLAgents

def test_imports():
    """Test that all modules can be imported"""
    try:
        print("✓ Testing imports...")
        from database_manager import DatabaseManager
        from agents import NL2SQLAgents
        from tasks import NL2SQLTasks
        from tools import DatabaseTools, create_database_tools
        from crew_setup import NL2SQLCrew
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_database_manager():
    """Test database manager functionality"""
    try:
        print("✓ Testing DatabaseManager...")
        db_manager = DatabaseManager()
        print("✓ DatabaseManager created successfully!")
        return True
    except Exception as e:
        print(f"✗ DatabaseManager test failed: {e}")
        return False

def test_agents():
    """Test agent creation"""
    try:
        print("✓ Testing agents creation...")
        db_manager = DatabaseManager()
        
        # Set a dummy API key for testing
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        agents = NL2SQLAgents(db_manager, "gpt-4o")
        all_agents = agents.get_all_agents()
        
        expected_agents = ["schema_analyst", "sql_generator", "sql_evaluator", "result_interpreter"]
        for agent_name in expected_agents:
            if agent_name not in all_agents:
                raise Exception(f"Missing agent: {agent_name}")
        
        print(f"✓ All {len(all_agents)} agents created successfully!")
        return True
    except Exception as e:
        print(f"✗ Agents test failed: {e}")
        return False

def test_tools():
    """Test tools functionality"""
    try:
        print("✓ Testing tools...")
        from tools import create_database_tools
        db_manager = DatabaseManager()
        tools = create_database_tools(db_manager)
        
        # Test that tools have the expected methods
        expected_methods = [
            'connect_database', 'analyze_schema', 'execute_query', 
            'validate_query', 'describe_table', 'analyze_query_type'
        ]
        
        for method in expected_methods:
            if not hasattr(tools, method):
                raise Exception(f"Missing tool method: {method}")
        
        print("✓ All tools created successfully!")
        return True
    except Exception as e:
        print(f"✗ Tools test failed: {e}")
        return False

def test_crew_setup():
    """Test crew setup"""
    try:
        print("✓ Testing crew setup...")
        db_manager = DatabaseManager()
        
        # Set a dummy API key for testing
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        crew = NL2SQLCrew(db_manager, "gpt-4o")
        validation = crew.validate_setup()
        
        print(f"✓ Crew setup validation: {validation['overall_status']}")
        return True
    except Exception as e:
        print(f"✗ Crew setup test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running NL2SQL Application Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_database_manager,
        test_tools,
        test_agents,
        test_crew_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        print("\nTo run the application:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        print("2. Run: streamlit run main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()