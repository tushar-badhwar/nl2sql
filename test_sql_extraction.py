#!/usr/bin/env python3
"""
Test SQL extraction functionality
"""

from main import extract_sql_from_text, extract_interpretation_from_text

def test_sql_extraction():
    """Test SQL extraction from various text formats"""
    
    test_cases = [
        # Case 1: SQL in code blocks
        {
            "text": """
Here's the SQL query to answer your question:

```sql
SELECT COUNT(*) as team_count FROM team
```

This query counts all teams in the database.
            """,
            "expected": "SELECT COUNT(*) as team_count FROM team"
        },
        
        # Case 2: SQL without code blocks
        {
            "text": """
To answer this question, I'll generate the following query:

SELECT full_name FROM team ORDER BY year_founded ASC LIMIT 5

This will return the 5 oldest teams.
            """,
            "expected": "SELECT full_name FROM team ORDER BY year_founded ASC LIMIT 5"
        },
        
        # Case 3: Mixed case SQL
        {
            "text": """
```SQL
select * from team where state = 'California'
```
            """,
            "expected": "select * from team where state = 'California'"
        }
    ]
    
    print("🧪 Testing SQL Extraction...")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {case['text'][:50]}...")
        
        extracted = extract_sql_from_text(case['text'])
        expected = case['expected']
        
        if extracted:
            print(f"✅ Extracted: {extracted}")
            if extracted.strip() == expected.strip():
                print("✅ Match!")
            else:
                print(f"❌ Expected: {expected}")
        else:
            print("❌ No SQL extracted")
    
    print("\n🧪 Testing Interpretation Extraction...")
    
    test_text = """
Here's the SQL query:

```sql
SELECT COUNT(*) FROM team
```

This query counts all the teams in the NBA database. The result shows there are 30 teams total, which represents all the franchises currently in the league.
    """
    
    sql_query = "SELECT COUNT(*) FROM team"
    interpretation = extract_interpretation_from_text(test_text, sql_query)
    
    print(f"Original text length: {len(test_text)}")
    print(f"Interpretation: {interpretation}")
    
    if "SELECT" not in interpretation:
        print("✅ SQL successfully removed from interpretation")
    else:
        print("❌ SQL still present in interpretation")

if __name__ == "__main__":
    test_sql_extraction()