# 🤖 Natural Language to SQL (NL2SQL)

Convert natural language questions into SQL queries using AI-powered multi-agent system.

## ✨ Features

- **Natural Language Processing**: Ask questions in plain English
- **Multi-Database Support**: SQLite, PostgreSQL, MySQL
- **AI-Powered**: Uses OpenAI/Anthropic models with CrewAI multi-agent framework
- **Interactive Interface**: Streamlit web application
- **Human Feedback**: Thumbs up/down system for human in the loop learning
- **Ground Truth Testing**: Compare AI queries with your own SQL
- **Zero shot accuracy**: Can achieve upto ~85% semantic sql accuracy without any human in the loop learning.

## 🚀 How to deploy locally

1. **Clone and Install**:
   ```bash
   git clone https://github.com/tushar-badhwar/nl2sql.git
   cd nl2sql
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

3. **Run Application**:
   - Make sure you have streamlit installed:
   ```bash
   streamlit --version
   ```
   - If the above returns an error, run this:
   ```bash
   pip install streamlit
   ```
   ```bash
   streamlit run main.py
   ```

4. **Use the App**:
   - Upload a SQLite database or add crednetials for your PostgresSQL/MySQL DB
   - Ask questions like "How many customers do we have?"
   - Review generated SQL and results
   - Provide feedback to improve accuracy using like/dislike buttons

## 🏗️ Architecture

Multi-agent system with specialized roles:
- **Schema Analyst**: Analyzes database structure
- **SQL Generator**: Converts natural language to SQL
- **SQL Evaluator**: Validates and executes queries
- **Result Interpreter**: Explains results in business terms


### Docker
```bash
docker build -t nl2sql .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key nl2sql
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Credits

Built with CrewAI, LLMs, and Streamlit.

**Made with ❤️ by Tushar Badhwar**