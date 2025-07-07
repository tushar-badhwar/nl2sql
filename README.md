# 🤖 Natural Language to SQL (NL2SQL)

Convert natural language questions into SQL queries using AI-powered multi-agent system.

## ✨ Features

- **Natural Language Processing**: Ask questions in plain English
- **Multi-Database Support**: SQLite, PostgreSQL, MySQL
- **AI-Powered**: Uses OpenAI GPT-4o with CrewAI multi-agent framework
- **Interactive Interface**: Streamlit web application
- **Human Feedback**: Thumbs up/down system for continuous learning
- **Ground Truth Testing**: Compare AI queries with your own SQL

## 🚀 Quick Start

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
   ```bash
   streamlit run main.py
   ```

4. **Use the App**:
   - Upload a SQLite database
   - Ask questions like "How many customers do we have?"
   - Review generated SQL and results
   - Provide feedback to improve accuracy

## 🏗️ Architecture

Multi-agent system with specialized roles:
- **Schema Analyst**: Analyzes database structure
- **SQL Generator**: Converts natural language to SQL
- **SQL Evaluator**: Validates and executes queries
- **Result Interpreter**: Explains results in business terms

## 🚀 Deploy

### Streamlit Cloud (Recommended)
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Connect this GitHub repository
3. Add `OPENAI_API_KEY` in secrets
4. Deploy!

### Docker
```bash
docker build -t nl2sql .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key nl2sql
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Credits

Built with CrewAI, OpenAI GPT-4o, and Streamlit.

**Made with ❤️ by Tushar Badhwar**