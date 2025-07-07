# Natural Language to SQL (NL2SQL)

Convert natural language questions into SQL queries using AI-powered multi-agent system.

## Features

- **Natural Language Processing**: Ask questions in plain English
- **Multi-Database Support**: SQLite, PostgreSQL, MySQL
- **AI-Powered**: Uses OpenAI/Anthropic models with CrewAI multi-agent framework
- **Interactive Interface**: Streamlit web application
- **Human Feedback**: Thumbs up/down system for human in the loop learning
- **Ground Truth Testing**: Compare AI queries with your own SQL
- **Zero shot accuracy**: Can achieve upto ~85% semantic sql accuracy without any human in the loop learning
- **Sample Dataset**: Pre-loaded NBA database with sample questions for instant testing
- **Performance Optimization**: Fast mode to skip schema analysis for 40-50% faster query processing
- **Token Efficiency**: Optimized with 1000 token limits and reduced verbosity for faster LLM responses
- **Schema Caching**: Intelligent caching system to avoid repeated database schema analysis
- **Streamlined Workflow**: Three-agent architecture (Schema Analyst, SQL Generator, SQL Evaluator) for optimal speed

## How to deploy locally

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
   streamlit run main.py
   ```

4. **Use the App**:
   - **Quick Start**: Click "Try Sample NBA Dataset" for instant demo with pre-loaded data
   - **Your Data**: Upload a SQLite database or add credentials for your PostgreSQL/MySQL DB
   - Ask questions like "How many teams are in the NBA?" or "How many customers do we have?"
   - Toggle Fast Mode for quicker responses (skips schema re-analysis)
   - Review generated SQL and results
   - Provide feedback to improve accuracy using like/dislike buttons
   - Compare with your own SQL queries for ground truth validation

## Architecture

Multi-agent system with specialized roles:
- **Schema Analyst**: Analyzes database structure and relationships with intelligent caching
- **SQL Generator**: Converts natural language to SQL with token-optimized processing
- **SQL Evaluator**: Validates and executes queries with error feedback loops

## Performance Features

- **Fast Mode**: Skip schema analysis for repeat queries (40-50% speed improvement)
- **Token Optimization**: 1000 token limit per agent with 30-second timeouts
- **Reduced Verbosity**: Minimal logging for faster processing
- **Schema Caching**: Reuse analyzed schemas across multiple queries
- **Streamlined Pipeline**: Removed business intelligence analysis for core SQL functionality

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