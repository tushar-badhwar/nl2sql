# Natural Language to SQL CrewAI Application

A production-ready application that converts natural language questions into SQL queries using CrewAI multi-agent architecture and OpenAI APIs.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for different aspects of NL2SQL conversion
- **Multiple Database Support**: SQLite, PostgreSQL, and MySQL
- **Web Interface**: Streamlit-based user-friendly interface
- **Schema Analysis**: Automatic database schema understanding
- **Query Validation**: SQL syntax and logic validation
- **Result Interpretation**: Business-friendly explanations of results
- **Performance Metrics**: Comprehensive performance tracking
- **File Upload**: SQLite database file upload capability

## 🏗️ Architecture

The application uses a multi-agent CrewAI architecture with four specialized agents:

### 1. Schema Analyst Agent
- Analyzes database schema and structure
- Identifies table relationships and constraints
- Provides context for accurate SQL generation

### 2. SQL Generator Agent
- Converts natural language questions to SQL queries
- Uses advanced prompt engineering techniques
- Handles different query types (counting, filtering, aggregation, ranking, etc.)

### 3. SQL Evaluator Agent
- Validates SQL queries for correctness
- Executes queries safely
- Provides detailed error messages and optimization suggestions

### 4. Result Interpreter Agent
- Explains query results in business-friendly terms
- Provides insights and recommendations
- Translates technical data into actionable information

## 📁 Project Structure

```
nlsql/
├── main.py                 # Streamlit app entry point
├── agents.py              # CrewAI agent definitions
├── tasks.py               # CrewAI task definitions
├── tools.py               # Custom database tools
├── database_manager.py    # Database abstraction layer
├── crew_setup.py          # Main crew orchestrator
├── requirements.txt       # Python dependencies
├── .env.template         # Environment variables template
└── README.md             # This file
```

## 🛠️ Installation

1. **Clone or download the application files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env file with your configuration
   ```

4. **Configure OpenAI API Key**:
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Add it to your `.env` file or set it in the application interface

## 🚀 Quick Start

### 1. Start the Application
```bash
streamlit run main.py
```

### 2. Configure Database Connection
- **SQLite**: Upload a database file or provide file path
- **PostgreSQL/MySQL**: Enter connection details in the sidebar

### 3. OpenAI API Access
- API access is pre-configured by the administrator
- Users do not need to enter API keys
- Select your preferred model (gpt-4o, gpt-4o-mini, etc.)

### 4. Ask Questions
- Enter natural language questions in the query interface
- View generated SQL queries and results
- Get business-friendly explanations

## 💻 Usage Examples

### Example Questions
- "How many teams are in the NBA?"
- "What are the top 5 oldest teams?"
- "List all teams from California"
- "What's the average height of NBA players?"
- "Which team has the highest scoring game?"

### Supported Query Types
- **Counting**: "How many...", "Count of..."
- **Filtering**: "List all...", "Show teams where..."
- **Aggregation**: "Average...", "Sum of...", "Maximum..."
- **Ranking**: "Top 5...", "Highest...", "Lowest..."
- **Comparison**: "Compare...", "Difference between..."
- **Detail**: "What is...", "Tell me about..."

## 🗄️ Database Support

### SQLite
- Upload database files directly in the interface
- Supports .sqlite, .db, .sqlite3 file extensions
- Ideal for development and small datasets

### PostgreSQL
- Production-ready database support
- Connection pooling and optimization
- Enterprise-grade features

### MySQL
- Wide compatibility and performance
- Supports MySQL 5.7+ and MariaDB
- Cloud database integration

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.template`:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional Database Configuration
DB_TYPE=sqlite
DB_PATH=/path/to/your/database.sqlite

# Application Settings
LOG_LEVEL=INFO
MODEL_NAME=gpt-4o
```

### Model Options

The application supports multiple OpenAI models:
- **gpt-4o** (default) - Most capable, best for complex queries
- **gpt-4o-mini** - Faster, cost-effective for simple queries  
- **gpt-4-turbo** - Good balance of capability and speed
- **gpt-3.5-turbo** - Fastest, most cost-effective

## 📊 Features Detail

### Schema Analysis
- Automatic table and column discovery
- Relationship mapping (foreign keys, primary keys)
- Data type identification
- Sample data preview
- Business context understanding

### Prompt Engineering
The application incorporates advanced prompt engineering techniques from the original notebook:
- **System-level prompts** for role definition
- **Few-shot examples** with schema context
- **Precognition** (thinking before generating SQL)
- **Query type awareness** for different SQL patterns
- **Schema relationship detection**

### Query Validation
- Syntax validation before execution
- Table and column name verification
- Performance impact assessment
- Security checks for safe execution

### Error Handling
- Comprehensive error messages
- Debugging guidance
- Automatic retry mechanisms
- Graceful degradation

## 🎯 Performance

### Metrics Tracking
- Query success rate
- Average processing time
- Model performance comparison
- Schema analysis efficiency

### Optimization
- Schema caching for repeated queries
- Connection pooling for databases
- Optimized prompt templates
- Efficient result processing

## 🔒 Security

### Database Security
- Parameterized queries to prevent SQL injection
- Connection string encryption
- Query complexity limits
- Safe execution sandbox

### API Security
- Secure API key handling
- Request rate limiting
- Error message sanitization
- Audit logging

## 🐛 Troubleshooting

### Common Issues

**1. Database Connection Failed**
- Verify connection parameters
- Check database server status
- Ensure network connectivity
- Validate credentials

**2. OpenAI API Errors**
- Verify API key is correct
- Check API quota and billing
- Ensure model availability
- Check rate limits

**3. Query Generation Issues**
- Ensure schema is properly analyzed
- Try simpler question phrasing
- Check if tables contain relevant data
- Verify column names and types

**4. Performance Issues**
- Use smaller sample datasets for testing
- Enable schema caching
- Choose appropriate model for query complexity
- Optimize database indexes

### Debug Mode

Enable verbose logging by setting `LOG_LEVEL=DEBUG` in your environment:

```bash
export LOG_LEVEL=DEBUG
streamlit run main.py
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key_here

# Run application
streamlit run main.py
```

### Production Deployment

**Using Docker** (recommended):
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Using Cloud Platforms**:
- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Use Procfile with streamlit command
- **AWS/GCP/Azure**: Container deployment with load balancing

### Environment Setup for Production
```env
OPENAI_API_KEY=your_production_key
LOG_LEVEL=INFO
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Include docstrings for all functions
- Write unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CrewAI** for the multi-agent framework
- **OpenAI** for language model APIs
- **Streamlit** for the web interface framework
- **SQLAlchemy** for database abstraction
- Original notebook implementation for prompt engineering techniques

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the configuration documentation
- Consult the CrewAI and OpenAI documentation

## 🔄 Version History

- **v1.0.0** - Initial release with full CrewAI integration
  - Multi-agent architecture
  - Streamlit web interface
  - Multi-database support
  - Performance metrics
  - Advanced prompt engineering

---

**Made with ❤️ using CrewAI and OpenAI**