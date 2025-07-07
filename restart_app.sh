#!/bin/bash
# Script to restart the NL2SQL Streamlit application

echo "🔄 Restarting NL2SQL Application..."

# Kill existing streamlit processes
echo "Stopping existing Streamlit processes..."
pkill -f "streamlit run main.py" 2>/dev/null || true

# Wait a moment for processes to stop
sleep 2

# Start the application
echo "Starting Streamlit application..."
cd /home/tbadhwar/nlsql
streamlit run main.py --server.address=0.0.0.0 --server.port=8501 > streamlit.log 2>&1 &

# Wait a moment for startup
sleep 3

# Check if it's running
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Application started successfully!"
    echo "📍 Available at: http://localhost:8501"
    echo "📋 Users will NOT be prompted for API keys - pre-configured by admin"
else
    echo "❌ Failed to start application. Check streamlit.log for details."
    cat streamlit.log
fi