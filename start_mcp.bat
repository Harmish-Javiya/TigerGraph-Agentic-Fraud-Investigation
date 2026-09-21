@echo off
REM === Start MCP Server ===
echo.
echo ============================================
echo   Starting TigerGraph MCP Server
echo ============================================
echo.

REM Kill any existing MCP server
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

REM Start MCP server
cd /d "C:\Users\123\OneDrive\Documents\TigerGraph Agentic Fraud Investigation"
start "MCP" cmd /c "venv314\Scripts\python.exe -m tigergraph_mcp run --host 0.0.0.0 --port 8000"

echo Waiting for server to start...
timeout /t 5 /nobreak >nul

REM Check port
netstat -an | findstr ":8000.*LISTENING" && (
    echo.
    echo ✅ MCP Server running on port 8000
    echo    Dashboard:  http://localhost:5000/fraud_dashboard.html
    echo    API status: http://localhost:5000/api/status
    echo    MCP logs:   mcpserver.log
) || (
    echo.
    echo ❌ MCP Server did not start
    echo    Check mcpserver.log for errors
    type mcpserver.log 2>nul | tail -20
)

echo.
