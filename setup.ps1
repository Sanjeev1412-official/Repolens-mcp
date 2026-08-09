<#
.SYNOPSIS
    Automated setup script for RepoLens MCP on Windows.

.DESCRIPTION
    Creates a virtual environment, installs dependencies, runs the test suite,
    and outputs the correct Claude Desktop configuration snippet.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  RepoLens MCP Setup (Windows)" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

# 1. Create Virtual Environment
$venvDir = ".venv"
if (-Not (Test-Path $venvDir)) {
    Write-Host "-> Creating virtual environment..." -ForegroundColor Cyan
    python -m venv $venvDir
} else {
    Write-Host "-> Virtual environment already exists." -ForegroundColor Cyan
}

# 2. Install Dependencies
Write-Host "-> Installing dependencies..." -ForegroundColor Cyan
& ".\$venvDir\Scripts\python.exe" -m pip install --upgrade pip
& ".\$venvDir\Scripts\python.exe" -m pip install -e .[dev]

# 3. Run Tests
Write-Host "-> Running test suite..." -ForegroundColor Cyan
$env:PYTHONIOENCODING="utf-8"
& ".\$venvDir\Scripts\python.exe" -m pytest tests/ -v --tb=short
$testResult = $LASTEXITCODE

if ($testResult -ne 0) {
    Write-Host ""
    Write-Host "[!] Tests failed. Please check the output above." -ForegroundColor Red
    exit $testResult
} else {
    Write-Host ""
    Write-Host "✓ All tests passed successfully!" -ForegroundColor Green
}

# 4. Generate Claude Desktop Config Snippet
$repoPath = (Get-Item .).FullName
$pythonPath = (Join-Path $repoPath ".venv\Scripts\python.exe")
$serverPath = (Join-Path $repoPath "src\server.py")
$chromaPath = (Join-Path $repoPath "chroma_db")

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  Setup Complete! " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "To use with Claude Desktop, add the following to your config file:" -ForegroundColor Cyan
Write-Host "(Usually located at %APPDATA%\Claude\claude_desktop_config.json)"
Write-Host ""

$configSnippet = @"
{
  "mcpServers": {
    "repolens": {
      "command": "$pythonPath",
      "args": [
        "$serverPath"
      ],
      "env": {
        "REPO_PATH": "$repoPath",
        "CHROMA_PATH": "$chromaPath",
        "PYTHONPATH": "$repoPath"
      }
    }
  }
}
"@

Write-Host $configSnippet -ForegroundColor Yellow
Write-Host ""
Write-Host "To use the dev inspector, run: .venv\Scripts\fastmcp dev src\server.py" -ForegroundColor Cyan
