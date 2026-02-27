# Ensure errors stop execution
$ErrorActionPreference = "Stop"

# Path to your virtual environment
$venvPath = ".\venv"

# Activate the virtual environment
Write-Host "Activating virtual environment..."
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Error "Virtual environment activation script not found at $activateScript"
    exit 1
}

# Run your Python script
$pythonScript = ".\csv_manager\main.py"  # <-- replace with your app filename
if (Test-Path $pythonScript) {
    Write-Host "Running Python app..."
    python $pythonScript
} else {
    Write-Error "Python script not found at $pythonScript"
    exit 1
}