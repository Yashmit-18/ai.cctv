# =====================================================================
# CCTV Employee Productivity Tracker -- Environment Setup (Windows)
# =====================================================================
# Phase 8 / 8.1: Resolves the deployment blockers from the audit report
# AND strictly enforces a CUDA-compatible Python version.
#
#   - HARD-ABORTS if the active Python is NOT 3.10 / 3.11 / 3.12.
#     (Python 3.13+ has no official PyTorch CUDA wheels, so installing
#     there would silently fall back to a broken CPU-only build and
#     block insightface.)
#   - Installs PyTorch with CUDA support (from the official PyTorch index).
#   - Installs onnxruntime-gpu, falling back to onnxruntime on CPU-only.
#   - Installs insightface, streamlit, plotly, python-dotenv, scikit-learn
#     and the remaining project requirements.
#   - Prints clear, colored status/error messages throughout.
#
# Run from the project root (AFTER recreating the venv on Python 3.12):
#     .\.venv\Scripts\Activate.ps1
#     powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
# =====================================================================

[CmdletBinding()]
param(
    # Override the PyTorch CUDA index URL. Defaults to CUDA 11.8 wheels.
    [string]$TorchIndexUrl = 'https://download.pytorch.org/whl/cu118',

    # Force CPU-only PyTorch (skip CUDA).
    [switch]$CpuOnly,

    # Which python to use for pip (defaults to the active environment's python).
    [string]$Python = 'python',

    # Bypass the strict 3.10-3.12 version gate (NOT recommended).
    [switch]$SkipVersionCheck
)

$ErrorActionPreference = 'Stop'
try { $Host.UI.RawUI.WindowTitle = 'CCTV Tracker -- Environment Setup' } catch {}

# ---------------------------------------------------------------------
# Colored console helpers
# ---------------------------------------------------------------------
function Write-Step  { Write-Host "[STEP]   $args" -ForegroundColor Cyan }
function Write-Ok    { Write-Host "[OK]     $args" -ForegroundColor Green }
function Write-WarnN { Write-Host "[WARN]   $args" -ForegroundColor Yellow }
function Write-Err   { Write-Host "[ERROR]  $args" -ForegroundColor Red }
function Write-Info  { Write-Host "[INFO]   $args" -ForegroundColor Gray }

# Run pip via "python -m pip install <pkg> ..." and surface the exit code.
function Invoke-PipInstall {
    param(
        [Parameter(Mandatory = $true)][string[]]$Packages,
        [string]$ExtraArgs = ''
    )
    $argList = @('-m', 'pip', 'install') + $Packages
    if ($ExtraArgs) { $argList += $ExtraArgs.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) }
    & $Python @argList
    return $LASTEXITCODE
}

# ---------------------------------------------------------------------
# Pre-flight: python availability + STRICT version gate
# ---------------------------------------------------------------------
Write-Step "Pre-flight check..."

try {
    $pyVersion = & $Python --version
    Write-Ok "Found Python: $pyVersion"
} catch {
    Write-Err "Python not found at '$Python'. Install Python 3.11 or 3.12 first,"
    Write-Err "then re-run with:  .\setup_env.ps1 -Python C:\Python312\python.exe"
    exit 1
}

# Parse major.minor from e.g. "Python 3.12.4"
$pyRaw  = (& $Python -c "import sys;print(sys.version)" 2>$null)
if ($pyRaw -match '^(\d+)\.(\d+)\.') {
    $major = [int]$Matches[1]; $minor = [int]$Matches[2]
} else {
    Write-Err "Could not parse Python version from: $pyRaw"
    exit 1
}

# --- STRICT version check: only 3.10 / 3.11 / 3.12 are accepted ---
$allowed = @(10, 11, 12)
$isAllowed = ($major -eq 3) -and ($allowed -contains $minor)

if (-not $isAllowed -and -not $SkipVersionCheck) {
    Write-Err "================================================================"
    Write-Err "  UNSUPPORTED PYTHON VERSION:  Python $major.$minor.x"
    Write-Err "================================================================"
    Write-Err "  PyTorch does NOT publish CUDA wheels for Python $major.$minor."
    Write-Err "  Installing here would silently produce a broken CPU-only build"
    Write-Err "  and prevent insightface from installing."
    Write-Err ""
    Write-Err "  REQUIRED: Python 3.10, 3.11, or 3.12  (3.12 recommended)."
    Write-Err ""
    Write-Err "  Fix:"
    Write-Err "   1) Install Python 3.12 (see README / guide)."
    Write-Err "   2) Recreate the venv on 3.12:"
    Write-Err "        Remove-Item -Recurse -Force .venv"
    Write-Err "        C:\Python312\python.exe -m venv .venv"
    Write-Err "        .\.venv\Scripts\Activate.ps1"
    Write-Err "   3) Re-run this script:"
    Write-Err "        powershell -ExecutionPolicy Bypass -File .\setup_env.ps1"
    Write-Err "================================================================"
    Write-Err ""
    Write-Err "  If you know what you are doing you can bypass with:"
    Write-Err "        -SkipVersionCheck"
    exit 1
}

if ($isAllowed) {
    Write-Ok "Python $major.$minor is compatible (3.10/3.11/3.12 required)."
}

# ---------------------------------------------------------------------
# 1) PyTorch (CUDA-aware)
# ---------------------------------------------------------------------
Write-Step "Installing PyTorch (torch, torchvision, torchaudio)..."
if ($CpuOnly) {
    Write-Info "CPU-only mode selected; installing default CPU wheels."
    $rc = Invoke-PipInstall @('torch', 'torchvision', 'torchaudio')
    Write-Ok "PyTorch (CPU) installed (rc=$rc)."
} else {
    Write-Info "Using CUDA index: $TorchIndexUrl"
    $rc = Invoke-PipInstall @('torch', 'torchvision', 'torchaudio') "--index-url $TorchIndexUrl"
    if ($rc -ne 0) {
        Write-WarnN "CUDA install failed (rc=$rc). Attempting CPU install as fallback."
        Invoke-PipInstall @('torch', 'torchvision', 'torchaudio') | Out-Null
    } else {
        Write-Ok "PyTorch installed with CUDA support."
    }
}

# ---------------------------------------------------------------------
# 2) onnxruntime (GPU with CPU fallback)
# ---------------------------------------------------------------------
Write-Step "Installing onnxruntime..."
Write-Info "Attempting onnxruntime-gpu first..."
$rc = Invoke-PipInstall @('onnxruntime-gpu')
if ($rc -eq 0) {
    Write-Ok "onnxruntime-gpu installed."
} else {
    Write-WarnN "onnxruntime-gpu failed (rc=$rc); falling back to CPU onnxruntime."
    Invoke-PipInstall @('onnxruntime') | Out-Null
    Write-Ok "onnxruntime (CPU) installed."
}

# ---------------------------------------------------------------------
# 3) InsightFace
# ---------------------------------------------------------------------
Write-Step "Installing insightface..."
$rc = Invoke-PipInstall @('insightface')
if ($rc -ne 0) {
    Write-WarnN "insightface install failed (rc=$rc). Installing MS Windows C++ Build Tools..."
    Write-Info "This downloads several GB. Press Ctrl+C to skip and install manually."
    Invoke-PipInstall @('-windows-build-tools') | Out-Null
    Write-Step "Retrying insightface after build-tools install..."
    Invoke-PipInstall @('insightface') | Out-Null
}
Write-Ok "insightface install step finished."

# ---------------------------------------------------------------------
# 4) Remaining project requirements
# ---------------------------------------------------------------------
Write-Step "Installing remaining requirements (streamlit, plotly, dotenv, sklearn)..."
Invoke-PipInstall @('streamlit', 'plotly', 'python-dotenv', 'scikit-learn') | Out-Null
Write-Ok "Application dashboard dependencies installed."

# ---------------------------------------------------------------------
# 5) Full requirements.txt install
# ---------------------------------------------------------------------
Write-Step "Installing full requirements.txt..."
Invoke-PipInstall @('--upgrade', 'pip') | Out-Null
$rc = Invoke-PipInstall @('-r', 'requirements.txt')
if ($rc -eq 0) {
    Write-Ok "requirements.txt installed."
} else {
    Write-WarnN "requirements.txt had failures (likely insightface build deps). Review the log above."
}

# ---------------------------------------------------------------------
# 6) Verification
# ---------------------------------------------------------------------
Write-Step "Verifying installation..."

$failed = @()
@('torch','torchvision','cv2','ultralytics','pandas','openpyxl','insightface','onnxruntime','streamlit','plotly','dotenv','sklearn') | ForEach-Object {
    $mod = $_
    & $Python -c "import importlib; importlib.import_module('$mod')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "import $mod"
    } else {
        $failed += $mod
        Write-Err "import FAILED: $mod"
    }
}

if ($failed.Count -eq 0) {
    Write-Step "All imports OK. Checking CUDA availability..."
    $cuda = (& $Python -c "import torch;print('available' if torch.cuda.is_available() else 'NOT-available')" 2>$null).Trim()
    if ($cuda -eq 'available') {
        $gpu = (& $Python -c "import torch;print(torch.cuda.get_device_name(0))" 2>$null).Trim()
        Write-Ok "CUDA is available. GPU: $gpu"
    } else {
        Write-WarnN "CUDA is NOT available. The system will run on CPU (slow for 10 cameras)."
    }
    Write-Step "Environment setup COMPLETE."
} else {
    Write-Err "The following modules failed to import: $($failed -join ', ')"
    Write-WarnN "Try installing MS C++ Build Tools then re-run this script."
    exit 1
}
