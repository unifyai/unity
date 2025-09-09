# Requires: Run as Administrator. Configures TightVNC password and runs websockify to expose noVNC on http://localhost:6080/vnc.html
# Assumes dependencies are already installed via desktop/windows/install.ps1

param(
  [string]$VncPassword = "changeme",
  [string]$UnifyKey
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Derive VNC password from UNIFY_KEY if provided
if (-not $UnifyKey -and $env:UNIFY_KEY) {
  $UnifyKey = $env:UNIFY_KEY
}
# if ($UnifyKey) {
#   $VncPassword = $UnifyKey
# }
# # TightVNC enforces 8-char max password
# if ($VncPassword.Length -gt 8) {
#   Write-Warning "VNC password exceeds 8 characters. Truncating to 8."
#   $VncPassword = $VncPassword.Substring(0,8)
# }
# if (-not $VncPassword) {
#   throw "VNC password is empty. Set -UnifyKey parameter or UNIFY_KEY environment variable."
# }

# Stop any existing TightVNC server to avoid port conflicts
try { & net stop tvnserver | Out-Host } catch {}
Start-Sleep -Milliseconds 500

# Configure TightVNC password and start the server
# try {
#   Write-Host "Configuring TightVNC password..."
#   & tvnserver -controlapp -passwd $VncPassword | Out-Host
# } catch {
#   Write-Warning "Failed to set VNC password via tvnserver controlapp. Ensure TightVNC is installed and on PATH. $_"
# }
try {
  Write-Host "Starting TightVNC Server..."
  & net start tvnserver | Out-Host
} catch {
  Write-Warning "Could not start tvnserver via 'net start'. $_"
}

# Serve noVNC. Prefer consolidated web root prepared by installer
$noVncWeb = "C:\\ProgramData\\noVNC";
if (-not (Test-Path $noVncWeb)) {
  $noVncWeb = "C:\\ProgramData\\chocolatey\\lib\\novnc\\tools\\web";
  if (-not (Test-Path $noVncWeb)) {
    $noVncWeb = "C:\\ProgramData\\chocolatey\\lib\\novnc\\tools";
  }
}

if (-not (Test-Path $noVncWeb)) {
  throw "noVNC web assets not found. Ensure novnc is installed and the web assets exist at C:\\ProgramData\\noVNC or Chocolatey novnc tools directory."
}

# Start websockify via Python module (per https://github.com/novnc/websockify and guide https://mannygyan.com/novnc/#toc-4)
$py = Get-Command py -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $py -and -not $python) {
  throw "Python is required to run websockify. Install Python or run install.ps1 first."
}

Write-Host "Starting websockify on http://localhost:6080/vnc.html (proxying to localhost:5900)"
$websockifyProc = $null
if ($py) {
  $websockifyProc = Start-Process -FilePath py -ArgumentList "-m websockify --web=$noVncWeb 6080 localhost:5900" -WindowStyle Hidden -PassThru
} else {
  $websockifyProc = Start-Process -FilePath python -ArgumentList "-m websockify --web=$noVncWeb 6080 localhost:5900" -WindowStyle Hidden -PassThru
}

# Start agent-service like Linux script
$agentProc = $null
if (Get-Command npx -ErrorAction SilentlyContinue) {
  try {
    $agentProc = Start-Process -FilePath npx -ArgumentList "ts-node agent-service/src/index.ts" -PassThru
  } catch {
    Write-Warning "Failed to start agent-service via npx. $_"
  }
} else {
  Write-Warning "'npx' not found. Skipping agent-service startup."
}

Write-Host "Remote desktop available at http://localhost:6080/vnc.html"

# Wait and cleanup similar to Linux script
try {
  if ($agentProc) {
    Wait-Process -Id $agentProc.Id
  } elseif ($websockifyProc) {
    Wait-Process -Id $websockifyProc.Id
  } else {
    Start-Sleep -Seconds 2
  }
} finally {
  Write-Host "[remote] Shutting down..."
  if ($websockifyProc) {
    try { Stop-Process -Id $websockifyProc.Id -Force -ErrorAction SilentlyContinue } catch {}
  }
  try { & net stop tvnserver | Out-Host } catch {}
}
