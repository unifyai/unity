# Requires: Run as User. Configures TightVNC password and runs websockify to expose noVNC on http://localhost:6080/vnc.html
# Assumes dependencies are already installed via desktop/windows/install.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Stop any existing TightVNC server to avoid port conflicts
try { & net stop tvnserver | Out-Host } catch {}
Start-Sleep -Milliseconds 500

# Resolve TightVNC executable path (app-mode)
$tvnExe = "C:\\Program Files\\TightVNC\\tvnserver.exe"
if (-not (Test-Path $tvnExe)) { $tvnExe = "C:\\Program Files (x86)\\TightVNC\\tvnserver.exe" }

# Start TightVNC Server in app-mode, then reload settings from registry
try {
  Write-Host "Starting TightVNC Server in app-mode..."
  Start-Process -FilePath $tvnExe -ArgumentList "-run" -PassThru | Out-Host
} catch {
  Write-Warning "Could not start tvnserver in app-mode. $_"
}
try {
  Write-Host "Reloading TightVNC settings from registry..."
  & $tvnExe -controlapp -reload | Out-Host
} catch {
  Write-Warning "Failed to reload TightVNC settings. $_"
}

# Serve noVNC. Prefer consolidated web root prepared by installer, validate vnc.html exists
$candidateRoots = @(
  "C:\\ProgramData\\noVNC",
  "C:\\ProgramData\\chocolatey\\lib\\novnc\\tools\\web",
  "C:\\ProgramData\\chocolatey\\lib\\novnc\\tools"
)
$noVncWeb = $null
foreach ($root in $candidateRoots) {
  if (Test-Path (Join-Path $root 'vnc.html')) { $noVncWeb = $root; break }
}
if (-not $noVncWeb) {
  throw "noVNC web assets not found (vnc.html missing). Re-run install.ps1 to populate C:\\ProgramData\\noVNC."
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
  $websockifyProc = Start-Process -FilePath py -ArgumentList "-m websockify --web=$noVncWeb 6080 10.128.0.75:5900" -WindowStyle Hidden -PassThru
} else {
  $websockifyProc = Start-Process -FilePath python -ArgumentList "-m websockify --web=$noVncWeb 6080 10.128.0.75:5900" -WindowStyle Hidden -PassThru
}

# Start agent-service in foreground and clean up on Ctrl+C
Write-Host "Remote desktop available at http://localhost:6080/vnc.html"

try {
  $agentRan = $false
  if (Get-Command npx -ErrorAction SilentlyContinue) {
    try {
      & cmd.exe /c "npx --yes ts-node agent-service/src/index.ts"
      $agentRan = $true
    } catch {
      Write-Warning "Agent-service exited with error. $_"
    }
  } else {
    Write-Warning "'npx' not found. Skipping agent-service startup."
  }

  if (-not $agentRan) {
    if ($websockifyProc) {
      Wait-Process -Id $websockifyProc.Id
    } else {
      Start-Sleep -Seconds 2
    }
  }
} finally {
  Write-Host "[remote] Shutting down..."
  if ($websockifyProc) {
    try { Stop-Process -Id $websockifyProc.Id -Force -ErrorAction SilentlyContinue } catch {}
  }
  try { & $tvnExe -controlapp -shutdown | Out-Host } catch {}
}
