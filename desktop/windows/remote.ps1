# Requires: Run as Administrator. Configures TightVNC password and runs websockify to expose noVNC on http://localhost:6080/vnc.html
# Assumes dependencies are already installed via desktop/windows/install.ps1

param(
  [string]$VncPassword = "changeme"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Configure TightVNC password (8-char limit). TightVNC stores hashed password in registry.
# Write-Host "Configuring TightVNC password..."
# & tvnserver -controlapp -passwd $VncPassword | Out-Host

# # Start TightVNC Server
# Write-Host "Starting TightVNC Server..."
# Start-Process -FilePath tvnserver -ArgumentList "-start" -WindowStyle Hidden

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
if ($py) {
  Start-Process -FilePath py -ArgumentList "-m websockify --web=$noVncWeb 6080 localhost:5900" -WindowStyle Hidden
} else {
  Start-Process -FilePath python -ArgumentList "-m websockify --web=$noVncWeb 6080 localhost:5900" -WindowStyle Hidden
}

Write-Host "Remote desktop available at http://localhost:6080/vnc.html"
