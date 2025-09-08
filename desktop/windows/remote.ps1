# Requires: Run as Administrator. Configures TightVNC password and runs websockify to expose noVNC on http://localhost:6080/vnc.html
# Assumes dependencies are already installed via desktop/windows/install.ps1

param(
  [string]$VncPassword = "changeme"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Configure TightVNC password (8-char limit). TightVNC stores hashed password in registry.
Write-Host "Configuring TightVNC password..."
& tvnserver -controlapp -passwd $VncPassword | Out-Host

# Start TightVNC Server
Write-Host "Starting TightVNC Server..."
Start-Process -FilePath tvnserver -ArgumentList "-start" -WindowStyle Hidden

# Serve noVNC. Try to detect noVNC web root
$noVncWeb = "C:\\ProgramData\\chocolatey\\lib\\novnc\\tools\\web";
if (-not (Test-Path $noVncWeb)) {
  $noVncWeb = "C:\\ProgramData\\chocolatey\\lib\\novnc\\tools";
}

Write-Host "Starting websockify on http://localhost:6080/vnc.html"
Start-Process -FilePath websockify -ArgumentList "--web=$noVncWeb 6080 localhost:5900" -WindowStyle Hidden

Write-Host "Remote desktop available at http://localhost:6080/vnc.html"
