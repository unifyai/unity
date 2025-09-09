# Requires: Run as Administrator. Installs TightVNC, noVNC, websockify, and cloudflared.
# Run desktop/windows/remote.ps1 afterward to configure/start services.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'


function Install-TightVNC {
  if (-not (Get-Command tvnserver -ErrorAction SilentlyContinue)) {
    choco install tightvnc -y --no-progress | Out-Host
  }
}

function Install-NoVNC {
  if (-not (Test-Path "C:\\ProgramData\\noVNC")) {
    choco install novnc -y --no-progress | Out-Host
    New-Item -ItemType Directory -Force -Path "C:\\ProgramData\\noVNC" | Out-Null
  }
}

function Install-Websockify {
  # Prefer Python module per official README: https://github.com/novnc/websockify
  $hasPy = Get-Command py -ErrorAction SilentlyContinue
  $hasPython = Get-Command python -ErrorAction SilentlyContinue
  if (-not $hasPy -and -not $hasPython) {
    choco install python -y --no-progress | Out-Host
  }
  try { py -m pip install --user --upgrade pip | Out-Host } catch {}
  try {
    py -m pip install --user --upgrade websockify | Out-Host
  } catch {
    try {
      python -m pip install --user --upgrade websockify | Out-Host
    } catch {
      Write-Warning "Failed to install websockify via pip. Ensure Python is installed and accessible. $_"
    }
  }
}


Install-TightVNC
Install-NoVNC
Install-Websockify

Write-Host "Install complete. Use desktop\\windows\\remote.ps1 to configure and start the services."
