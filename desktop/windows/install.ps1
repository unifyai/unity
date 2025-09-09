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
  if (-not (Get-Command websockify -ErrorAction SilentlyContinue)) {
    choco install websockify -y --no-progress | Out-Host
  }
}

function Install-Cloudflared {
  if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    choco install cloudflared -y --no-progress | Out-Host
  }
}


Install-TightVNC
Install-NoVNC
Install-Websockify
Install-Cloudflared

Write-Host "Install complete. Use desktop\\windows\\remote.ps1 to configure and start the services."
