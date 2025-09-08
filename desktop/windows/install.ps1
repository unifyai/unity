# Requires: Run as Administrator. Installs TightVNC, noVNC, websockify, and cloudflared.
# Run desktop/windows/remote.ps1 afterward to configure/start services.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ensure-Choco {
  if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
      try {
        winget install --id Chocolatey.Chocolatey --source winget --accept-package-agreements --accept-source-agreements | Out-Host
      } catch {
        Write-Warning "winget install of Chocolatey failed, falling back to bootstrap script."
      }
    } else {
      Write-Host "winget not found; using Chocolatey bootstrap script."
    }
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
      Set-ExecutionPolicy Bypass -Scope Process -Force
      [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
      Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    }
  }
}

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
    winget install --id Cloudflare.cloudflared --accept-package-agreements --accept-source-agreements | Out-Host
  }
}

Ensure-Choco
Install-TightVNC
Install-NoVNC
Install-Websockify
Install-Cloudflared

Write-Host "Install complete. Use desktop\\windows\\remote.ps1 to configure and start the services."
