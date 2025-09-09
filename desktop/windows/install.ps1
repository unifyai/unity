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

function Install-NodeTools {
  $node = Get-Command node -ErrorAction SilentlyContinue
  $npm = Get-Command npm -ErrorAction SilentlyContinue
  $npx = Get-Command npx -ErrorAction SilentlyContinue
  if (-not $node -or -not $npm -or -not $npx) {
    Write-Host "Installing Node.js LTS (includes npm and npx)..."
    choco install nodejs-lts -y --no-progress | Out-Host
    # Refresh PATH in current session
    $machinePath = [System.Environment]::GetEnvironmentVariable('Path','Machine')
    $userPath = [System.Environment]::GetEnvironmentVariable('Path','User')
    if ($machinePath -or $userPath) {
      $env:Path = ($machinePath + ';' + $userPath).Trim(';')
    }
  }
}

function Install-NativeBuildPrereqs {
  Write-Host "Installing native build prerequisites for Node.js addons..."
  try { choco install vcredist140 -y --no-progress | Out-Host } catch { Write-Warning "vcredist140 install failed or already present. $_" }
  # Visual Studio 2022 Build Tools with VC Tools workload
  try {
    choco install visualstudio2022buildtools -y --no-progress --package-parameters '--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --includeOptional' | Out-Host
  } catch {
    Write-Warning "Build Tools install encountered an issue (may already be installed). $_"
  }
  # Windows 10 SDK is commonly required by some native modules
  try { choco install windows-sdk-10 -y --no-progress | Out-Host } catch { Write-Warning "Windows 10 SDK install failed or already present. $_" }
  # Configure npm to use VS 2022 toolset
  $npm = Get-Command npm -ErrorAction SilentlyContinue
  if ($npm) {
    try { npm config set msvs_version 2022 | Out-Host } catch { Write-Warning "Failed to set npm msvs_version. $_" }
  }
}

function Install-AgentServiceDeps {
  $npm = Get-Command npm -ErrorAction SilentlyContinue
  if (-not $npm) {
    Write-Warning "npm not found; skipping agent-service dependency install."
    return
  }
  Write-Host "Installing global TypeScript tools (ts-node, typescript)..."
  try {
    npm install -g ts-node typescript | Out-Host
  } catch {
    Write-Warning "Global install of ts-node/typescript failed. Continuing with project install. $_"
  }

  # Resolve repo root and agent-service directory relative to this script (desktop/windows/ -> repo root)
  $repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
  $agentDir = Join-Path $repoRoot 'agent-service'
  if (-not (Test-Path (Join-Path $agentDir 'package.json'))) {
    Write-Warning "agent-service directory not found at $agentDir or package.json missing. Skipping npm install."
    return
  }
  Write-Host "Installing agent-service dependencies in $agentDir ..."
  Push-Location $agentDir
  try {
    if (Test-Path 'package-lock.json') {
      npm ci | Out-Host
    } else {
      npm install | Out-Host
    }
  } finally {
    Pop-Location
  }
}


Install-TightVNC
Install-NoVNC
Install-Websockify
Install-NodeTools
Install-NativeBuildPrereqs
Install-AgentServiceDeps

Write-Host "Install complete. Use desktop\\windows\\remote.ps1 to configure and start the services."
