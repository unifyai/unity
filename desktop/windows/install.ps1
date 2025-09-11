# Requires: Run as Administrator. Installs TightVNC, noVNC, websockify, and cloudflared.
# Run desktop/windows/remote.ps1 afterward to configure/start services.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'


function Install-TightVNC {

  if (-not (Get-Command tvnserver -ErrorAction SilentlyContinue)) {
    Write-Host "Installing TightVNC..."
    choco install tightvnc -y --no-progress | Out-Host
  } else {
    Write-Host "TightVNC already installed. Ensuring settings..."
  }

  # Enforce auth in registry and allow RFB connections
  $regPaths = @('HKLM:\Software\TightVNC\Server','HKLM:\Software\WOW6432Node\TightVNC\Server')
  foreach ($p in $regPaths) {
    if (Test-Path $p) {
      New-ItemProperty -Path $p -Name 'UseVncAuthentication' -PropertyType DWord -Value 1 -Force | Out-Null
      New-ItemProperty -Path $p -Name 'AcceptRfbConnections' -PropertyType DWord -Value 1 -Force | Out-Null
      New-ItemProperty -Path $p -Name 'QueryIfNoPassword' -PropertyType DWord -Value 0 -Force | Out-Null
      New-ItemProperty -Path $p -Name 'RfbPort' -PropertyType DWord -Value 5900 -Force | Out-Null
    }
  }

  # NOTE: TightVNC does not provide a CLI to set plaintext passwords.
  # Ensure a password is configured via GUI once (or pre-hashed in registry) then use -reload.
  $tvnExe = 'C:\\Program Files\\TightVNC\\tvnserver.exe'
  if (-not (Test-Path $tvnExe)) { $tvnExe = 'C:\\Program Files (x86)\\TightVNC\\tvnserver.exe' }
  if (Test-Path $tvnExe) {
    try { & $tvnExe -configapp | Out-Host } catch {}
  }
}

function Install-NoVNC {
  $dest = "C:\\ProgramData\\noVNC"
  if (-not (Test-Path $dest)) {
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
  }
  $vncHtml = Join-Path $dest 'vnc.html'
  if (Test-Path $vncHtml) {
    Write-Host "noVNC already present at $dest"
    return
  }

  Write-Host "Installing noVNC to $dest"
  $tempBase = Join-Path $env:TEMP ("novnc_install_" + [Guid]::NewGuid().ToString())
  New-Item -ItemType Directory -Force -Path $tempBase | Out-Null
  $zipUrl = 'https://github.com/novnc/noVNC/archive/refs/heads/master.zip'
  $zipPath = Join-Path $tempBase 'noVNC-master.zip'
  try {
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
    Expand-Archive -LiteralPath $zipPath -DestinationPath $tempBase -Force
    $extracted = Join-Path $tempBase 'noVNC-master'
    if (-not (Test-Path $extracted)) { throw "Extracted folder not found at $extracted" }
    Write-Host "Copying files to $dest ..."
    Copy-Item -Path (Join-Path $extracted '*') -Destination $dest -Recurse -Force
  } catch {
    Write-Warning "noVNC download/extract failed. Falling back to Chocolatey. $_"
    try {
      choco install novnc -y --no-progress | Out-Host
      $chocoWeb = "C:\\ProgramData\\chocolatey\\lib\\novnc\\tools\\web"
      if (Test-Path (Join-Path $chocoWeb 'vnc.html')) {
        Copy-Item -Path (Join-Path $chocoWeb '*') -Destination $dest -Recurse -Force
      }
    } catch {
      Write-Warning "Chocolatey noVNC install also failed. $_"
    }
  } finally {
    try { Remove-Item -Path $tempBase -Recurse -Force -ErrorAction SilentlyContinue } catch {}
    try { Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue } catch {}
  }

  if (-not (Test-Path (Join-Path $dest 'vnc.html'))) {
    throw "noVNC installation did not produce vnc.html at $dest."
  }
  Write-Host "noVNC installed at $dest"
}

function Install-Websockify {
  # Prefer Python module per official README: https://github.com/novnc/websockify
  $hasPy = Get-Command py -ErrorAction SilentlyContinue
  $hasPython = Get-Command python -ErrorAction SilentlyContinue
  if (-not $hasPy -and -not $hasPython) {
    choco install python -y --no-progress | Out-Host
  }
  try { py -m pip install --upgrade pip | Out-Host } catch {}
  try {
    py -m pip install --upgrade websockify | Out-Host
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
      npx playwright@1.52.0 install --with-deps chromium | Out-Host
    } else {
      npm install | Out-Host
      npx playwright@1.52.0 install --with-deps chromium | Out-Host
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
