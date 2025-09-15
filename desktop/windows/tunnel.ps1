param(
  [string]$Hostname,
  [string]$TunnelName,
  [int]$LocalPort
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Apply defaults from env or hardcoded values
if ([string]::IsNullOrWhiteSpace($Hostname)) { $Hostname = $env:TUNNEL_HOSTNAME }
if ([string]::IsNullOrWhiteSpace($TunnelName)) {
  if (-not [string]::IsNullOrWhiteSpace($env:TUNNEL_NAME)) { $TunnelName = $env:TUNNEL_NAME } else { $TunnelName = 'myapp' }
}
if (-not $LocalPort -or $LocalPort -eq 0) { $LocalPort = 3000 }

function Ensure-Cloudflared {
  $cf = Get-Command cloudflared -ErrorAction SilentlyContinue
  if ($cf) {
    Write-Host "[tunnel] cloudflared is available: $((cloudflared --version) -split "`n")[0]"
    return
  }
  Write-Host "[tunnel] cloudflared not found. Attempting installation via Chocolatey..."
  $choco = Get-Command choco -ErrorAction SilentlyContinue
  if (-not $choco) {
    throw "[tunnel] Chocolatey not found and cloudflared missing. Install cloudflared from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/ and retry."
  }
  choco install cloudflared -y --no-progress | Out-Host
  $cf = Get-Command cloudflared -ErrorAction SilentlyContinue
  if (-not $cf) {
    throw "[tunnel] cloudflared installation appears to have failed or PATH not updated. Open a new PowerShell window and retry."
  }
}

function Get-CredentialsFilePath {
  param(
    [string]$CloudflaredDir
  )
  $latest = Get-ChildItem -Path $CloudflaredDir -Filter '*.json' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($latest) { return $latest.FullName }
  return $null
}

function Write-ConfigYaml {
  param(
    [string]$CloudflaredDir,
    [string]$TunnelName,
    [string]$CredentialsFile,
    [string]$Hostname,
    [int]$LocalPort
  )
  $configPath = Join-Path $CloudflaredDir 'config.yml'
  $yaml = @"
tunnel: $TunnelName
credentials-file: $CredentialsFile
ingress:
  - hostname: $Hostname
    service: http://localhost:$LocalPort
  - service: http_status:404
"@
  Set-Content -Path $configPath -Value $yaml -Encoding UTF8
  return $configPath
}

Write-Host "[tunnel] Starting Cloudflare Tunnel setup..."
Ensure-Cloudflared

$cfDir = Join-Path $env:USERPROFILE '.cloudflared'
if (-not (Test-Path $cfDir)) {
  New-Item -ItemType Directory -Force -Path $cfDir | Out-Null
}

# if ([string]::IsNullOrWhiteSpace($Hostname)) {
Write-Host "[tunnel] INFO: No hostname provided. Starting ad-hoc tunnel to http://localhost:$LocalPort ..."
cloudflared tunnel --url "http://localhost:$LocalPort"
#   exit $LASTEXITCODE
# }

# if (-not (Test-Path (Join-Path $cfDir 'cert.pem'))) {
#   Write-Host "[tunnel] ERROR: cloudflared is not logged in. Run: cloudflared tunnel login" -ForegroundColor Red
#   exit 1
# }

# # Create tunnel if missing
# $tunnelExists = $false
# try {
#   & cloudflared tunnel info "$TunnelName" | Out-Null
#   if ($LASTEXITCODE -eq 0) { $tunnelExists = $true }
# } catch {}

# $credentialsFile = $null
# if (-not $tunnelExists) {
#   Write-Host "[tunnel] Creating tunnel '$TunnelName'..."
#   $createOut = & cloudflared tunnel create "$TunnelName" 2>&1
#   $pattern = [Regex]::Escape($cfDir) + "\\[a-f0-9-]+\.json"
#   $match = [Regex]::Match($createOut, $pattern)
#   if ($match.Success) { $credentialsFile = $match.Value }
#   if (-not $credentialsFile) {
#     $credentialsFile = Get-CredentialsFilePath -CloudflaredDir $cfDir
#   }
# } else {
#   $credentialsFile = Get-CredentialsFilePath -CloudflaredDir $cfDir
# }

# if (-not $credentialsFile -or -not (Test-Path $credentialsFile)) {
#   Write-Host "[tunnel] ERROR: Could not find tunnel credentials JSON in $cfDir" -ForegroundColor Red
#   exit 1
# }

# $configPath = Write-ConfigYaml -CloudflaredDir $cfDir -TunnelName $TunnelName -CredentialsFile $credentialsFile -Hostname $Hostname -LocalPort $LocalPort

# try {
#   & cloudflared tunnel route dns "$TunnelName" "$Hostname" | Out-Host
# } catch {}

# Write-Host "[tunnel] Running tunnel '$TunnelName' for https://$Hostname → http://localhost:$LocalPort"
# & cloudflared tunnel run "$TunnelName"
