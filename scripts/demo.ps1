# SkyMind uctan uca demo (Windows, PowerShell 5.1). ASCII only.
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 [-Model outputs\gguf\skymind-sft-Q5_K_M.gguf] [-Port 8091]
# C++ sunucuyu (BelowNormal oncelik) baslatir, saglik kontrolunden sonra demoyu calistirir, sunucuyu kapatir.
param(
    [string]$Model = "outputs\gguf\skymind-sft-Q5_K_M.gguf",
    [int]$Port = 8091
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = "build\server\skymind_server.exe"
if (-not (Test-Path $exe)) { Write-Host "Sunucu derlenmemis: cpp\README.md"; exit 2 }
if (-not (Test-Path $Model)) { Write-Host "GGUF yok: $Model (scripts\make_gguf.ps1)"; exit 2 }

$srv = Start-Process -FilePath $exe -ArgumentList "-m $Model --port $Port --ngl 99 --system cpp\configs\system_short.txt" -NoNewWindow -PassThru
$code = 1
try {
    $srv.PriorityClass = "BelowNormal"
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        try { Invoke-RestMethod "http://127.0.0.1:$Port/healthz" -TimeoutSec 2 | Out-Null; $ok = $true; break } catch { }
    }
    if (-not $ok) { Write-Host "Sunucu 60 sn icinde hazir olmadi"; exit 3 }
    $env:PYTHONIOENCODING = "utf-8"
    uv run --no-sync --extra serve python -m skymind.demo --url "http://127.0.0.1:$Port"
    $code = $LASTEXITCODE
} finally {
    if ($srv -and -not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
}
exit $code
