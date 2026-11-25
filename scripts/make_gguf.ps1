# LoRA adapterlerinden nicemlenmis GGUF uretir (CPU; GPU kullanmaz). Dusuk oncelikte calisir.
#
#   scripts\make_gguf.ps1 -Name dpo -Adapter outputs\dpo-v2\adapter `
#       -BaseAdapter outputs\sft-v1\adapter -Quant Q5_K_M
#
# Cikti: outputs\merged-<Name>\ (HF), outputs\gguf\skymind-<Name>-f16.gguf, ...-<Quant>.gguf
# Gereken: third_party\llama.cpp-src (convert betigi + gguf-py), third_party\llama-cpu\llama-quantize.exe
#
# Not: dosya bilerek ASCII (Windows PowerShell 5.1 BOM'suz UTF-8'i yanlis okur). Hata kontrolu
# $LASTEXITCODE ile yapilir; uv ilerleme satirlarini stderr'e yazdigi icin "Stop" kullanilmaz.
param(
  [Parameter(Mandatory)][string]$Name,
  [Parameter(Mandatory)][string]$Adapter,
  [string]$BaseAdapter = "",
  [string]$Quant = "Q5_K_M"
)
$ErrorActionPreference = "Continue"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"   # cocuk surecler de devralir

$merged = "outputs\merged-$Name"
$f16 = "outputs\gguf\skymind-$Name-f16.gguf"
$out = "outputs\gguf\skymind-$Name-$Quant.gguf"
New-Item -ItemType Directory -Force outputs\gguf | Out-Null

$mergeArgs = @("run", "--extra", "train", "python", "-m", "skymind.optimize.merge",
               "--adapter", $Adapter, "--out", $merged)
if ($BaseAdapter) { $mergeArgs += @("--base-adapter", $BaseAdapter) }
Write-Output "[1/3] adapter birlestiriliyor -> $merged"
& uv @mergeArgs
if ($LASTEXITCODE) { throw "merge basarisiz ($LASTEXITCODE)" }

Write-Output "[2/3] GGUF (f16) -> $f16"
$env:PYTHONPATH = "third_party\llama.cpp-src\gguf-py"
& uv run --extra train --with sentencepiece python third_party\llama.cpp-src\convert_hf_to_gguf.py `
    $merged --outfile $f16 --outtype f16
if ($LASTEXITCODE) { throw "donusturme basarisiz ($LASTEXITCODE)" }

Write-Output "[3/3] $Quant -> $out"
& third_party\llama-cpu\llama-quantize.exe $f16 $out $Quant
if ($LASTEXITCODE) { throw "nicemleme basarisiz ($LASTEXITCODE)" }
Write-Output "bitti: $out"
