# Redirects every download / cache / temp path into the folder holding the code,
# so that nothing lands on the system drive.
# Usage:  . .\env.ps1      (must be dot-sourced: a dot, then a space)

$ROOT = $PSScriptRoot

$env:PIP_CACHE_DIR   = "$ROOT\.cache\pip"
$env:HF_HOME         = "$ROOT\.cache\huggingface"
$env:HF_HUB_CACHE    = "$ROOT\.cache\huggingface\hub"
$env:TRANSFORMERS_CACHE = "$ROOT\.cache\huggingface\transformers"
$env:TORCH_HOME      = "$ROOT\.cache\torch"
$env:XDG_CACHE_HOME  = "$ROOT\.cache"
$env:TMP             = "$ROOT\.cache\tmp"
$env:TEMP            = "$ROOT\.cache\tmp"
$env:MPLCONFIGDIR    = "$ROOT\.cache\matplotlib"

foreach ($d in @($env:PIP_CACHE_DIR, $env:HF_HOME, $env:TORCH_HOME, $env:TMP, $env:MPLCONFIGDIR)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

Write-Host "Cache/temp -> $ROOT\.cache"
