# Runs every method x stemmer configuration for one collection.
# They run in parallel over the same collection, so the OS page cache is shared.
#
# Usage:  .\run_all.ps1 -Collection CW09B -Parallel 2

param(
    [string]$Collection = "CW09B",
    [int]$Parallel = 2,
    [string]$Models = "BM25k1.2b0.75,DPH",
    [int]$Threads = 4
)

$ROOT = $PSScriptRoot
$DIST = "SET_LUCENE_TOOLKIT_DIR_HERE"
$CP   = "$DIST\repo\*;$DIST\FrequencyDistributionAnalysis-8.0.jar;$ROOT\java"
$TFD  = "SET_TFD_HOME_HERE"
$LOG  = "$ROOT\out\logs"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

$methods = @("subword_mean", "cls_masked", "bert_static", "sbert", "fasttext", "query_ctx")
$stems   = @{ "KStem" = "SynonymKStem"; "SnowballEng" = "SynonymSnowballEng" }

$jobs = @()
foreach ($stem in $stems.Keys) {
    foreach ($m in $methods) {
        $tag = $stems[$stem]
        $jobs += [pscustomobject]@{
            Tag      = $tag
            SynFile  = "${tag}_X_${m}.txt"
            RunsPath = "X_${m}_${stem}"
            Name     = "$Collection/$stem/$m"
            Log      = "$LOG\$Collection`_$stem`_$m.log"
        }
    }
}

Write-Host "$Collection : $($jobs.Count) configurations, $Parallel in parallel, models=$Models"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$done = 0

for ($i = 0; $i -lt $jobs.Count; $i += $Parallel) {
    $batch = $jobs[$i..([Math]::Min($i + $Parallel - 1, $jobs.Count - 1))]
    $procs = @()
    foreach ($j in $batch) {
        $syn = Join-Path (Join-Path $TFD $Collection) $j.SynFile
        if (-not (Test-Path $syn)) { Write-Host "  SKIPPED (no synonym file): $($j.Name)"; continue }
        Write-Host ("  starting: {0}" -f $j.Name)
        $argList = @("-Xmx3g", "-Dfile.encoding=UTF-8", "-cp", $CP, "RunBatch",
                     $TFD, $Collection, $j.Tag, $j.SynFile, $j.RunsPath, $Models, $Threads)
        $procs += Start-Process -FilePath "java" -ArgumentList $argList `
                    -RedirectStandardOutput $j.Log -RedirectStandardError "$($j.Log).err" `
                    -NoNewWindow -PassThru
    }
    if ($procs.Count) { $procs | Wait-Process }
    $done += $batch.Count
    Write-Host ("  progress: {0}/{1}   elapsed: {2:hh\:mm\:ss}" -f $done, $jobs.Count, $sw.Elapsed)
}

$sw.Stop()
Write-Host ("DONE {0} -- total {1:hh\:mm\:ss}" -f $Collection, $sw.Elapsed)
