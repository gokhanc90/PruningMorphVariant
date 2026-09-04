# Runs the 11x11 (beta, Th) grid. Only UNIQUE and NON-EMPTY synonym files are
# actually retrieved; the rest are mapped back through the manifest.
#
# Usage: .\run_grid.ps1 -Collection CW09B -Parallel 2

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
$LOG  = "$ROOT\out\logs\grid"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

$manifest = Get-Content "$ROOT\out\grid_manifest.json" -Raw | ConvertFrom-Json
$jobs = @($manifest | Where-Object {
    $_.collection -eq $Collection -and $_.unique -eq $true -and $_.empty -eq $false
})

Write-Host "$Collection : $($jobs.Count) unique runs, $Parallel in parallel"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$done = 0
$marks = @(,@(0, 0.0))

for ($i = 0; $i -lt $jobs.Count; $i += $Parallel) {
    $batch = @($jobs[$i..([Math]::Min($i + $Parallel - 1, $jobs.Count - 1))])
    $procs = @()
    foreach ($j in $batch) {
        $tag  = if ($j.stemmer -eq "KStem") { "SynonymKStem" } else { "SynonymSnowballEng" }
        $runs = "G_" + [System.IO.Path]::GetFileNameWithoutExtension($j.synonym_file)
        $log  = "$LOG\$Collection`_$runs.log"
        if (Test-Path "$TFD\$Collection\$runs") { continue }
        $argList = @("-Xmx3g", "-Dfile.encoding=UTF-8", "-cp", $CP, "RunBatch",
                     $TFD, $Collection, $tag, $j.synonym_file, $runs, $Models, $Threads)
        $procs += Start-Process -FilePath "java" -ArgumentList $argList `
                    -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
                    -NoNewWindow -PassThru
    }
    if ($procs.Count) { $procs | Wait-Process }
    $done += $batch.Count
    # ETA: moving average over the last 5 batches, so a single outlier does not
    # skew the estimate.
    $marks += ,@($done, $sw.Elapsed.TotalMinutes)
    $w = [Math]::Min(5, $marks.Count)
    $a = $marks[$marks.Count - $w]
    $b = $marks[$marks.Count - 1]
    $rate = if (($b[0] - $a[0]) -gt 0) { ($b[1] - $a[1]) / ($b[0] - $a[0]) }
            else { $sw.Elapsed.TotalMinutes / [Math]::Max($done, 1) }
    $eta = [TimeSpan]::FromMinutes($rate * ($jobs.Count - $done))
    Write-Host ("  {0}/{1}  elapsed {2:hh\:mm\:ss}  {3:N2} min/config  eta {4:hh\:mm\:ss}" -f
                $done, $jobs.Count, $sw.Elapsed, $rate, $eta)
}
$sw.Stop()
Write-Host ("GRID DONE {0} -- {1:hh\:mm\:ss}" -f $Collection, $sw.Elapsed)
