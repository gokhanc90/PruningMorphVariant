# Re-runs the baseline systems through the verified pipeline.
# NoStem / Porter / KStem / HPS / QBS-Porter / QBS-KStem

param(
    [string]$Collection = "CW09B",
    [int]$Parallel = 2,
    [string]$Models = "BM25k1.2b0.75,DPH",
    [int]$Threads = 4,
    [string[]]$Only = @()      # example: -Only NoStem,Porter,KStem
)

$ROOT = $PSScriptRoot
$DIST = "SET_LUCENE_TOOLKIT_DIR_HERE"
$CP   = "$DIST\repo\*;$DIST\FrequencyDistributionAnalysis-8.0.jar;$ROOT\java"
$TFD  = "SET_TFD_HOME_HERE"
$LOG  = "$ROOT\out\logs"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

# name, Tag, synonym file, heap, threads
# The HPS synonym file is very large (1.8 MB): every thread builds its own
# SynonymMap, so it needs a large heap and few threads. QBS takes the weighted
# query path (PMI/IDF read from the index), which also costs extra memory.
$base = @(
    @("NoStem",      "NoStem",                "NONE",                   "3g", 4),
    @("Porter",      "SynonymSnowballEng",    "SynonymSnowballEng.txt", "3g", 4),
    @("KStem",       "SynonymKStem",          "SynonymKStem.txt",       "3g", 4),
    @("HPS",         "SynonymHPS",            "SynonymHPS.txt",         "9g", 2),
    @("QBS-Porter",  "SynonymSnowballEngQBS", "SynonymSnowballEng.txt", "9g", 2),
    @("QBS-KStem",   "SynonymKStemQBS",       "SynonymKStem.txt",       "9g", 2)
)

if ($Only.Count) { $base = @($base | Where-Object { $Only -contains $_[0] }) }
Write-Host "$Collection : $($base.Count) baselines, $Parallel in parallel"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

for ($i = 0; $i -lt $base.Count; $i += $Parallel) {
    $batch = $base[$i..([Math]::Min($i + $Parallel - 1, $base.Count - 1))]
    $procs = @()
    foreach ($b in $batch) {
        $name, $tag, $syn, $heap, $thr = $b[0], $b[1], $b[2], $b[3], $b[4]
        $runs = "B_$name"
        $log = Join-Path $LOG ("{0}_baseline_{1}.log" -f $Collection, $name)
        Write-Host "  starting: $Collection/$name  (heap=$heap threads=$thr)"
        $argList = @("-Xmx$heap", "-Dfile.encoding=UTF-8", "-cp", $CP, "RunBatch",
                     $TFD, $Collection, $tag, $syn, $runs, $Models, $thr)
        $procs += Start-Process -FilePath "java" -ArgumentList $argList `
                    -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
                    -NoNewWindow -PassThru
    }
    $procs | Wait-Process
    Write-Host ("  progress: {0}/{1}  elapsed: {2:hh\:mm\:ss}" -f
                [Math]::Min($i + $Parallel, $base.Count), $base.Count, $sw.Elapsed)
}
$sw.Stop()

# --- check that the outputs are valid (an empty run file means the run failed)
Write-Host "`nOUTPUT CHECK"
$bad = 0
foreach ($b in $base) {
    $name, $tag = $b[0], $b[1]
    $dir = Join-Path (Join-Path $TFD $Collection) "B_$name"
    $files = @(Get-ChildItem -Path $dir -Recurse -Filter "*.txt" -ErrorAction SilentlyContinue)
    $empty = @($files | Where-Object { $_.Length -eq 0 })
    $status = if ($files.Count -eq 0) { "NO FILES" }
              elseif ($empty.Count -gt 0) { "EMPTY: $($empty.Count)/$($files.Count)" }
              else { "ok ($($files.Count) files)" }
    if ($status -ne "ok ($($files.Count) files)") { $bad++ }
    Write-Host ("  {0,-14} {1}" -f $name, $status)
}
Write-Host ("BASELINES DONE {0} -- {1:hh\:mm\:ss}  failed: {2}" -f $Collection, $sw.Elapsed, $bad)
