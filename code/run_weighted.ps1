# Runs the weighted-query systems: QBS (full NPMI) and MLFN-ART (predicted NPMI).
#
# Both use the same driver (RunMlfn) and the same tf scheme
# (SumRoundSynonymQuery); the ONLY difference is where NPMI comes from. That
# makes "how much does prediction cost?" a controlled question.
#
# Usage: .\run_weighted.ps1 -Collection CW09B -System QBS
#        .\run_weighted.ps1 -Collection CW09B -System MLFN

param(
    [string]$Collection = "CW09B",
    [ValidateSet("QBS", "MLFN")][string]$System = "QBS",
    [string]$Models = "BM25k1.2b0.75,DPH",
    [string]$Heap = "4g"
)

$ROOT = $PSScriptRoot
$DIST = "SET_LUCENE_TOOLKIT_DIR_HERE"
$CP   = "$DIST\repo\*;$DIST\FrequencyDistributionAnalysis-8.0.jar;$ROOT\java"
$TFD  = "SET_TFD_HOME_HERE"
$LOG  = "$ROOT\out\logs"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

# stemmer -> (QBS tag, base synonym file, MLFN selection file, MLFN prediction file)
$cfg = @(
    @{ stem = "SnowballEng"; tag = "SynonymSnowballEngQBS"
       baseSyn = "SynonymSnowballEng.txt"
       mlfnSyn = "SynonymSnowballEng_MLFN_sel.txt"
       mlfnPred = "SynonymSnowballEngMLFN.txt" },
    @{ stem = "KStem"; tag = "SynonymKStemQBS"
       baseSyn = "SynonymKStem.txt"
       mlfnSyn = "SynonymKStem_MLFN_sel.txt"
       mlfnPred = "SynonymKStemMLFN.txt" }
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
foreach ($c in $cfg) {
    if ($System -eq "QBS") {
        $syn  = $c.baseSyn        # the full conflation set (QBS weights, it does not prune)
        $pred = "NONE"            # -> full NPMI, read from the index
        $runs = "W_QBS_" + $c.stem
    } else {
        $syn  = $c.mlfnSyn        # the set selected by predicted NPMI > 0.30
        $pred = $c.mlfnPred
        $runs = "W_MLFN_" + $c.stem
    }
    $log = Join-Path $LOG ("{0}_{1}_{2}.log" -f $Collection, $System, $c.stem)
    Write-Host ("  {0} / {1} / {2}  -> {3}" -f $Collection, $System, $c.stem, $runs)

    $argList = @("-Xmx$Heap", "-Dfile.encoding=UTF-8", "-cp", $CP, "RunMlfn",
                 $TFD, $Collection, $c.tag, $syn, $pred, "ART", $runs, $Models)
    $p = Start-Process -FilePath "java" -ArgumentList $argList `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
            -NoNewWindow -PassThru
    $p | Wait-Process
    Write-Host ("     elapsed: {0:hh\:mm\:ss}" -f $sw.Elapsed)
}

Write-Host "`nOUTPUT CHECK"
foreach ($c in $cfg) {
    $runs = if ($System -eq "QBS") { "W_QBS_" + $c.stem } else { "W_MLFN_" + $c.stem }
    $dir = Join-Path (Join-Path $TFD $Collection) $runs
    $f = @(Get-ChildItem $dir -Recurse -Filter *.txt -ErrorAction SilentlyContinue)
    $empty = @($f | Where-Object { $_.Length -eq 0 })
    $msg = if ($f.Count -eq 0) { "NO FILES" }
           elseif ($empty.Count) { "EMPTY: $($empty.Count)/$($f.Count)" }
           else { "ok ($($f.Count) files)" }
    Write-Host ("  {0,-14} {1}" -f $c.stem, $msg)
}
Write-Host ("{0}/{1} DONE -- {2:hh\:mm\:ss}" -f $Collection, $System, $sw.Elapsed)
