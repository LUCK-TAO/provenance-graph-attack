param(
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Device = "cuda:0",
    [int]$MaxRounds = 5,
    [int]$TargetsPerRound = 10,
    [ValidateSet("initial_tp_cycle", "initial_tp_random")]
    [string]$TargetPolicy = "initial_tp_random",
    [switch]$SkipPrepare,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "D:\App\anaconda\Scripts\conda.exe"
$Attack = Join-Path $Workspace "ATTACK.py"
$Prepare = Join-Path $Workspace "STGAN_master\stgan_trace.py"
$DataDir = Join-Path $Workspace "STGAN_master\data\trace"
$Archive = Join-Path $DataDir "ta1-trace-e3-official-1.json.tar.gz"
$GroundTruth = Join-Path $DataDir "trace.json"
$State = Join-Path $DataDir "trace_stgan_native_test4.npz"
$UuidMap = Join-Path $Workspace "MAGIC_main\data\trace\uuid_to_index_test4.json"
$CompatW2V = Join-Path $Workspace "STGAN_master\train\word2vec_trace_compat.npz"
$NativeW2V = Join-Path $Workspace "STGAN_master\train\word2vec_trace.model"
$FallbackW2V = Join-Path $Workspace "Flash_IDS_main\trained_weights\trace\word2vec_trace_E3.model"
$FlashState = Join-Path $Workspace "Flash_IDS_main\data_files\trace_flash_native_test4.npz"
$Graph = Join-Path $Workspace "MAGIC_main\data\trace\test4.pkl"
$Weights = Join-Path $Workspace "STGAN_master\trained_weights\trace"
$Output = Join-Path $Workspace "attack_results\stgan_trace_$TargetPolicy"

foreach ($Required in @(
    $Conda, $Attack, $Prepare, $Archive, $GroundTruth, $Graph,
    $Weights, $FallbackW2V, $FlashState
)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required TRACE asset is missing: $Required"
    }
}
if (-not (Test-Path -LiteralPath $NativeW2V)) {
    Write-Warning (
        "The STGAN release does not contain word2vec_trace.model. " +
        "Preparation will use the available pretrained TRACE E3 compatibility " +
        "vectors and record this provenance in the audit report."
    )
}

$ExpectedSnapshots = 50
foreach ($Prefix in @("gat", "tgn", "selfattention")) {
    $Count = @(Get-ChildItem -LiteralPath $Weights -File -Filter "${Prefix}*.pth").Count
    if ($Count -ne $ExpectedSnapshots) {
        throw "TRACE STGAN requires $ExpectedSnapshots $Prefix checkpoints; found $Count"
    }
}

$StateReady = (
    (Test-Path -LiteralPath $State) -and
    (Test-Path -LiteralPath $UuidMap) -and
    (Test-Path -LiteralPath $CompatW2V)
)

Write-Host "STGAN TRACE preflight passed."
Write-Host "Native split: test4; 12 node classes; 50 frozen snapshots."
if ($TargetPolicy -eq "initial_tp_random") {
    Write-Host "Target sampling: seeded shuffle without replacement from fixed initial direct TPs."
} else {
    Write-Host "Target sampling: cyclic coverage of every active initial direct TP."
}
Write-Host "Archive remains compressed: $Archive"
Write-Host "Native state ready: $StateReady"
if ($PreflightOnly) {
    if (-not $StateReady) {
        Write-Host "The first normal run will stream the compressed test member and build the state."
    }
    exit 0
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (-not $StateReady) {
    if ($SkipPrepare) {
        throw "TRACE STGAN state is incomplete and -SkipPrepare was specified."
    }
    Write-Host "Preparing STGAN TRACE native state from the compressed archive..."
    & $Conda run --no-capture-output -n STGAN python $Prepare prepare
    if ($LASTEXITCODE -ne 0) {
        throw "STGAN TRACE state preparation failed with exit code $LASTEXITCODE"
    }
}

foreach ($Required in @($State, $UuidMap, $CompatW2V)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "State preparation did not create required file: $Required"
    }
}

foreach ($Seed in $Seeds) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $RunName = "stgan_trace_linucb_${TargetPolicy}_k${TargetsPerRound}_r${MaxRounds}_seed${Seed}_${Timestamp}"
    $Arguments = @(
        "--dataset", "trace",
        "--detector", "stgan",
        "--test-index", "4",
        "--device", $Device,
        "--top-k", "$TargetsPerRound",
        "--max-rounds", "$MaxRounds",
        "--action-policy", "linucb",
        "--target-policy", $TargetPolicy,
        "--seed", "$Seed",
        "--candidate-policy", "semantic_mixed",
        "--reward-mode", "full",
        "--linucb-alpha", "1.0",
        "--linucb-context-mode", "constant",
        "--macro-action-length", "5",
        "--threshold-source", "native",
        "--deterministic",
        "--stgan-semantic-state", $State,
        "--output-dir", $Output,
        "--run-name", $RunName
    )

    Write-Host "Starting $RunName"
    & $Conda run --no-capture-output -n STGAN python $Attack @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "STGAN TRACE attack failed with exit code $LASTEXITCODE"
    }

    $SummaryPath = Join-Path (Join-Path $Output $RunName) "summary.json"
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $Result = $Summary.result
    Write-Host "Completed $RunName"
    Write-Host ("ASR = {0}/{1} = {2}" -f `
        $Result.targeted_asr_numerator, `
        $Result.targeted_asr_denominator, `
        $Result.targeted_asr)
    Write-Host ("AAB = {0} primitive operations per escaped initial TP" -f `
        $Result.targeted_aab_primitive_operations)
    Write-Host ("AQT = {0} seconds/query" -f `
        $Result.average_query_time_seconds)
    Write-Host "Summary: $SummaryPath"
}
