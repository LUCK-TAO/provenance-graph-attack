param(
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Device = "cuda:0",
    [int]$MaxRounds = 5,
    [int]$TargetsPerRound = 10,
    [switch]$SkipPrepare,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "D:\App\anaconda\Scripts\conda.exe"
$Attack = Join-Path $Workspace "ATTACK.py"
$Prepare = Join-Path $Workspace "STGAN_master\stgan_theia.py"
$DataDir = Join-Path $Workspace "STGAN_master\data\theia"
$Archive = Join-Path $DataDir "ta1-theia-e3-official-6r.json.tar.gz"
$GroundTruth = Join-Path $DataDir "theia.json"
$State = Join-Path $DataDir "theia_stgan_native_test0.npz"
$UuidMap = Join-Path $Workspace "MAGIC_main\data\theia\uuid_to_index_test0.json"
$CompatW2V = Join-Path $Workspace "STGAN_master\train\word2vec_theia_compat.npz"
$NativeW2V = Join-Path $Workspace "STGAN_master\train\word2vec_theia.model"
$FallbackW2V = Join-Path $Workspace "Flash_IDS_main\trained_weights\theia\word2vec_theia_E3.model"
$FlashState = Join-Path $Workspace "Flash_IDS_main\data_files\theia_flash_native_test0.npz"
$Graph = Join-Path $Workspace "MAGIC_main\data\theia\test0.pkl"
$Weights = Join-Path $Workspace "STGAN_master\trained_weights\theia"
$Output = Join-Path $Workspace "attack_results\stgan_theia_initial_tp_margin"

foreach ($Required in @($Conda, $Attack, $Prepare, $Archive, $GroundTruth, $Graph, $Weights, $FlashState)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required THEIA asset is missing: $Required"
    }
}
if (-not (Test-Path -LiteralPath $NativeW2V) -and
    -not (Test-Path -LiteralPath $FallbackW2V)) {
    throw "No THEIA Word2Vec model found. Expected $NativeW2V or $FallbackW2V"
}
if (-not (Test-Path -LiteralPath $NativeW2V)) {
    Write-Warning (
        "The STGAN release does not contain word2vec_theia.model. " +
        "Preparation will use the available pretrained THEIA E3 compatibility " +
        "vectors and record this provenance in the audit report."
    )
}

$ExpectedSnapshots = 20
foreach ($Prefix in @("gat", "tgn", "selfattention")) {
    $Count = @(Get-ChildItem -LiteralPath $Weights -File -Filter "${Prefix}*.pth").Count
    if ($Count -ne $ExpectedSnapshots) {
        throw "THEIA STGAN requires $ExpectedSnapshots $Prefix checkpoints; found $Count"
    }
}

$StateReady = (
    (Test-Path -LiteralPath $State) -and
    (Test-Path -LiteralPath $UuidMap) -and
    (Test-Path -LiteralPath $CompatW2V)
)

Write-Host "STGAN THEIA preflight passed."
Write-Host "Archive remains compressed: $Archive"
Write-Host "Weights: 20 GAT + 20 TGN + 20 self-attention snapshots"
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
        throw "THEIA STGAN state is incomplete and -SkipPrepare was specified."
    }
    Write-Host "Preparing STGAN THEIA native state from the compressed archive..."
    & $Conda run --no-capture-output -n STGAN python $Prepare prepare
    if ($LASTEXITCODE -ne 0) {
        throw "STGAN THEIA state preparation failed with exit code $LASTEXITCODE"
    }
}

foreach ($Required in @($State, $UuidMap, $CompatW2V)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "State preparation did not create required file: $Required"
    }
}

foreach ($Seed in $Seeds) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $RunName = "stgan_theia_linucb_initial_tp_margin_k${TargetsPerRound}_r${MaxRounds}_seed${Seed}_${Timestamp}"
    $Arguments = @(
        "--dataset", "theia",
        "--detector", "stgan",
        "--test-index", "0",
        "--device", $Device,
        "--top-k", "$TargetsPerRound",
        "--max-rounds", "$MaxRounds",
        "--action-policy", "linucb",
        "--target-policy", "initial_tp_margin",
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
        throw "STGAN THEIA attack failed with exit code $LASTEXITCODE"
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
