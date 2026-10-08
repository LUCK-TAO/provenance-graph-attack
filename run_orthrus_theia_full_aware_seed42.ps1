param(
    [int]$Seed = 42,
    [string]$Device = "cuda:0",
    [int]$MaxRounds = 5,
    [int]$TargetsPerRound = 10,
    [int]$MacroActionLength = 5,
    [int]$CandidateProbeSize = 20,
    [string]$OutputDir = "attack_results\orthrus_theia_full_aware"
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "D:\App\anaconda\Scripts\conda.exe"
$Attack = Join-Path $Workspace "ATTACK.py"
$State = Join-Path $Workspace "Orthrus_official\artifacts\attack_state\theia_e3_native_full_tp.npz"
$Output = Join-Path $Workspace $OutputDir
$PolicyCheckpoint = Join-Path $Output "linucb_full_aware_seed${Seed}.pth"
$NativeFullThreshold = 6.4582357

foreach ($Required in @($Conda, $Attack, $State)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file is missing: $Required"
    }
}
if ($CandidateProbeSize -lt $MacroActionLength) {
    throw "CandidateProbeSize must be at least MacroActionLength."
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:NLTK_DATA = Join-Path $Workspace "Orthrus_official\nltk_data"
New-Item -ItemType Directory -Force -Path $Output | Out-Null
New-Item -ItemType Directory -Force -Path $env:NLTK_DATA | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunName = (
    "orthrus_theia_full_aware_k{0}_r{1}_seed{2}_{3}" -f `
    $TargetsPerRound, $MaxRounds, $Seed, $Timestamp
)
$Arguments = @(
    "--dataset", "theia",
    "--detector", "orthrus",
    "--test-index", "0",
    "--device", $Device,
    "--top-k", "$TargetsPerRound",
    "--max-rounds", "$MaxRounds",
    "--action-policy", "linucb",
    "--target-policy", "initial_tp_cycle",
    "--seed", "$Seed",
    "--candidate-policy", "semantic_mixed",
    "--reward-mode", "full",
    "--linucb-alpha", "1.0",
    "--linucb-context-mode", "full",
    "--macro-action-length", "$MacroActionLength",
    "--threshold-source", "fixed",
    "--threshold", "$NativeFullThreshold",
    "--deterministic",
    "--orthrus-attack-state", $State,
    "--orthrus-full-aware",
    "--orthrus-candidate-probe-size", "$CandidateProbeSize",
    "--experiment-variant", "orthrus_full_aware",
    "--checkpoint", $PolicyCheckpoint,
    "--output-dir", $Output,
    "--run-name", $RunName,
    "--no-generate-plots"
)

Write-Host "Starting Orthrus THEIA full-aware seed $Seed"
Write-Host "Run directory: $(Join-Path $Output $RunName)"
& $Conda run --no-capture-output -n STGAN python $Attack @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Orthrus THEIA full-aware seed $Seed failed with exit code $LASTEXITCODE"
}

$SummaryPath = Join-Path (Join-Path $Output $RunName) "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath)) {
    throw "Run completed without summary.json: $SummaryPath"
}
$Result = (Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json).result
Write-Host ("Completed seed {0}: ASR={1}, AAB={2}, AQT={3}s" -f `
    $Seed, $Result.direct_tp_asr, $Result.average_attack_budget, `
    $Result.average_query_time_seconds)
Write-Host "Full-aware statistics:"
$Result.orthrus_full_aware_stats | Format-List
Write-Host "Summary: $SummaryPath"
