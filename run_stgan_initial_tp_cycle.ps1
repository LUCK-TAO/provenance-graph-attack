param(
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Device = "cuda:0",
    [int]$MaxRounds = 5,
    [int]$TargetsPerRound = 10,
    [ValidateSet("constant", "full")]
    [string]$LinUCBContextMode = "constant",
    [string]$OutputSubdirectory = ""
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "D:\App\anaconda\Scripts\conda.exe"
$Attack = Join-Path $Workspace "ATTACK.py"
$State = Join-Path $Workspace "STGAN_master\data\cadets\cadets_stgan_official_attack_test0.npz"
if ([string]::IsNullOrWhiteSpace($OutputSubdirectory)) {
    $OutputSubdirectory = if ($LinUCBContextMode -eq "full") {
        "stgan_cadets_initial_tp_cycle_state"
    }
    else {
        "stgan_initial_tp_cycle"
    }
}
$Output = Join-Path $Workspace ("attack_results\{0}" -f $OutputSubdirectory)

foreach ($Required in @($Conda, $Attack, $State)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file is missing: $Required"
    }
}

# Stream UTF-8 directly from the STGAN environment. --no-capture-output also
# avoids the Windows GBK error that conda can raise after a successful run.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

foreach ($Seed in $Seeds) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $ContextLabel = if ($LinUCBContextMode -eq "full") { "state" } else { "constant" }
    $RunName = "stgan_cadets_linucb_${ContextLabel}_initial_tp_cycle_k${TargetsPerRound}_r${MaxRounds}_seed${Seed}_${Timestamp}"
    $Arguments = @(
        "--dataset", "cadets",
        "--detector", "stgan",
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
        "--linucb-context-mode", $LinUCBContextMode,
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
        throw "STGAN attack failed with exit code $LASTEXITCODE"
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
