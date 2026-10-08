param(
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Device = "cuda:0",
    [int]$MaxRounds = 5,
    [int]$TargetsPerRound = 8,
    [int]$MacroActionLength = 5,
    [string]$OutputDir = "attack_results\orthrus_cadets_initial_tp"
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "D:\App\anaconda\Scripts\conda.exe"
$Attack = Join-Path $Workspace "ATTACK.py"
$State = Join-Path $Workspace "Orthrus_official\artifacts\attack_state\cadets_e3_tw228_229.npz"
$Output = Join-Path $Workspace $OutputDir

foreach ($Required in @($Conda, $Attack, $State)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file is missing: $Required"
    }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:NLTK_DATA = Join-Path $Workspace "Orthrus_official\nltk_data"
New-Item -ItemType Directory -Force -Path $Output | Out-Null
New-Item -ItemType Directory -Force -Path $env:NLTK_DATA | Out-Null

$Rows = @()
foreach ($Seed in $Seeds) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $RunName = "orthrus_cadets_linucb_initial8_k${TargetsPerRound}_r${MaxRounds}_seed${Seed}_${Timestamp}"
    $Arguments = @(
        "--dataset", "cadets",
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
        "--threshold-source", "native",
        "--deterministic",
        "--orthrus-attack-state", $State,
        "--output-dir", $Output,
        "--run-name", $RunName,
        "--no-generate-plots"
    )

    Write-Host "Starting $RunName"
    & $Conda run --no-capture-output -n STGAN python $Attack @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Orthrus seed $Seed failed with exit code $LASTEXITCODE"
    }

    $SummaryPath = Join-Path (Join-Path $Output $RunName) "summary.json"
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $Result = $Summary.result
    $Rows += [pscustomobject]@{
        seed = $Seed
        run_name = $RunName
        fixed_initial_tp_pool = $Result.initial_native_tp
        attempted_initial_tp = $Result.attempted_initial_tp_count
        directly_escaped_attempted_tp = $Result.direct_native_tp_escaped
        indirectly_escaped_initial_tp = $Result.indirect_native_tp_escaped
        all_escaped_initial_tp = $Result.tp_reduction_count
        direct_tp_asr = $Result.direct_tp_asr
        initial_tp_evasion_rate = $Result.tp_reduction_rate
        evasion_amplification = $Result.evasion_amplification
        average_attack_budget = $Result.average_attack_budget
        graph_modification_rate = $Result.graph_modification_rate
        total_detector_queries = $Result.total_detector_queries
        average_query_time_seconds = $Result.average_query_time_seconds
        summary_path = $SummaryPath
    }
    Write-Host ("Completed seed {0}: DirectTP-ASR={1}/{2}={3}" -f `
        $Seed, $Result.direct_native_tp_escaped, `
        $Result.attacked_initial_tp_count, $Result.direct_tp_asr)
}

$Aggregate = Join-Path $Output "orthrus_three_seed_summary.csv"
$Rows | Export-Csv -LiteralPath $Aggregate -NoTypeInformation -Encoding UTF8
Write-Host "Three-seed summary: $Aggregate"
