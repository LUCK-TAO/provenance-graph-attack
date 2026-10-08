param(
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Device = "cuda:0",
    [int]$MaxRounds = 5,
    [int]$TargetsPerRound = 10,
    [int]$MacroActionLength = 5,
    [int]$CooldownSeconds = 30,
    [int]$MinimumAvailableMemoryMB = 12000,
    [string]$OutputDir = "attack_results\orthrus_theia_native_full_initial_tp"
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "D:\App\anaconda\Scripts\conda.exe"
$Attack = Join-Path $Workspace "ATTACK.py"
$State = Join-Path $Workspace "Orthrus_official\artifacts\attack_state\theia_e3_native_full_tp.npz"
$Output = Join-Path $Workspace $OutputDir
$NativeFullThreshold = 6.4582357

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

$LockPath = Join-Path $Output "orthrus_theia_three_seed.lock"
try {
    $LockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
} catch {
    throw "Another Orthrus THEIA three-seed script is already running: $LockPath"
}

function Convert-SummaryToRow {
    param([int]$Seed, [string]$RunName, [string]$SummaryPath)
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $Result = $Summary.result
    return [pscustomobject]@{
        seed = $Seed
        run_name = $RunName
        fixed_initial_tp_pool = $Result.initial_native_tp
        attempted_initial_tp = $Result.attempted_initial_tp_count
        directly_escaped_attempted_tp = $Result.direct_native_tp_escaped
        indirectly_escaped_initial_tp = $Result.indirect_native_tp_escaped
        all_escaped_initial_tp = $Result.tp_reduction_count
        ASR = $Result.direct_tp_asr
        initial_tp_evasion_rate = $Result.tp_reduction_rate
        evasion_amplification = $Result.evasion_amplification
        AAB = $Result.average_attack_budget
        graph_modification_rate = $Result.graph_modification_rate
        total_detector_queries = $Result.total_detector_queries
        AQT_seconds = $Result.average_query_time_seconds
        summary_path = $SummaryPath
    }
}

function Wait-ForAvailableMemory {
    for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {
        $Sample = Get-Counter '\Memory\Available MBytes' -ErrorAction Stop
        $Available = [int]$Sample.CounterSamples[0].CookedValue
        if ($Available -ge $MinimumAvailableMemoryMB) {
            Write-Host "Available memory before next seed: ${Available} MB"
            return
        }
        Write-Warning (
            "Only ${Available} MB memory is available; waiting 30 seconds " +
            "for the previous Orthrus process to release memory."
        )
        Start-Sleep -Seconds 30
    }
    throw "Available memory stayed below ${MinimumAvailableMemoryMB} MB. Stop other Python processes and rerun."
}

$Rows = @()
try {
    foreach ($Seed in $Seeds) {
        $Prefix = "orthrus_theia_full_initial10_k${TargetsPerRound}_r${MaxRounds}_seed${Seed}_"
        $CompletedRun = Get-ChildItem -LiteralPath $Output -Directory |
            Where-Object { $_.Name.StartsWith($Prefix) } |
            Sort-Object LastWriteTime -Descending |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "summary.json") } |
            Select-Object -First 1

        if ($null -ne $CompletedRun) {
            $SummaryPath = Join-Path $CompletedRun.FullName "summary.json"
            Write-Host "Seed $Seed already completed; reusing $SummaryPath"
            $Rows += Convert-SummaryToRow `
                -Seed $Seed -RunName $CompletedRun.Name -SummaryPath $SummaryPath
            $Rows | Sort-Object seed | Export-Csv `
                -LiteralPath (Join-Path $Output "orthrus_theia_three_seed_summary.csv") `
                -NoTypeInformation -Encoding UTF8
            continue
        }

        Wait-ForAvailableMemory
        $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $RunName = "${Prefix}${Timestamp}"
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
            "--output-dir", $Output,
            "--run-name", $RunName,
            "--no-generate-plots"
        )

        Write-Host "Starting $RunName"
        & $Conda run --no-capture-output -n STGAN python $Attack @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Orthrus THEIA seed $Seed failed with exit code $LASTEXITCODE"
        }

        $SummaryPath = Join-Path (Join-Path $Output $RunName) "summary.json"
        if (-not (Test-Path -LiteralPath $SummaryPath)) {
            throw "Run completed without summary.json: $SummaryPath"
        }
        $Rows += Convert-SummaryToRow `
            -Seed $Seed -RunName $RunName -SummaryPath $SummaryPath
        $Rows | Sort-Object seed | Export-Csv `
            -LiteralPath (Join-Path $Output "orthrus_theia_three_seed_summary.csv") `
            -NoTypeInformation -Encoding UTF8
        $Result = (Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json).result
        Write-Host ("Completed seed {0}: ASR={1}, AAB={2}, AQT={3}s" -f `
            $Seed, $Result.direct_tp_asr, $Result.average_attack_budget, `
            $Result.average_query_time_seconds)

        if ($CooldownSeconds -gt 0) {
            Write-Host "Cooling down for $CooldownSeconds seconds before the next seed..."
            Start-Sleep -Seconds $CooldownSeconds
        }
    }
} finally {
    $LockStream.Dispose()
}

$Aggregate = Join-Path $Output "orthrus_theia_three_seed_summary.csv"
Write-Host "Three-seed summary: $Aggregate"
