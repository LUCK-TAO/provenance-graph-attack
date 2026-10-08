param(
    [ValidateSet(
        "check", "smoke", "table2", "ppo_probe", "table3", "core",
        "budget", "ablation", "aggregate", "all"
    )]
    [string]$Stage = "smoke",
    [ValidateSet("cadets", "theia", "trace")]
    [string]$Dataset = "theia",
    [string]$ExperimentRoot = "attack_results\paper_suite_theia_v1",
    [ValidateSet("greedy", "linucb", "double_dqn", "ppo_online", "ppo_frozen")]
    [string]$BudgetPolicy = "ppo_frozen",
    [string[]]$Seeds = @("42", "43", "44"),
    [string]$PythonPath = "D:\App\anaconda\envs\stgan\python.exe"
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
if ($env:OS -eq "Windows_NT") {
    & chcp.com 65001 | Out-Null
}
$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
Set-Location -LiteralPath $ProjectRoot

# ``powershell -File ... -Seeds 43,44`` may deliver "43,44" as one
# culture-formatted numeric token (and previously became seed 4344).  Parse
# seed tokens explicitly so both comma-separated CLI input and native arrays
# produce the intended integer sequence.
$ParsedSeeds = @()
foreach ($SeedToken in $Seeds) {
    foreach ($SeedPart in ([string]$SeedToken -split ",")) {
        $SeedText = $SeedPart.Trim()
        if (-not $SeedText) {
            continue
        }
        $ParsedSeed = 0
        if (-not [int]::TryParse(
            $SeedText,
            [System.Globalization.NumberStyles]::Integer,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$ParsedSeed
        )) {
            throw "Invalid seed value: '$SeedText'"
        }
        $ParsedSeeds += $ParsedSeed
    }
}
if (-not $ParsedSeeds.Count) {
    throw "At least one seed is required."
}
$Seeds = [int[]]$ParsedSeeds

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "STGAN Python not found: $PythonPath"
}

if ([System.IO.Path]::IsPathRooted($ExperimentRoot)) {
    $Root = [System.IO.Path]::GetFullPath($ExperimentRoot)
} else {
    $Root = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $ExperimentRoot))
}
if (-not $Root.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ExperimentRoot must stay inside the project: $Root"
}

$RunRoot = Join-Path $Root "runs"
$SmokeRoot = Join-Path $Root "smoke"
$OutputRoot = Join-Path $Root "paper_outputs"

function Initialize-CleanRoot {
    if (Test-Path -LiteralPath $Root) {
        throw "Clean experiment root already exists: $Root. Choose a new -ExperimentRoot."
    }
    New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $SmokeRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    $manifest = @(
        "created=$(Get-Date -Format o)",
        "python=$PythonPath",
        "seeds=$($Seeds -join ',')",
        "dataset=$Dataset",
        "project=$ProjectRoot"
    )
    Set-Content -LiteralPath (Join-Path $Root "experiment_manifest.txt") -Value $manifest -Encoding UTF8
}

function Assert-RootExists {
    if (-not (Test-Path -LiteralPath $RunRoot -PathType Container)) {
        throw "Experiment root is not initialized. Run -Stage smoke first: $Root"
    }
}

function Assert-RunSummary {
    param(
        [string]$RunName,
        [string]$RequiredStage
    )
    $SummaryPath = Join-Path (Join-Path $RunRoot $RunName) "summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw (
            "Missing prerequisite run: $RunName. " +
            "Run -Stage $RequiredStage for the same seed first."
        )
    }
    return $SummaryPath
}

function Run-Checks {
    Write-Host "`n===== ENVIRONMENT CHECK =====" -ForegroundColor Cyan
    & $PythonPath -c "import torch,dgl,sklearn,matplotlib; print('torch='+torch.__version__); print('dgl='+dgl.__version__); print('cuda='+str(torch.cuda.is_available()))"
    if ($LASTEXITCODE -ne 0) { throw "STGAN dependency check failed" }
    & $PythonPath -B -c "from pathlib import Path; files=list(Path('MAGIC_main').rglob('*.py'))+[Path('ATTACK.py'),Path('Detector_Factory.py')]; [compile(p.read_text(encoding='utf-8-sig'),str(p),'exec') for p in files]; print(f'syntax_ok files={len(files)}')"
    if ($LASTEXITCODE -ne 0) { throw "Python syntax check failed" }
    foreach ($dataset in @("cadets", "theia", "trace")) {
        & $PythonPath -B MAGIC_main\validate_action_space.py --dataset $dataset --test-index 0
        if ($LASTEXITCODE -ne 0) {
            throw "Action-space validation failed for $dataset"
        }
    }
    & $PythonPath -B MAGIC_main\validate_attack_objective.py
    if ($LASTEXITCODE -ne 0) {
        throw "Attack-objective validation failed"
    }
    & $PythonPath -B MAGIC_main\validate_baseline_policies.py
    if ($LASTEXITCODE -ne 0) {
        throw "LinUCB/Double DQN policy validation failed"
    }
    Write-Host "===== CHECKS COMPLETE =====" -ForegroundColor Green
}

function Invoke-Experiment {
    param(
        [string]$Name,
        [string]$OutputDirectory,
        [string]$ActionPolicy,
        [string]$TargetPolicy,
        [int]$TopK,
        [int]$Rounds,
        [int]$Seed,
        [string[]]$Extra = @()
    )
    $RunDirectory = Join-Path $OutputDirectory $Name
    if (Test-Path -LiteralPath $RunDirectory) {
        $ExistingSummary = Join-Path $RunDirectory "summary.json"
        if (Test-Path -LiteralPath $ExistingSummary -PathType Leaf) {
            Write-Host "===== REUSE COMPLETED $Name =====" -ForegroundColor Yellow
            return
        }
        throw "Incomplete run directory already exists; refusing to mix/overwrite data: $RunDirectory"
    }
    $CommandArgs = @(
        "ATTACK.py",
        "--dataset", $Dataset,
        "--detector", "magic",
        "--test-index", "0",
        "--top-k", "$TopK",
        "--max-rounds", "$Rounds",
        "--action-policy", $ActionPolicy,
        "--target-policy", $TargetPolicy,
        "--seed", "$Seed",
        "--deterministic",
        "--output-dir", $OutputDirectory,
        "--run-name", $Name
    ) + $Extra
    Write-Host "`n===== START $Name =====" -ForegroundColor Cyan
    & $PythonPath @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $Name (exit=$LASTEXITCODE)"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RunDirectory "summary.json"))) {
        throw "Experiment produced no summary.json: $Name"
    }
    Write-Host "===== COMPLETE $Name =====" -ForegroundColor Green
}

function Invoke-Aggregation {
    Assert-RootExists
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    & $PythonPath "MAGIC_main\compare_experiments.py" `
        --results-dir $RunRoot `
        --output (Join-Path $OutputRoot "comparison_all.csv")
    if ($LASTEXITCODE -ne 0) { throw "compare_experiments.py failed" }
    & $PythonPath "MAGIC_main\paper_outputs.py" `
        --results-dir $RunRoot `
        --output-dir $OutputRoot `
        --main-top-k 10 `
        --main-rounds 5
    if ($LASTEXITCODE -ne 0) { throw "paper_outputs.py failed" }
    Write-Host "Paper outputs: $OutputRoot" -ForegroundColor Green
}

function Run-Smoke {
    foreach ($Policy in @(
        "no_attack", "random", "greedy", "linucb", "double_dqn",
        "benign_mimicry"
    )) {
        $Extra = @("--reward-mode", "full")
        if ($Policy -eq "double_dqn") {
            # A one-sample batch guarantees an optimizer step even if the
            # first macro action immediately crosses the threshold.
            $Extra += @("--ddqn-batch-size", "1")
        }
        Invoke-Experiment `
            -Name "smoke_${Policy}_margin_k1_r1_seed42" `
            -OutputDirectory $SmokeRoot `
            -ActionPolicy $Policy `
            -TargetPolicy "threshold_margin" `
            -TopK 1 `
            -Rounds 1 `
            -Seed 42 `
            -Extra $Extra
    }

    $PpoSmokeName = "smoke_ppo_online_margin_k1_r1_seed42"
    Invoke-Experiment `
        -Name $PpoSmokeName `
        -OutputDirectory $SmokeRoot `
        -ActionPolicy "ppo_online" `
        -TargetPolicy "threshold_margin" `
        -TopK 1 `
        -Rounds 1 `
        -Seed 42 `
        -Extra @(
            "--clean-repeat-checks", "1",
            "--warmup-updates", "0",
            "--reward-mode", "full",
            "--experiment-variant", "full_model"
        )

    $ExpectedSmokeRuns = @(
        "smoke_no_attack_margin_k1_r1_seed42",
        "smoke_random_margin_k1_r1_seed42",
        "smoke_greedy_margin_k1_r1_seed42",
        "smoke_linucb_margin_k1_r1_seed42",
        "smoke_double_dqn_margin_k1_r1_seed42",
        "smoke_benign_mimicry_margin_k1_r1_seed42",
        $PpoSmokeName
    )
    foreach ($RunName in $ExpectedSmokeRuns) {
        $SummaryPath = Join-Path (Join-Path $SmokeRoot $RunName) "summary.json"
        if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
            throw "Unified smoke missing summary: $RunName"
        }
    }
    $LinUcbSummary = Get-Content -LiteralPath `
        (Join-Path (Join-Path $SmokeRoot "smoke_linucb_margin_k1_r1_seed42") "summary.json") `
        -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($LinUcbSummary.result.linucb_stats.total_updates -lt 1) {
        throw "LinUCB smoke consumed no feedback"
    }
    $DqnSummary = Get-Content -LiteralPath `
        (Join-Path (Join-Path $SmokeRoot "smoke_double_dqn_margin_k1_r1_seed42") "summary.json") `
        -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($DqnSummary.result.double_dqn_stats.total_updates -lt 1) {
        throw "Double DQN smoke performed no optimizer update"
    }
    $PpoSummary = Get-Content -LiteralPath `
        (Join-Path (Join-Path $SmokeRoot $PpoSmokeName) "summary.json") `
        -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($PpoSummary.result.checkpoint_updates_end -le $PpoSummary.result.checkpoint_updates_start) {
        throw "Online PPO smoke performed no optimizer update"
    }
    if ($PpoSummary.result.ppo_selection_mode -ne "sample") {
        throw "Online PPO smoke did not use sample selection"
    }
    Write-Host "===== UNIFIED POLICY SMOKE VERIFIED =====" -ForegroundColor Green
}

function Run-Table2 {
    Assert-RootExists
    foreach ($Seed in $Seeds) {
        # Table 2: target selection under the same random action policy.
        # Rotate execution order by seed to reduce time-of-day, thermal and
        # background-load bias in the runtime comparison:
        #   seed 42: margin -> highest -> random
        #   seed 43: highest -> random -> margin
        #   seed 44: random -> margin -> highest
        $Targets = @("threshold_margin", "highest_score", "random_alert")
        $TargetOffset = (($Seed % $Targets.Count) + $Targets.Count) % $Targets.Count
        $OrderedTargets = @(
            for ($Index = 0; $Index -lt $Targets.Count; $Index++) {
                $Targets[($Index + $TargetOffset) % $Targets.Count]
            }
        )
        Write-Host (
            "Table 2 seed=$Seed execution order: " +
            ($OrderedTargets -join " -> ")
        ) -ForegroundColor DarkCyan
        foreach ($Target in $OrderedTargets) {
            Invoke-Experiment `
                -Name "t2_random_${Target}_k10_r5_seed${Seed}" `
                -OutputDirectory $RunRoot `
                -ActionPolicy "random" `
                -TargetPolicy $Target `
                -TopK 10 `
                -Rounds 5 `
                -Seed $Seed `
                -Extra @("--experiment-variant", "no_ppo")
        }
        foreach ($Target in @("threshold_margin", "highest_score", "random_alert")) {
            Assert-RunSummary `
                -RunName "t2_random_${Target}_k10_r5_seed${Seed}" `
                -RequiredStage "table2" | Out-Null
        }
    }
    Write-Host "===== TABLE 2 RUNS VERIFIED =====" -ForegroundColor Green
}

function Run-PpoProbe {
    Assert-RootExists
    foreach ($Seed in $Seeds) {
        # The Random+Margin row from Table 2 is a prerequisite and will later
        # be reused as the Random row of Table 3.
        Assert-RunSummary `
            -RunName "t2_random_threshold_margin_k10_r5_seed${Seed}" `
            -RequiredStage "table2" | Out-Null

        # This is a full K=10, R=5 paper run, not a throw-away diagnostic.
        # Table 3 reuses it without rerunning PPO.
        $TrainName = "t3_ppo_online_margin_k10_r5_seed${Seed}"
        Invoke-Experiment `
            -Name $TrainName `
            -OutputDirectory $RunRoot `
            -ActionPolicy "ppo_online" `
            -TargetPolicy "threshold_margin" `
            -TopK 10 `
            -Rounds 5 `
            -Seed $Seed `
            -Extra @(
                # Macro PPO starts learning after the first 8-transition
                # rollout. Warmup is disabled so the learned policy is used
                # immediately after the first update.
                "--warmup-updates", "0",
                "--reward-mode", "full",
                "--experiment-variant", "full_model"
            )

        $Checkpoint = Join-Path (Join-Path $RunRoot $TrainName) "ppo_checkpoint.pth"
        if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
            throw "Online PPO produced no checkpoint: $Checkpoint"
        }
        $OnlineSummaryPath = Assert-RunSummary `
            -RunName $TrainName -RequiredStage "ppo_probe"
        $OnlineSummary = Get-Content -LiteralPath $OnlineSummaryPath `
            -Raw -Encoding UTF8 | ConvertFrom-Json
        if (
            $OnlineSummary.result.checkpoint_updates_end `
                -le $OnlineSummary.result.checkpoint_updates_start
        ) {
            throw "Online PPO formal probe performed no optimizer update: seed=$Seed"
        }
        if ($OnlineSummary.result.ppo_selection_mode -ne "sample") {
            throw "Online PPO formal probe did not use sample selection: seed=$Seed"
        }
    }
    Write-Host "===== FORMAL PPO PROBE VERIFIED =====" -ForegroundColor Green
}

function Run-Table3 {
    Assert-RootExists
    foreach ($Seed in $Seeds) {
        # Table 3 reuses Random+Margin from Table 2 and Online PPO from
        # ppo_probe. Only the five remaining baseline rows run here.
        Assert-RunSummary `
            -RunName "t2_random_threshold_margin_k10_r5_seed${Seed}" `
            -RequiredStage "table2" | Out-Null
        Assert-RunSummary `
            -RunName "t3_ppo_online_margin_k10_r5_seed${Seed}" `
            -RequiredStage "ppo_probe" | Out-Null
        Invoke-Experiment `
            -Name "t3_no_attack_margin_k10_r5_seed${Seed}" `
            -OutputDirectory $RunRoot `
            -ActionPolicy "no_attack" `
            -TargetPolicy "threshold_margin" `
            -TopK 10 `
            -Rounds 5 `
            -Seed $Seed

        Invoke-Experiment `
            -Name "t3_greedy_margin_k10_r5_seed${Seed}" `
            -OutputDirectory $RunRoot `
            -ActionPolicy "greedy" `
            -TargetPolicy "threshold_margin" `
            -TopK 10 `
            -Rounds 5 `
            -Seed $Seed

        Invoke-Experiment `
            -Name "t3_linucb_margin_k10_r5_seed${Seed}" `
            -OutputDirectory $RunRoot `
            -ActionPolicy "linucb" `
            -TargetPolicy "threshold_margin" `
            -TopK 10 `
            -Rounds 5 `
            -Seed $Seed `
            -Extra @("--experiment-variant", "linucb_baseline")

        Invoke-Experiment `
            -Name "t3_double_dqn_margin_k10_r5_seed${Seed}" `
            -OutputDirectory $RunRoot `
            -ActionPolicy "double_dqn" `
            -TargetPolicy "threshold_margin" `
            -TopK 10 `
            -Rounds 5 `
            -Seed $Seed `
            -Extra @(
                "--ddqn-batch-size", "16",
                "--experiment-variant", "double_dqn_baseline"
            )

        Invoke-Experiment `
            -Name "t3_benign_mimicry_margin_k10_r5_seed${Seed}" `
            -OutputDirectory $RunRoot `
            -ActionPolicy "benign_mimicry" `
            -TargetPolicy "threshold_margin" `
            -TopK 10 `
            -Rounds 5 `
            -Seed $Seed `
            -Extra @(
                "--experiment-variant", "benign_mimicry_baseline"
            )

        foreach ($RunName in @(
            "t2_random_threshold_margin_k10_r5_seed${Seed}",
            "t3_no_attack_margin_k10_r5_seed${Seed}",
            "t3_greedy_margin_k10_r5_seed${Seed}",
            "t3_linucb_margin_k10_r5_seed${Seed}",
            "t3_double_dqn_margin_k10_r5_seed${Seed}",
            "t3_benign_mimicry_margin_k10_r5_seed${Seed}",
            "t3_ppo_online_margin_k10_r5_seed${Seed}"
        )) {
            Assert-RunSummary -RunName $RunName -RequiredStage "table3" | Out-Null
        }
    }
    Write-Host "===== TABLE 3 RUNS VERIFIED =====" -ForegroundColor Green
}

function Run-Core {
    Assert-RootExists
    Run-Table2
    Run-PpoProbe
    Run-Table3
    Invoke-Aggregation
}

function Run-Budget {
    Assert-RootExists
    foreach ($Seed in $Seeds) {
        $PolicyExtra = @()
        if ($BudgetPolicy -eq "ppo_frozen") {
            $Checkpoint = Join-Path `
                (Join-Path $RunRoot "t3_ppo_online_margin_k10_r5_seed${Seed}") `
                "ppo_checkpoint.pth"
            if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
                throw (
                    "Missing full-model checkpoint; run -Stage ppo_probe " +
                    "for the same seed first: $Checkpoint"
                )
            }
            $PolicyExtra = @(
                "--load-checkpoint", "--checkpoint", $Checkpoint,
                "--experiment-variant", "full_model"
            )
        } elseif ($BudgetPolicy -eq "ppo_online") {
            $PolicyExtra = @(
                "--warmup-updates", "0",
                "--experiment-variant", "full_model"
            )
        } elseif ($BudgetPolicy -eq "linucb") {
            $PolicyExtra = @("--experiment-variant", "linucb_baseline")
        } elseif ($BudgetPolicy -eq "double_dqn") {
            $PolicyExtra = @(
                "--ddqn-batch-size", "16",
                "--experiment-variant", "double_dqn_baseline"
            )
        }

        # Table 4-1: fixed 5 rounds, K = 1,5,10,20. K=10 is reused from core.
        foreach ($K in @(1, 5, 20)) {
            Invoke-Experiment -Name "t41_random_margin_k${K}_r5_seed${Seed}" -OutputDirectory $RunRoot -ActionPolicy "random" -TargetPolicy "threshold_margin" -TopK $K -Rounds 5 -Seed $Seed -Extra @("--experiment-variant", "no_ppo")
            Invoke-Experiment -Name "t41_${BudgetPolicy}_margin_k${K}_r5_seed${Seed}" -OutputDirectory $RunRoot -ActionPolicy $BudgetPolicy -TargetPolicy "threshold_margin" -TopK $K -Rounds 5 -Seed $Seed -Extra $PolicyExtra
        }

        # Table 4-2: fixed 10 rounds, K = 1,3,5,10.
        foreach ($K in @(1, 3, 5, 10)) {
            Invoke-Experiment -Name "t42_random_margin_k${K}_r10_seed${Seed}" -OutputDirectory $RunRoot -ActionPolicy "random" -TargetPolicy "threshold_margin" -TopK $K -Rounds 10 -Seed $Seed -Extra @("--experiment-variant", "no_ppo")
            Invoke-Experiment -Name "t42_${BudgetPolicy}_margin_k${K}_r10_seed${Seed}" -OutputDirectory $RunRoot -ActionPolicy $BudgetPolicy -TargetPolicy "threshold_margin" -TopK $K -Rounds 10 -Seed $Seed -Extra $PolicyExtra
        }
    }
    Invoke-Aggregation
}

function Run-Ablation {
    Assert-RootExists
    foreach ($Seed in $Seeds) {
        # Full LinUCB is reused from Table 3; never spend time rerunning it.
        Assert-RunSummary `
            -RunName "t3_linucb_margin_k10_r5_seed${Seed}" `
            -RequiredStage "table3" | Out-Null

        # The five removals target components that LinUCB actually consumes.
        # The no-macro run is intentionally last because querying after every
        # edge is substantially more expensive than the other four variants.
        $Variants = @(
            @{
                Name = "no_context"
                Label = "linucb_no_context"
                Extra = @("--linucb-context-mode", "constant")
            },
            @{
                Name = "no_exploration"
                Label = "linucb_no_exploration"
                Extra = @("--linucb-alpha", "0")
            },
            @{
                Name = "no_candidate_selector"
                Label = "linucb_no_candidate_selector"
                Extra = @("--candidate-policy", "random_legal")
            },
            @{
                Name = "no_reward_feedback"
                Label = "linucb_no_reward_feedback"
                Extra = @("--reward-mode", "none")
            },
            @{
                Name = "no_macro_action"
                Label = "linucb_no_macro_action"
                Extra = @("--macro-action-length", "1")
            }
        )

        foreach ($Variant in $Variants) {
            $RunName = "t5_linucb_$($Variant.Name)_margin_k10_r5_seed${Seed}"
            $Extra = @(
                "--experiment-variant", $Variant.Label,
                "--no-generate-plots"
            ) + $Variant.Extra
            Invoke-Experiment `
                -Name $RunName `
                -OutputDirectory $RunRoot `
                -ActionPolicy "linucb" `
                -TargetPolicy "threshold_margin" `
                -TopK 10 `
                -Rounds 5 `
                -Seed $Seed `
                -Extra $Extra

            $SummaryPath = Assert-RunSummary `
                -RunName $RunName -RequiredStage "ablation"
            $Summary = Get-Content -LiteralPath $SummaryPath `
                -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($Summary.result.action_policy -ne "linucb") {
                throw "Ablation did not use LinUCB: $RunName"
            }
            switch ($Variant.Name) {
                "no_context" {
                    if ($Summary.result.linucb_stats.context_mode -ne "constant") {
                        throw "No-context ablation leaked full context: $RunName"
                    }
                }
                "no_exploration" {
                    if ([double]$Summary.result.linucb_stats.alpha -ne 0.0) {
                        throw "No-exploration ablation retained alpha: $RunName"
                    }
                }
                "no_candidate_selector" {
                    if ($Summary.result.candidate_policy -ne "random_legal") {
                        throw "Candidate-selector ablation was not random legal: $RunName"
                    }
                }
                "no_reward_feedback" {
                    $Reward = $Summary.result.reward_config
                    $RewardTotal = (
                        [double]$Reward.score_weight +
                        [double]$Reward.crossing_weight +
                        [double]$Reward.indirect_weight +
                        [double]$Reward.perturbation_weight
                    )
                    if ($RewardTotal -ne 0.0) {
                        throw "No-reward ablation retained reward weights: $RunName"
                    }
                }
                "no_macro_action" {
                    if ([int]$Summary.result.macro_action_length -ne 1) {
                        throw "No-macro ablation retained macro batching: $RunName"
                    }
                }
            }
        }

        foreach ($Variant in $Variants) {
            Assert-RunSummary `
                -RunName "t5_linucb_$($Variant.Name)_margin_k10_r5_seed${Seed}" `
                -RequiredStage "ablation" | Out-Null
        }
    }
    Write-Host "===== LINUCB ABLATION RUNS VERIFIED =====" -ForegroundColor Green
    # Aggregate only after every requested seed and every ablation variant has
    # a verified summary.  This prevents a partial seed set from being mistaken
    # for the final mean +/- standard-deviation table.
    Invoke-Aggregation
}

if ($Stage -in @("check", "all")) { Run-Checks }
if ($Stage -in @("smoke", "all")) {
    Initialize-CleanRoot
    Run-Smoke
}
if ($Stage -eq "table2") {
    Run-Table2
    Invoke-Aggregation
}
if ($Stage -eq "ppo_probe") {
    Run-PpoProbe
    Invoke-Aggregation
}
if ($Stage -eq "table3") {
    Run-Table3
    Invoke-Aggregation
}
if ($Stage -eq "core" -or $Stage -eq "all") { Run-Core }
if ($Stage -eq "budget" -or $Stage -eq "all") { Run-Budget }
if ($Stage -eq "ablation" -or $Stage -eq "all") { Run-Ablation }
if ($Stage -eq "aggregate") { Invoke-Aggregation }
