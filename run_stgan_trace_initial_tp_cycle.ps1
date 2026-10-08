param(
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Device = "cuda:0",
    [int]$MaxRounds = 5,
    [int]$TargetsPerRound = 10,
    [switch]$SkipPrepare,
    [switch]$PreflightOnly
)

$Arguments = @{
    Seeds = $Seeds
    Device = $Device
    MaxRounds = $MaxRounds
    TargetsPerRound = $TargetsPerRound
    TargetPolicy = "initial_tp_cycle"
    SkipPrepare = $SkipPrepare
    PreflightOnly = $PreflightOnly
}
& (Join-Path $PSScriptRoot "run_stgan_trace_initial_tp_random.ps1") @Arguments
exit $LASTEXITCODE
