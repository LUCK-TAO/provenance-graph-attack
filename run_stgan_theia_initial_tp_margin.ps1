param(
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Device = "cuda:0",
    [int]$MaxRounds = 5,
    [int]$TargetsPerRound = 10,
    [switch]$SkipPrepare,
    [switch]$PreflightOnly
)

# The original entry point is retained for compatibility with commands already
# shared in experiment notes. Its implementation now uses initial_tp_margin.
& (Join-Path $PSScriptRoot "run_stgan_theia_initial_tp_cycle.ps1") @PSBoundParameters
exit $LASTEXITCODE
