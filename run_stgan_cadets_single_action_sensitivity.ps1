param(
    [string]$Device = "cuda:0",
    [int]$Seed = 42,
    [int]$MacroLength = 5,
    [int[]]$Targets = @(7, 8, 10, 11, 49, 181, 342, 760, 103054)
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "D:\App\anaconda\Scripts\conda.exe"
$Script = Join-Path $Workspace "MAGIC_main\stgan_single_action_sensitivity.py"
$State = Join-Path $Workspace "STGAN_master\data\cadets\cadets_stgan_official_attack_test0.npz"
$Output = Join-Path $Workspace "attack_results\stgan_cadets_single_action_sensitivity"

foreach ($Required in @($Conda, $Script, $State)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file is missing: $Required"
    }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$Arguments = @(
    "--dataset", "cadets",
    "--test-index", "0",
    "--device", $Device,
    "--seed", "$Seed",
    "--macro-length", "$MacroLength",
    "--semantic-state", $State,
    "--output-dir", $Output,
    "--targets"
) + ($Targets | ForEach-Object { "$_" })

& $Conda run --no-capture-output -n STGAN python $Script @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "STGAN single-action sensitivity test failed with exit code $LASTEXITCODE"
}
