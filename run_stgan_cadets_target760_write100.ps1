param(
    [string]$Device = "cuda:0",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "D:\App\anaconda\Scripts\conda.exe"
$Script = Join-Path $Workspace "MAGIC_main\stgan_repeated_action_probe.py"
$State = Join-Path $Workspace "STGAN_master\data\cadets\cadets_stgan_official_attack_test0.npz"
$Output = Join-Path $Workspace "attack_results\stgan_cadets_repeated_action_probe"

foreach ($Required in @($Conda, $Script, $State)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required file is missing: $Required"
    }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

& $Conda run --no-capture-output -n STGAN python $Script `
    --dataset cadets `
    --test-index 0 `
    --device $Device `
    --seed $Seed `
    --target 760 `
    --action-id 2 `
    --macro-length 5 `
    --macro-count 20 `
    --semantic-state $State `
    --output-dir $Output

if ($LASTEXITCODE -ne 0) {
    throw "STGAN target-760 repeated-WRITE probe failed with exit code $LASTEXITCODE"
}
