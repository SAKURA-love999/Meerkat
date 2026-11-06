param(
    [ValidateSet('Prepare','Fit','Summary')]
    [string]$Stage = 'Prepare',
    [string]$Python = 'C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$private = 'C:\Users\ccccc\.aris\projects\MeerKAT\20261101_023934'
$output = Join-Path $private 'experiments\baseline_capacity_b1_v1'
$reference = Join-Path $private 'experiments\calibration_asymmetric_null400_v1'
$logger = Join-Path $private 'refine-logs\provenance\run_with_receipt.py'
$script = Join-Path $PSScriptRoot 'run_baseline_capacity.py'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$kind = 'audit'
if ($Stage -eq 'Fit') { $kind = 'fit' }
if ($Stage -eq 'Summary') { $kind = 'summary' }
$receipt = @('--run-id',"MK-BASECAP-B1-$($Stage.ToUpper())-01",'--kind',$kind,
    '--receipt-root',(Join-Path $private 'logged_runs'),'--cwd',$project)
if ($Stage -ne 'Prepare') {
    $protocol = Join-Path $output 'protocol.json'
    if (-not (Test-Path -LiteralPath $protocol)) { throw 'Run Prepare first.' }
    $receipt += @('--protocol',$protocol)
}
foreach ($name in @('baseline_capacity_models.py','run_baseline_capacity.py','run_baseline_capacity400.ps1')) {
    $receipt += @('--code-file',(Join-Path $PSScriptRoot $name))
}
foreach ($name in @('protocol.json','synthetic_oof_manifest.csv','verification.json')) {
    $receipt += @('--input-file',(Join-Path $reference $name))
}
if ($Stage -eq 'Prepare') {
    $receipt += @('--artifact',(Join-Path $output 'protocol.json'),
        '--artifact',(Join-Path $output 'prepare_check.json'),
        '--artifact',(Join-Path $output 'reference_oof_manifest.csv'))
} else {
    foreach ($name in @('comparison_metrics_and_ci.csv','paired_changes_vs_linear.csv','verification.json')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
}
$arguments = @($script,'--project-root',$project,
    '--shared-dir',(Join-Path $private 'experiments\shared400'),
    '--controls-dir',(Join-Path $private 'experiments\control_extensions400'),
    '--reference-run',$reference,'--output-dir',$output,'--stage',$Stage.ToLower())
Write-Host "Starting $Stage. B1 synthetic capacity sensitivity; CPU only."
Write-Host 'Two new learners,40 new pipeline fits. Existing reference OOF is reused.'
Write-Host "Output: $output. Logs are captured in the receipt directory."
& $Python $logger @receipt -- $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'B1 failed;inspect its receipt before continuing.' }
Write-Host 'Completed. No automatic B2/B3 run or monitoring is started.'
