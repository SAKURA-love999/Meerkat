param(
    [ValidateSet('Prepare','Fit','Summary')]
    [string]$Stage = 'Prepare',
    [string]$Python = 'C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$private = 'C:\Users\ccccc\.aris\projects\MeerKAT\20261101_023934'
$output = Join-Path $private 'experiments\baseline_capacity_b3_v1'
$controls = Join-Path $private 'experiments\control_extensions400'
$b1 = Join-Path $private 'experiments\baseline_capacity_b1_v1'
$b2 = Join-Path $private 'experiments\baseline_capacity_b2_v1'
$logger = Join-Path $private 'refine-logs\provenance\run_with_receipt.py'
$script = Join-Path $PSScriptRoot 'run_baseline_capacity_b3.py'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$kind = 'audit'
if ($Stage -eq 'Fit') { $kind = 'fit' }
if ($Stage -eq 'Summary') { $kind = 'summary' }
$receipt = @('--run-id',"MK-BASECAP-B3-$($Stage.ToUpper())-01",'--kind',$kind,
    '--receipt-root',(Join-Path $private 'logged_runs'),'--cwd',$project)
if ($Stage -ne 'Prepare') {
    $protocol = Join-Path $output 'protocol.json'
    if (-not (Test-Path -LiteralPath $protocol)) { throw 'Run Prepare first.' }
    $receipt += @('--protocol',$protocol)
}
foreach ($name in @('baseline_capacity_models.py','run_baseline_capacity_b3.py','run_baseline_capacity_b3_400.ps1')) {
    $receipt += @('--code-file',(Join-Path $PSScriptRoot $name))
}
foreach ($path in @((Join-Path $controls 'protocol.json'),(Join-Path $b1 'protocol.json'),
    (Join-Path $b2 'protocol.json'),(Join-Path $b2 'verification.json'))) {
    $receipt += @('--input-file',$path)
}
if ($Stage -eq 'Prepare') {
    foreach ($name in @('protocol.json','prepare_check.json','reference_oof_manifest.csv','cohort_label_fold_counts.csv')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
} else {
    foreach ($name in @('comparison_metrics_and_ci.csv','paired_changes_vs_linear.csv',
        'fold_metrics.csv','all_oof_manifest.csv','verification.json')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
}
$arguments = @($script,'--project-root',$project,
    '--shared-dir',(Join-Path $private 'experiments\shared400'),'--controls-dir',$controls,
    '--b1-run',$b1,'--b2-run',$b2,'--output-dir',$output,'--stage',$Stage.ToLower())
Write-Host "Starting $Stage. B3 real-call capacity sensitivity; CPU only,4 threads."
Write-Host 'Two new learners,40 formal fits;10 original raw shared-control linear jobs reused.'
Write-Host "Output: $output. Native logs are captured in the printed receipt directory."
& $Python $logger @receipt -- $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'B3 failed;inspect its receipt before continuing.' }
Write-Host 'Completed. No automatic follow-up or monitoring is started.'
