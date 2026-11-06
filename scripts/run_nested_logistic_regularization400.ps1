param(
    [ValidateSet('Prepare','Fit','Summary')]
    [string]$Stage = 'Prepare',
    [string]$Python = 'C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$private = 'C:\Users\ccccc\.aris\projects\MeerKAT\20261101_023934'
$output = Join-Path $private 'experiments\nested_logistic_regularization400_v1'
$b3 = Join-Path $private 'experiments\baseline_capacity_b3_v1'
$logger = Join-Path $private 'refine-logs\provenance\run_with_receipt.py'
$script = Join-Path $PSScriptRoot 'run_nested_logistic_regularization.py'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$kind = 'audit'
if ($Stage -eq 'Fit') { $kind = 'fit' }
if ($Stage -eq 'Summary') { $kind = 'summary' }
$receipt = @('--run-id',"MK-NESTED-C400-$($Stage.ToUpper())-01",'--kind',$kind,
    '--receipt-root',(Join-Path $private 'logged_runs'),'--cwd',$project)
if ($Stage -ne 'Prepare') {
    $protocol = Join-Path $output 'protocol.json'
    if (-not (Test-Path -LiteralPath $protocol)) { throw 'Run Prepare first.' }
    $receipt += @('--protocol',$protocol)
}
foreach ($name in @('run_nested_logistic_regularization.py','test_nested_logistic_regularization.py',
    'run_nested_logistic_regularization400.ps1','baseline_capacity_models.py','run_baseline_capacity_b3.py')) {
    $receipt += @('--code-file',(Join-Path $PSScriptRoot $name))
}
foreach ($name in @('protocol.json','all_oof_manifest.csv','verification.json')) {
    $receipt += @('--input-file',(Join-Path $b3 $name))
}
if ($Stage -eq 'Prepare') {
    foreach ($name in @('protocol.json','prepare_check.json','inner_clip_folds.csv','fit_plan.csv',
        'reference_oof_manifest.csv','cohort_label_fold_counts.csv','inner_fold_counts.csv')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
} else {
    foreach ($name in @('inner_fold_scores.csv','candidate_scores.csv','selected_C.csv',
        'selected_outer_fit_plan.csv','comparison_metrics_and_ci.csv','paired_changes_vs_fixed.csv',
        'fold_metrics.csv','all_oof_manifest.csv','fit_diagnostics.csv','verification.json')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
}
$arguments = @($script,'--project-root',$project,
    '--shared-dir',(Join-Path $private 'experiments\shared400'),
    '--controls-dir',(Join-Path $private 'experiments\control_extensions400'),
    '--b1-run',(Join-Path $private 'experiments\baseline_capacity_b1_v1'),
    '--b2-run',(Join-Path $private 'experiments\baseline_capacity_b2_v1'),
    '--b3-run',$b3,'--output-dir',$output,'--stage',$Stage.ToLower())
Write-Host "Starting $Stage. CPU only,4 threads. PCA30 and all original features/controls fixed."
Write-Host 'Fresh Fit:360 inner fits +40 selected outer refits; all candidates/PRE/POST retained.'
Write-Host "Output: $output"
& $Python $logger @receipt -- $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Nested C stage failed; inspect receipt. No automatic next stage.' }
Write-Host 'Completed. No automatic follow-up, GPU work or monitoring.'
