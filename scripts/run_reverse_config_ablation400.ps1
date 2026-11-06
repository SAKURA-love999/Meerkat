param(
    [ValidateSet('Prepare','Fit','Summary')]
    [string]$Stage = 'Prepare',
    [string]$Python = 'C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$private = 'C:\Users\ccccc\.aris\projects\MeerKAT\20261101_023934'
$output = Join-Path $private 'experiments\reverse_config_ablation400_v1'
$config = Join-Path $private 'experiments\config_source_attribution400_v1'
$analysis = Join-Path $private 'experiments\config_source_attribution400_analysis'
$logger = Join-Path $private 'refine-logs\provenance\run_with_receipt.py'
$script = Join-Path $PSScriptRoot 'run_reverse_config_ablation.py'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$kind = 'audit'
if ($Stage -eq 'Fit') { $kind = 'fit' }
if ($Stage -eq 'Summary') { $kind = 'summary' }
$receipt = @('--run-id',"MK-REVERSE-CONFIG-$($Stage.ToUpper())-01",'--kind',$kind,
    '--receipt-root',(Join-Path $private 'logged_runs'),'--cwd',$project)
if ($Stage -ne 'Prepare') {
    $protocol = Join-Path $output 'protocol.json'
    if (-not (Test-Path -LiteralPath $protocol)) { throw 'Run Prepare first.' }
    $receipt += @('--protocol',$protocol)
}
foreach ($name in @('run_reverse_config_ablation.py','test_reverse_config_ablation.py',
    'run_reverse_config_ablation400.ps1','run_config_source_attribution.py','run_pca_sensitivity.py')) {
    $receipt += @('--code-file',(Join-Path $PSScriptRoot $name))
}
foreach ($path in @((Join-Path $config 'protocol.json'),(Join-Path $config 'verification.json'),
    (Join-Path $config 'all_oof_manifest.csv'),(Join-Path $analysis 'verification.json'))) {
    $receipt += @('--input-file',$path)
}
if ($Stage -eq 'Prepare') {
    foreach ($name in @('protocol.json','prepare_check.json','fit_plan.csv','training_feature_support.csv',
        'reference_oof_manifest.csv','cohort_label_fold_counts.csv')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
} else {
    foreach ($name in @('paired_metrics_and_ci.csv','four_model_metrics.csv',
        'predictive_interaction_and_ci.csv','fold_metrics.csv','fit_diagnostics.csv',
        'all_oof_manifest.csv','verification.json')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
}
$arguments = @($script,'--project-root',$project,
    '--shared-dir',(Join-Path $private 'experiments\shared400'),
    '--controls-dir',(Join-Path $private 'experiments\control_extensions400'),
    '--b1-run',(Join-Path $private 'experiments\baseline_capacity_b1_v1'),
    '--b2-run',(Join-Path $private 'experiments\baseline_capacity_b2_v1'),
    '--b3-run',(Join-Path $private 'experiments\baseline_capacity_b3_v1'),
    '--pca-run',(Join-Path $private 'experiments\pca_sensitivity400_v1'),
    '--audit-dir',(Join-Path $private 'experiments\config_fields400_audit'),
    '--config-run',$config,'--config-analysis',$analysis,
    '--output-dir',$output,'--stage',$Stage.ToLower())
Write-Host "Starting $Stage. CPU4,30 M78-only fits;three existing models are reused."
Write-Host 'Fixed calls/folds/history/controls/C1/HGB/no-PCA. No feature extraction or tuning.'
Write-Host "Output: $output"
& $Python $logger @receipt -- $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Reverse ablation failed;inspect receipt. No automatic next stage.' }
Write-Host 'Completed. No automatic follow-up or background monitoring.'
