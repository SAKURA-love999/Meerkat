param(
    [ValidateSet('Prepare','Fit','Summary')]
    [string]$Stage = 'Prepare',
    [string]$Python = 'C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$private = 'C:\Users\ccccc\.aris\projects\MeerKAT\20261101_023934'
$source = Join-Path $private 'experiments\calibration_pilot400_quick_v1'
$output = Join-Path $private 'experiments\calibration_asymmetric_null400_v1'
$script = Join-Path $PSScriptRoot 'run_asymmetric_null_calibration.py'
$logger = Join-Path $private 'refine-logs\provenance\run_with_receipt.py'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Verified Python interpreter not found.' }
$arguments = @($script,'--project-root',$project,
    '--shared-dir',(Join-Path $private 'experiments\shared400'),
    '--controls-dir',(Join-Path $private 'experiments\control_extensions400'),
    '--base-run',$source,'--output-dir',$output,'--stage',$Stage.ToLower())
$kind = 'audit'
if ($Stage -eq 'Fit') { $kind = 'fit' }
if ($Stage -eq 'Summary') { $kind = 'summary' }
$receipt = @('--run-id',"MK-ASYMMETRIC-NULL400-$($Stage.ToUpper())-01",'--kind',$kind,
    '--receipt-root',(Join-Path $private 'logged_runs'),'--cwd',$project)
if ($Stage -ne 'Prepare') {
    $protocol = Join-Path $output 'protocol.json'
    if (-not (Test-Path -LiteralPath $protocol)) { throw 'Run Prepare first.' }
    $receipt += @('--protocol',$protocol)
}
foreach ($name in @('run_asymmetric_null_calibration.py','calibration_dgp_asymmetric.py',
    'run_asymmetric_null400.ps1','run_semisynthetic_calibration.py','calibration_dgp.py',
    'run_control_block_attribution.py','extend_shared_cohort_controls.py',
    'run_shared_cohort_selection_control.py','fit_meerkat_m1_acoustic_residual.py',
    'audit_egemaps_selection_on_librosa.py')) {
    $receipt += @('--code-file',(Join-Path $PSScriptRoot $name))
}
foreach ($name in @('protocol.json','verification.json','synthetic_oof_manifest.csv',
    'designs\sim_seed431_null_nonlinear.npz')) {
    $receipt += @('--input-file',(Join-Path $source $name))
}
$receipt += @('--input-file',(Join-Path $private 'experiments\shared400\cohort_with_history.pkl'),
    '--input-file',(Join-Path $private 'experiments\shared400\clip_fold_mapping.csv'),
    '--input-file',(Join-Path $private 'experiments\control_extensions400\cohort_controls.csv'))
if ($Stage -eq 'Prepare') {
    $receipt += @('--artifact',(Join-Path $output 'protocol.json'),
        '--artifact',(Join-Path $output 'pairing_check.json'),
        '--artifact',(Join-Path $output 'post_reuse_manifest.csv'))
} else {
    foreach ($name in @('synthetic_metrics.csv','verification.json','post_reuse_manifest.csv')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
}
Write-Host "Starting $Stage. Paired asymmetric null; CPU only, no audio extraction."
Write-Host 'Ten existing POST jobs are reused; ten PRE jobs require20 model fits.'
Write-Host "Output: $output. Logs are captured in the receipt directory."
& $Python $logger @receipt -- $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Asymmetric-null stage failed; inspect its receipt logs before continuing.' }
Write-Host 'Completed. No background monitoring is running.'
