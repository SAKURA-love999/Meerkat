param(
    [ValidateSet('Prepare','Fit','Summary')]
    [string]$Stage = 'Prepare',
    [switch]$FullPilot,
    [string]$Python = 'C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$private = 'C:\Users\ccccc\.aris\projects\MeerKAT\20261101_023934'
$script = Join-Path $PSScriptRoot 'run_semisynthetic_calibration.py'
$logger = Join-Path $private 'refine-logs\provenance\run_with_receipt.py'
$name = 'calibration_pilot400_quick_v1'
$seeds = '431'
if ($FullPilot) { $name = 'calibration_pilot400_v1'; $seeds = '431,907,1601,2203,3251' }
$output = Join-Path $private "experiments\$name"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Verified Python interpreter not found.' }

$arguments = @($script,'--project-root',$project,
    '--shared-dir',(Join-Path $private 'experiments\shared400'),
    '--controls-dir',(Join-Path $private 'experiments\control_extensions400'),
    '--output-dir',$output,'--seeds',$seeds,'--stage',$Stage.ToLower())
$kind = 'audit'
if ($Stage -eq 'Fit') { $kind = 'fit' }
if ($Stage -eq 'Summary') { $kind = 'summary' }
$mode = 'QUICK'
if ($FullPilot) { $mode = 'FULL' }
$receipt = @('--run-id',"MK-CALIBRATION400-$mode-$($Stage.ToUpper())-01",'--kind',$kind,
    '--receipt-root',(Join-Path $private 'logged_runs'),'--cwd',$project)
if ($Stage -ne 'Prepare') {
    $protocol = Join-Path $output 'protocol.json'
    if (-not (Test-Path -LiteralPath $protocol)) { throw 'Run matching Prepare first.' }
    $receipt += @('--protocol',$protocol,'--input-file',(Join-Path $output 'synthetic_design_manifest.json'))
}
foreach ($name in @('run_semisynthetic_calibration.py','calibration_dgp.py',
    'run_control_block_attribution.py','extend_shared_cohort_controls.py',
    'run_shared_cohort_selection_control.py','fit_meerkat_m1_acoustic_residual.py',
    'audit_egemaps_selection_on_librosa.py','run_calibration_pilot400.ps1')) {
    $receipt += @('--code-file',(Join-Path $PSScriptRoot $name))
}
$receipt += @('--input-file',(Join-Path $private 'experiments\shared400\cohort_with_history.pkl'),
    '--input-file',(Join-Path $private 'experiments\shared400\clip_fold_mapping.csv'),
    '--input-file',(Join-Path $private 'experiments\control_extensions400\cohort_controls.csv'))
if ($Stage -eq 'Prepare') {
    $receipt += @('--artifact',(Join-Path $output 'protocol.json'),
        '--artifact',(Join-Path $output 'synthetic_design_manifest.json'))
} else {
    foreach ($name in @('synthetic_metrics.csv','replicate_diagnostic_rates.csv','verification.json')) {
        $receipt += @('--artifact',(Join-Path $output $name))
    }
}
Write-Host "Starting $Stage ($mode). Synthetic assay, CPU only, no audio extraction."
Write-Host "Output: $output. Logs are captured in the receipt directory."
& $Python $logger @receipt -- $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Calibration stage failed; inspect its receipt logs before continuing.' }
Write-Host 'Completed. No background monitoring is running.'
