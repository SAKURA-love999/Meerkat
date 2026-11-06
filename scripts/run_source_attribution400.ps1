param(
    [ValidateSet('Prepare', 'Preflight', 'Fit', 'Audit')]
    [string]$Stage = 'Prepare',
    [string]$Python = 'C:\Users\ccccc\Documents\Codex\envs\meerkat-animal2vec-py39\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$private = 'C:\Users\ccccc\.aris\projects\MeerKAT\20261101_023934'
$scripts = Join-Path $project 'scripts'
$output = Join-Path $private 'experiments\control_blocks400_v3'
$logger = Join-Path $private 'refine-logs\provenance\run_with_receipt.py'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path -LiteralPath $Python)) { throw 'Python not found. Supply -Python with the verified compatible interpreter.' }

if ($Stage -eq 'Audit') {
    $script = Join-Path $scripts 'audit_confirmation_clip_pool.py'
    $auditOutput = Join-Path $private 'experiments\confirmation_pool400'
    $commandArgs = @($script, '--manifest', (Join-Path $project 'data_manifest\call_manifest.csv'),
        '--scan-root', (Join-Path $project 'data_manifest'),
        '--scan-root', (Join-Path $project 'features'),
        '--scan-root', (Join-Path $project 'reports'),
        '--scan-root', (Join-Path $private 'experiments'),
        '--output-dir', $auditOutput)
    $receiptArgs = @('--run-id', 'MK-CONFIRMATION-POOL400-01', '--kind', 'audit',
        '--receipt-root', (Join-Path $private 'logged_runs'), '--cwd', $project,
        '--code-file', $script, '--input-file', (Join-Path $project 'data_manifest\call_manifest.csv'),
        '--artifact', (Join-Path $auditOutput 'summary.json'),
        '--artifact', (Join-Path $auditOutput 'table_coverage.csv'))
} else {
    $script = Join-Path $scripts 'run_control_block_attribution.py'
    $commandArgs = @($script, '--project-root', $project,
        '--shared-dir', (Join-Path $private 'experiments\shared400'),
        '--controls-dir', (Join-Path $private 'experiments\control_extensions400'),
        '--output-dir', $output)
    $kind = 'audit'
    if ($Stage -eq 'Preflight') { $commandArgs += '--preflight'; $kind = 'fit' }
    if ($Stage -eq 'Fit') { $commandArgs += '--run-models'; $kind = 'fit' }
    $receiptArgs = @('--run-id', ('MK-CONTROL-BLOCKS400-' + $Stage.ToUpper() + '-01'),
        '--kind', $kind, '--receipt-root', (Join-Path $private 'logged_runs'), '--cwd', $project)
    if ($Stage -ne 'Prepare') {
        $protocol = Join-Path $output 'protocol.json'
        if (-not (Test-Path -LiteralPath $protocol)) { throw 'Run -Stage Prepare first.' }
        $receiptArgs += @('--protocol', $protocol)
    }
    foreach ($name in @('run_control_block_attribution.py', 'extend_shared_cohort_controls.py',
            'run_shared_cohort_selection_control.py', 'fit_meerkat_m1_acoustic_residual.py',
            'audit_egemaps_selection_on_librosa.py', 'run_source_attribution400.ps1')) {
        $receiptArgs += @('--code-file', (Join-Path $scripts $name))
    }
    foreach ($path in @((Join-Path $private 'experiments\shared400\oof_predictions.csv'),
            (Join-Path $private 'experiments\shared400\cohort_with_history.pkl'),
            (Join-Path $private 'experiments\shared400\clip_fold_mapping.csv'),
            (Join-Path $private 'experiments\control_extensions400\oof_predictions_added.csv'),
            (Join-Path $private 'experiments\control_extensions400\cohort_controls.csv'))) {
        $receiptArgs += @('--input-file', $path)
    }
    $receiptArgs += @('--artifact', (Join-Path $output 'protocol.json'))
    if ($Stage -eq 'Preflight') {
        $receiptArgs += @('--artifact', (Join-Path $output 'preflight.json'),
            '--artifact', (Join-Path $output 'preflight_predictions.csv'))
    }
    if ($Stage -eq 'Fit') {
        foreach ($name in @('paired_metrics_and_ci.csv', 'oof_predictions_added.csv', 'verification.json')) {
            $receiptArgs += @('--artifact', (Join-Path $output $name))
        }
    }
}

Write-Host "Starting $Stage. CPU only. Logs are captured in $private\logged_runs."
& $Python $logger @receiptArgs -- $Python @commandArgs
if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed. Inspect the printed receipt directory; do not proceed to Fit after a failed preflight." }
Write-Host "Completed $Stage. No background monitoring is running."
