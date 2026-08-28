param(
    [ValidateSet("pretrained", "finetuned")]
    [string]$Checkpoint = "pretrained",
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"

$checkpoints = @{
    pretrained = @{
        DatafileId = "253220"
        Filename = "animal2vec_large_pretrained_MeerKAT_240507.pt"
        Md5 = "c0ae0cb16afd0501f00a5955fb6482ed"
    }
    finetuned = @{
        DatafileId = "253219"
        Filename = "animal2vec_large_finetuned_MeerKAT_240507.pt"
        Md5 = "b377ea79700f3bbc98b6154f21545158"
    }
}

$selected = $checkpoints[$Checkpoint]
$outDir = Join-Path $ProjectRoot "models\animal2vec"
$outPath = Join-Path $outDir $selected.Filename
$url = "https://edmond.mpg.de/api/access/datafile/$($selected.DatafileId)"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "Downloading $Checkpoint checkpoint to $outPath"
Write-Host "Source: $url"
curl.exe -L --ssl-no-revoke -C - -o $outPath $url

$hash = Get-FileHash -Algorithm MD5 -LiteralPath $outPath
Write-Host "MD5: $($hash.Hash.ToLowerInvariant())"
if ($hash.Hash.ToLowerInvariant() -ne $selected.Md5) {
    throw "MD5 mismatch. Expected $($selected.Md5)."
}

Write-Host "Checkpoint download verified."
