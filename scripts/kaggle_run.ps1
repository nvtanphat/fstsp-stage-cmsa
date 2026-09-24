param(
  [Parameter(Mandatory=$true)][string]$Username,
  [string]$Slug = "fstsp-stage-cmsa-reimplementation",
  [int]$Timeout = 1200
)

$ErrorActionPreference = "Stop"

$kaggleJson = Join-Path $HOME ".kaggle\kaggle.json"
if (Test-Path $kaggleJson) {
  $creds = Get-Content $kaggleJson -Raw | ConvertFrom-Json
  $env:KAGGLE_USERNAME = $creds.username
  $env:KAGGLE_KEY = $creds.key
} elseif ($Username) {
  $env:KAGGLE_USERNAME = $Username
}

python scripts/prepare_kaggle_kernel.py --username $Username --slug $Slug --private
$Kernel = "$Username/$Slug"

Write-Host "Pushing and running $Kernel ..."
kaggle kernels push -p dist/kaggle_kernel -t $Timeout

Write-Host "Current status:"
kaggle kernels status $Kernel

Write-Host "When the status becomes complete, download outputs with:"
Write-Host "kaggle kernels output $Kernel -p artifacts/kaggle_download -o"
