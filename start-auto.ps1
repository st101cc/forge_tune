param([string]$VmHost = '', [switch]$CheckOnly, [switch]$SelfTest)
$ErrorActionPreference = 'Stop'
function Has-Local24GbGpu($MemoryMiB) {
    return @($MemoryMiB | Where-Object { $_ -and [double]$_ -ge 24000 }).Count -gt 0
}
if ($SelfTest) {
    if ((Has-Local24GbGpu @()) -or (Has-Local24GbGpu @(15360,15360)) -or -not (Has-Local24GbGpu @(24564))) { throw 'GPU selection failed' }
    Write-Output 'GPU selection checks passed: no GPU, two T4s, nominal 24 GB GPU.'
    return
}
$projectPath = $PSScriptRoot
$runtimePath = Join-Path $projectPath '.forge-data'
$logPath = Join-Path $runtimePath 'logs'
New-Item -ItemType Directory -Force -Path $logPath | Out-Null
# Nominal 24 GB cards can report slightly less usable VRAM.
$localGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $memory = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
    $localGpu = Has-Local24GbGpu $memory
}
if (-not $localGpu -and -not $VmHost) { throw 'No local NVIDIA GPU with at least 24 GB was found. Pass -VmHost <ssh-host> to use a remote GPU machine.' }
$target = if ($localGpu) { 'local' } else { $VmHost }
Write-Output "Backend target: $target"
if ($CheckOnly) { return }
$pythonPath = Join-Path $projectPath '.venv/Scripts/python.exe'
$env:FORGETUNE_DATA_ROOT = $runtimePath
if ($localGpu) {
    & $pythonPath -c 'import torch; assert torch.cuda.is_available(), "Install CUDA-enabled PyTorch in the project environment"'
    if ($LASTEXITCODE -ne 0) { throw 'Local CUDA runtime is unavailable.' }
    if (-not (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) {
        Start-Process $pythonPath -ArgumentList '-m uvicorn app:app --app-dir trainer --host 127.0.0.1 --port 8000' -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput "$logPath/api.out.log" -RedirectStandardError "$logPath/api.err.log" | Out-Null
    }
    $env:TRAINER_URL = 'http://127.0.0.1:8000'
} else {
    # The VM backend is installed separately; SSH never exposes its API publicly.
    & ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 $VmHost 'curl --fail --silent http://127.0.0.1:18000/hardware'
    if ($LASTEXITCODE -ne 0) { throw "Remote backend is offline. Run trainer/start-vm-backend.sh from the ForgeTune checkout on $VmHost." }
    if (-not (Get-NetTCPConnection -LocalPort 18000 -State Listen -ErrorAction SilentlyContinue)) {
        Start-Process ssh -ArgumentList @('-N','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-L','127.0.0.1:18000:127.0.0.1:18000',$VmHost) -WindowStyle Hidden -RedirectStandardOutput "$logPath/tunnel.out.log" -RedirectStandardError "$logPath/tunnel.err.log" | Out-Null
    }
    $env:TRAINER_URL = 'http://127.0.0.1:18000'
}
$healthy = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try { $hardware = Invoke-RestMethod "$env:TRAINER_URL/hardware" -TimeoutSec 2; $healthy = $true; break } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $healthy) { throw 'Selected backend did not become reachable. Check .forge-data/logs.' }
Write-Output "Connected: $($hardware.name), $($hardware.memoryGb) GB per selected GPU"
$webPidFile = Join-Path $runtimePath 'web.pid'
if (Test-Path $webPidFile) {
    $priorPid = [int](Get-Content $webPidFile)
    $prior = Get-CimInstance Win32_Process -Filter "ProcessId=$priorPid"
    if ($prior -and $prior.CommandLine.Contains((Join-Path $PSScriptRoot 'server.mjs'))) { Stop-Process -Id $priorPid }
}
if (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 4173 is already in use. Stop the previous website process, then rerun this script.' }
$web = Start-Process (Get-Command node).Source -ArgumentList ('"' + (Join-Path $PSScriptRoot 'server.mjs') + '"') -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput "$logPath/web.out.log" -RedirectStandardError "$logPath/web.err.log" -PassThru
$web.Id | Set-Content $webPidFile
Write-Output 'Website: http://127.0.0.1:4173/'
