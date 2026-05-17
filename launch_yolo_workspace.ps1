param(
    [switch]$NewChromeWindow
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

$GitHubUrl = "https://github.com/Sourabhvk/realtime-road-perception-yolov8"
$ChatGptProjectUrl = "https://chatgpt.com/g/g-p-6a031464614c81919ec535af5dc36ebc-yolo-v8/project"

function Get-ChromePath {
    $candidates = @(
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        (Join-Path $env:LOCALAPPDATA "Google\Chrome\Application\chrome.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $command = Get-Command chrome.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "Google Chrome was not found. Install Chrome or add chrome.exe to PATH."
}

if (-not (Test-Path $VenvActivate)) {
    throw "Virtual environment activation script not found: $VenvActivate"
}

$chromeArgs = @($GitHubUrl, $ChatGptProjectUrl)
if ($NewChromeWindow) {
    $chromeArgs = @("--new-window") + $chromeArgs
}

Start-Process -FilePath (Get-ChromePath) -ArgumentList $chromeArgs

$codeCommand = Get-Command code -ErrorAction SilentlyContinue
if (-not $codeCommand) {
    throw "VS Code command 'code' was not found in PATH."
}

Start-Process -FilePath $codeCommand.Source -ArgumentList @($ProjectRoot)

$terminalCommand = @"
Set-Location "$ProjectRoot"
. "$VenvActivate"
Write-Host ""
Write-Host "YOLO workspace ready: $ProjectRoot" -ForegroundColor Green
Write-Host "Virtual environment activated: .venv" -ForegroundColor Green
"@

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
    $terminalCommand
)
