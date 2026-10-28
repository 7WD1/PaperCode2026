param([string]$PdfLaTeX='pdflatex',[string]$Python='python')
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.tex' | ForEach-Object {
        & $PdfLaTeX -interaction=nonstopmode -halt-on-error -file-line-error $_.Name
        if ($LASTEXITCODE -ne 0) { throw "Diagram build failed: $($_.Name)" }
        $builtPdf = Join-Path $PSScriptRoot ($_.BaseName + '.pdf')
        Copy-Item -LiteralPath $builtPdf -Destination (Join-Path '..\figures' ($_.BaseName + '.pdf'))
    }
    & $Python '..\scripts\export_diagrams.py'
    if ($LASTEXITCODE -ne 0) { throw 'PNG export failed' }
} finally { Pop-Location }
