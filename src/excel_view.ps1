# Export the top of the "Stock plan" sheet to PDF with Excel (invisible), for a screenshot.
$root = Split-Path $PSScriptRoot -Parent
$xl = New-Object -ComObject Excel.Application
$xl.Visible = $false; $xl.DisplayAlerts = $false
try {
    $wb = $xl.Workbooks.Open((Join-Path $root "output\forecast_next_8_weeks.xlsx"), 0, $true)
    $ws = $wb.Worksheets.Item("Stock plan")
    $ps = $ws.PageSetup
    $ps.PrintArea = '$A$1:$H$19'; $ps.Orientation = 2; $ps.Zoom = $false; $ps.FitToPagesWide = 1; $ps.FitToPagesTall = 1
    $ws.ExportAsFixedFormat(0, (Join-Path $root "portfolio\excel_view.pdf"))
    $wb.Close($false); "exported"
} finally { $xl.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($xl) | Out-Null }
