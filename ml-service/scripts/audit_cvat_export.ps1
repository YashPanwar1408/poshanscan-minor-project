<#!
.SYNOPSIS
  Audit and stage a CVAT Ultralytics YOLO segmentation export without changing active labels or split files.

.DESCRIPTION
  Maps exported labels to the existing PoshanScan 84/24/12 image split by exact image basename,
  checks label structure and polygon geometry, compares labels with the current staged set, and
  writes a static HTML overlay gallery plus JSON/CSV reports. Export labels are copied only into
  a new review staging directory. Missing labels are reported; old labels are never substituted.

.EXAMPLE
  .\scripts\audit_cvat_export.ps1 -ArchivePath "D:\Downloads\task_export.zip"
#>
param(
  [Parameter(Mandatory = $true)]
  [string]$ArchivePath,
  [string]$StageName = 'cvat_import_staging',
  [string]$ReportName = ''
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ArchivePath = (Resolve-Path -LiteralPath $ArchivePath).Path
$DataRoot = Join-Path $ProjectRoot 'dataset'
$WorkingRoot = Join-Path $DataRoot 'yolo_working'
$SplitRoot = Join-Path $DataRoot 'splits'
$ResultsRoot = Join-Path $ProjectRoot 'results'
if (-not $ReportName) { $ReportName = 'cvat_import_audit_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$ReportRoot = Join-Path $ResultsRoot $ReportName
$StageRoot = Join-Path $ResultsRoot $StageName
if (Test-Path -LiteralPath $ReportRoot) { throw "Audit report directory already exists: $ReportRoot" }
if (Test-Path -LiteralPath $StageRoot) { throw "Staging directory already exists: $StageRoot; choose a new -StageName." }

function Html([string]$Text) { return [System.Net.WebUtility]::HtmlEncode($Text) }
function SvgNumber([double]$Value) { return $Value.ToString('0.########', [System.Globalization.CultureInfo]::InvariantCulture) }
function Cross([object]$A, [object]$B, [object]$C) {
  return (($B.X - $A.X) * ($C.Y - $A.Y) - ($B.Y - $A.Y) * ($C.X - $A.X))
}
function Count-ProperCrossings([System.Collections.Generic.List[object]]$Points) {
  $n = $Points.Count
  $count = 0
  $eps = 1e-12
  for ($i = 0; $i -lt $n; $i++) {
    $i2 = ($i + 1) % $n
    $a = $Points[$i]
    $b = $Points[$i2]
    $minAx = [Math]::Min($a.X, $b.X); $maxAx = [Math]::Max($a.X, $b.X)
    $minAy = [Math]::Min($a.Y, $b.Y); $maxAy = [Math]::Max($a.Y, $b.Y)
    for ($j = $i + 1; $j -lt $n; $j++) {
      $j2 = ($j + 1) % $n
      if ($i2 -eq $j -or $j2 -eq $i) { continue }
      $c = $Points[$j]
      $d = $Points[$j2]
      if ([Math]::Max($c.X, $d.X) -lt $minAx -or [Math]::Min($c.X, $d.X) -gt $maxAx -or
          [Math]::Max($c.Y, $d.Y) -lt $minAy -or [Math]::Min($c.Y, $d.Y) -gt $maxAy) { continue }
      $o1 = Cross $a $b $c; $o2 = Cross $a $b $d
      $o3 = Cross $c $d $a; $o4 = Cross $c $d $b
      $opposite12 = (($o1 -gt $eps -and $o2 -lt -$eps) -or ($o1 -lt -$eps -and $o2 -gt $eps))
      $opposite34 = (($o3 -gt $eps -and $o4 -lt -$eps) -or ($o3 -lt -$eps -and $o4 -gt $eps))
      if ($opposite12 -and $opposite34) { $count++ }
    }
  }
  return $count
}

# Reject unsafe/ambiguous member paths before reading label data.
$zip = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
try {
  $members = @($zip.Entries | Where-Object { $_.Name })
  $unsafe = @($members | Where-Object {
      $_.FullName.StartsWith('/') -or $_.FullName.Contains('\') -or
      (@($_.FullName -split '/') -contains '..') -or $_.FullName -match '^[A-Za-z]:'
    })
  if ($unsafe.Count) { throw "Unsafe ZIP paths found: $($unsafe.FullName -join ', ')" }
  $duplicatePaths = @($members | Group-Object { $_.FullName.ToLowerInvariant() } | Where-Object Count -gt 1)
  if ($duplicatePaths.Count) { throw "Duplicate ZIP member paths found: $($duplicatePaths.Name -join ', ')" }
  $entryByPath = @{}
  foreach ($member in $members) { $entryByPath[$member.FullName] = $member }
  $yamlMembers = @($members | Where-Object { [System.IO.Path]::GetFileName($_.FullName).ToLowerInvariant() -eq 'data.yaml' })
  if ($yamlMembers.Count -ne 1) { throw "Expected one data.yaml; found $($yamlMembers.Count)." }
  if (-not $entryByPath.ContainsKey('train.txt')) { throw 'CVAT export does not contain train.txt.' }
  $labelEntries = @($members | Where-Object { $_.FullName -match '^labels/[^/]+/[^/]+\.txt$' })
  if ($labelEntries.Count -eq 0) { throw 'No labels/**/*.txt files found in the CVAT export.' }
  $archiveImageEntries = @($members | Where-Object { [System.IO.Path]::GetExtension($_.Name).ToLowerInvariant() -in @('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp') })

  $yamlReader = [System.IO.StreamReader]::new($yamlMembers[0].Open())
  $yamlText = $yamlReader.ReadToEnd(); $yamlReader.Dispose()
  $nameLines = @($yamlText -split "`r?`n" | Where-Object { $_ -match '^\s*\d+\s*:' })
  $classNames = @($nameLines | ForEach-Object { ($_ -split ':', 2)[1].Trim().Trim('"').Trim("'") })
  $classIds = @($nameLines | ForEach-Object { [int](($_ -split ':', 2)[0].Trim()) })
  if ($classNames.Count -ne 1 -or $classNames[0] -ne 'upper_arm' -or $classIds.Count -ne 1 -or $classIds[0] -ne 0) {
    throw "Expected only class 0: upper_arm. Found IDs=[$($classIds -join ',')] names=[$($classNames -join ',')]."
  }

  $trainReader = [System.IO.StreamReader]::new($entryByPath['train.txt'].Open())
  $archiveImageNames = @($trainReader.ReadToEnd() -split "`r?`n" | Where-Object { $_.Trim() } | ForEach-Object { [System.IO.Path]::GetFileName($_.Trim()) })
  $trainReader.Dispose()
  if (@($archiveImageNames | Select-Object -Unique).Count -ne $archiveImageNames.Count) { throw 'CVAT train.txt contains duplicate image names.' }

  # The ZIP's train.txt is an export task list, not the project split. Keep the local lists authoritative.
  $imagesBySplit = @{}
  $splitNameSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($split in @('train','val','test')) {
    $splitList = Join-Path $SplitRoot "$split.txt"
    if (-not (Test-Path -LiteralPath $splitList)) { throw "Missing existing split list: $splitList" }
    $splitFiles = @(Get-Content -LiteralPath $splitList | Where-Object { $_.Trim() } | ForEach-Object { [System.IO.Path]::GetFileName($_.Trim()) })
    $expected = @{train=84;val=24;test=12}[$split]
    if ($splitFiles.Count -ne $expected) { throw "Existing $split split has $($splitFiles.Count) entries; expected $expected. No split was changed." }
    $imagesBySplit[$split] = @()
    foreach ($name in $splitFiles) {
      if (-not $splitNameSet.Add($name)) { throw "Image name occurs in more than one existing split: $name" }
      $imagePath = Join-Path (Join-Path $WorkingRoot "images\$split") $name
      if (-not (Test-Path -LiteralPath $imagePath)) { throw "Split image not found: $imagePath" }
      $imagesBySplit[$split] += Get-Item -LiteralPath $imagePath
    }
  }
  $splitReportPath = Join-Path $DataRoot 'reports\split_report.json'
  if (-not (Test-Path -LiteralPath $splitReportPath)) { throw "Missing split report: $splitReportPath" }
  $existingSplitReport = Get-Content -LiteralPath $splitReportPath -Raw | ConvertFrom-Json
  if ($existingSplitReport.seed -ne 2026) { throw "Existing split report seed is $($existingSplitReport.seed), expected 2026. No split was changed." }
  if ($existingSplitReport.split_counts.train -ne 84 -or $existingSplitReport.split_counts.val -ne 24 -or $existingSplitReport.split_counts.test -ne 12) {
    throw 'Existing split report does not confirm the fixed 84/24/12 counts. No split was changed.'
  }
  if (-not $existingSplitReport.no_duplicate_or_near_duplicate_group_leakage) {
    throw 'Existing split report indicates duplicate/source-group leakage. No split was changed.'
  }
  if ($archiveImageNames.Count -ne 120) { throw "CVAT train.txt lists $($archiveImageNames.Count) images; expected the same 120 split images." }
  $archiveNameSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($name in $archiveImageNames) { [void]$archiveNameSet.Add($name) }
  $splitNameArray = @($splitNameSet)
  $archiveOnly = @($archiveNameSet | Where-Object { $_ -notin $splitNameArray })
  $splitOnly = @($splitNameArray | Where-Object { $_ -notin $archiveNameSet })
  if ($archiveOnly.Count -or $splitOnly.Count) { throw "CVAT image list does not match fixed local split names. Export-only=[$($archiveOnly -join ',')], split-only=[$($splitOnly -join ',')]." }

  $imagesByStem = @{}
  $splitByStem = @{}
  foreach ($split in @('train','val','test')) {
    foreach ($image in $imagesBySplit[$split]) {
      $stem = $image.BaseName.ToLowerInvariant()
      if ($imagesByStem.ContainsKey($stem)) { throw "Two active images share YOLO basename $stem." }
      $imagesByStem[$stem] = $image
      $splitByStem[$stem] = $split
    }
  }
  $labelsByStem = @{}
  foreach ($entry in $labelEntries) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($entry.Name).ToLowerInvariant()
    if ($labelsByStem.ContainsKey($stem)) { throw "Duplicate label basename in ZIP: $stem" }
    $labelsByStem[$stem] = $entry
  }

  New-Item -ItemType Directory -Path $ReportRoot | Out-Null
  New-Item -ItemType Directory -Path $StageRoot | Out-Null
  New-Item -ItemType Directory -Path (Join-Path $StageRoot 'labels') | Out-Null
  New-Item -ItemType Directory -Path (Join-Path $StageRoot 'source_export') | Out-Null
  foreach ($split in @('train','val','test')) { New-Item -ItemType Directory -Path (Join-Path $StageRoot "labels\$split") | Out-Null }
  [System.IO.File]::WriteAllText((Join-Path $StageRoot 'source_export\data.yaml'), $yamlText, [System.Text.UTF8Encoding]::new($false))
  [System.IO.File]::WriteAllLines((Join-Path $StageRoot 'source_export\train.txt'), $archiveImageNames, [System.Text.UTF8Encoding]::new($false))

  $imageAudit = [System.Collections.Generic.List[object]]::new()
  $polygonAudit = [System.Collections.Generic.List[object]]::new()
  $htmlBySplit = @{train=[System.Text.StringBuilder]::new();val=[System.Text.StringBuilder]::new();test=[System.Text.StringBuilder]::new()}
  foreach ($split in @('train','val','test')) {
    $headHtml = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CVAT import review - ' + $split + '</title><style>body{font:15px system-ui,sans-serif;margin:20px;background:#f4f6f8;color:#17212b}h1{font-size:22px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}.card{background:#fff;border:1px solid #ccd4dc;border-radius:8px;padding:9px}.frame{position:relative;width:100%;height:auto}.frame img{display:block;width:100%;height:100%;object-fit:fill}.frame svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.meta{font-size:12px;color:#44515d}.missing{color:#a11;font-weight:700}</style></head><body><h1>CVAT export overlays - ' + $split + ' split</h1><p>Review-only staging. Images are the existing fixed-split local files; polygons are from the ZIP. No source labels or split lists are modified.</p><div class="grid">'
    [void]$htmlBySplit[$split].AppendLine($headHtml)
    foreach ($image in $imagesBySplit[$split]) {
      $stem = $image.BaseName.ToLowerInvariant()
      $dimImage = [System.Drawing.Image]::FromFile($image.FullName)
      $width = $dimImage.Width; $height = $dimImage.Height; $dimImage.Dispose()
      $polygons = [System.Collections.Generic.List[string]]::new()
      $issuesForImage = [System.Collections.Generic.List[string]]::new()
      $labelPresent = $labelsByStem.ContainsKey($stem)
      $oldPath = Join-Path (Join-Path $WorkingRoot "labels\$split") ($image.BaseName + '.txt')
      $changed = 'missing'
      $rowCount = 0; $crossTotal = 0; $invalidTotal = 0; $edgeTouchTotal = 0
      if ($labelPresent) {
        $entry = $labelsByStem[$stem]
        $reader = [System.IO.StreamReader]::new($entry.Open()); $labelText = $reader.ReadToEnd(); $reader.Dispose()
        $oldText = if (Test-Path -LiteralPath $oldPath) { Get-Content -LiteralPath $oldPath -Raw } else { '' }
        $changed = if (($labelText -replace '\s','') -eq ($oldText -replace '\s','')) { 'unchanged' } else { 'changed' }
        $newLabelPath = Join-Path (Join-Path $StageRoot "labels\$split") ($image.BaseName + '.txt')
        [System.IO.File]::WriteAllText($newLabelPath, $labelText, [System.Text.UTF8Encoding]::new($false))
        $lines = @($labelText -split "`r?`n" | Where-Object { $_.Trim() })
        $rowCount = $lines.Count
        if ($rowCount -eq 0) { $issuesForImage.Add('empty_cvat_label') }
        $seenPolygonSignatures = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
        for ($lineNo = 1; $lineNo -le $lines.Count; $lineNo++) {
          $line = $lines[$lineNo - 1].Trim(); $fields = @($line -split '\s+')
          $rowIssues = [System.Collections.Generic.List[string]]::new()
          if ($fields.Count -lt 7 -or (($fields.Count - 1) % 2 -ne 0)) {
            $rowIssues.Add('bad_coordinate_count')
            $polygonAudit.Add([pscustomobject]@{split=$split;filename=$image.Name;label_row=$lineNo;vertex_count=0;proper_crossings=0;area=0;issues=($rowIssues -join ';')})
            $invalidTotal++; continue
          }
          $classId = -1
          if (-not [int]::TryParse($fields[0], [ref]$classId) -or $classId -ne 0) { $rowIssues.Add('class_id_not_0') }
          $coords = [System.Collections.Generic.List[double]]::new(); $numeric = $true
          for ($k = 1; $k -lt $fields.Count; $k++) {
            $value = 0.0
            if (-not [double]::TryParse($fields[$k], [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$value) -or [double]::IsNaN($value) -or [double]::IsInfinity($value)) { $numeric = $false; break }
            $coords.Add($value)
          }
          if (-not $numeric) { $rowIssues.Add('non_numeric_or_non_finite') }
          $points = [System.Collections.Generic.List[object]]::new()
          if ($numeric) {
            if (@($coords | Where-Object { $_ -lt 0 -or $_ -gt 1 }).Count) { $rowIssues.Add('coordinate_out_of_range') }
            for ($k = 0; $k -lt $coords.Count; $k += 2) { $points.Add([pscustomobject]@{X=[double]$coords[$k];Y=[double]$coords[$k+1]}) }
            $edgePoints = @($points | Where-Object { $_.X -le 1e-9 -or $_.X -ge (1.0 - 1e-9) -or $_.Y -le 1e-9 -or $_.Y -ge (1.0 - 1e-9) }).Count
            $edgeTouchTotal += $edgePoints
            if ($edgePoints -gt 0) { $rowIssues.Add("touches_image_edge_$edgePoints") }
            $distinct = @($points | ForEach-Object { (SvgNumber $_.X) + ',' + (SvgNumber $_.Y) } | Sort-Object -Unique).Count
            if ($distinct -lt 3) { $rowIssues.Add('fewer_than_3_distinct_points') }
            $signature = (@($points | ForEach-Object { (SvgNumber $_.X) + ',' + (SvgNumber $_.Y) } | Sort-Object) -join ';')
            if (-not $seenPolygonSignatures.Add($signature)) { $rowIssues.Add('duplicate_vertex_set') }
            $area2 = 0.0
            for ($k = 0; $k -lt $points.Count; $k++) { $next = ($k + 1) % $points.Count; $area2 += $points[$k].X * $points[$next].Y - $points[$next].X * $points[$k].Y }
            $area = [Math]::Abs($area2) / 2
            if ($area -le 1e-10) { $rowIssues.Add('zero_or_near_zero_area') }
            $crossings = Count-ProperCrossings $points
            $crossTotal += $crossings
            if ($crossings -gt 0) { $rowIssues.Add("self_crossings_$crossings") }
            $pointText = @($points | ForEach-Object { (SvgNumber $_.X) + ',' + (SvgNumber $_.Y) }) -join ' '
            $polygons.Add("<polygon points='$pointText' fill='rgba(0,255,80,.27)' stroke='#00b642' stroke-width='.0022' vector-effect='non-scaling-stroke'/>")
          } else { $area = 0; $crossings = 0 }
          if ($rowIssues.Count) { $invalidTotal++; foreach ($issue in $rowIssues) { $issuesForImage.Add("row${lineNo}:$issue") } }
          $polygonAudit.Add([pscustomobject]@{split=$split;filename=$image.Name;label_row=$lineNo;vertex_count=$points.Count;proper_crossings=$crossings;area=$area;issues=($rowIssues -join ';')})
        }
      } else {
        $issuesForImage.Add('missing_cvat_label_no_old_fallback')
      }
      $imageAudit.Add([pscustomobject]@{split=$split;filename=$image.Name;image_width=$width;image_height=$height;label_present=$labelPresent;polygon_count=$rowCount;proper_crossings=$crossTotal;edge_touching_vertices=$edgeTouchTotal;invalid_rows=$invalidTotal;label_vs_previous=$changed;issues=($issuesForImage -join ';')})
      $imageRelative = "../../dataset/yolo_working/images/$split/$($image.Name)"
      $status = if (-not $labelPresent) { "<span class='missing'>MISSING CVAT LABEL</span>" } elseif ($invalidTotal -or $crossTotal) { "<span class='missing'>geometry/format review</span>" } else { 'format/geometry checks passed' }
      $polygonsSvg = $polygons -join ''
      $imageNameHtml = Html $image.Name
      $issueSuffix = ''
      if ($issuesForImage.Count) { $issueSuffix = '; issues=' + (Html ($issuesForImage -join ', ')) }
      $cardHtml = "<section class='card'><div class='frame' style='aspect-ratio:$width/$height'><img loading='lazy' src='$imageRelative' alt='Original $imageNameHtml'><svg viewBox='0 0 1 1' preserveAspectRatio='none'>$polygonsSvg</svg></div><div><b>$imageNameHtml</b> - $status</div><div class='meta'>${width}x${height}; polygons=$rowCount; crossings=$crossTotal; edge vertices=$edgeTouchTotal; label=$changed$issueSuffix</div></section>"
      [void]$htmlBySplit[$split].AppendLine($cardHtml)
    }
    [void]$htmlBySplit[$split].AppendLine('</div></body></html>')
    [System.IO.File]::WriteAllText((Join-Path $ReportRoot "$split`_overlays.html"), $htmlBySplit[$split].ToString(), [System.Text.UTF8Encoding]::new($false))
  }

  foreach ($entry in $members | Where-Object { $_.FullName -eq 'train.txt' -or $_.FullName -eq 'data.yaml' }) {
    $target = Join-Path (Join-Path $StageRoot 'source_export') $entry.Name
    if ($entry.Name -eq 'train.txt') { continue }
    $reader = [System.IO.StreamReader]::new($entry.Open()); $content = $reader.ReadToEnd(); $reader.Dispose()
    [System.IO.File]::WriteAllText($target, $content, [System.Text.UTF8Encoding]::new($false))
  }
  $labelStems = @($labelsByStem.Keys)
  $activeStems = @($imagesByStem.Keys)
  $missingLabels = @($activeStems | Where-Object { -not $labelsByStem.ContainsKey($_) })
  $extraLabels = @($labelStems | Where-Object { -not $imagesByStem.ContainsKey($_) })
  $totalPolygons = ($polygonAudit | Measure-Object -Property vertex_count -Sum).Count
  $crossRows = @($polygonAudit | Where-Object proper_crossings -gt 0)
  $duplicateLeakage = $false
  $splitManifestPath = Join-Path $SplitRoot 'split_manifest.csv'
  if (Test-Path -LiteralPath $splitManifestPath) {
    $manifestRows = Import-Csv -LiteralPath $splitManifestPath
    $groupLeak = @($manifestRows | Group-Object duplicate_group | Where-Object { (@($_.Group.split | Select-Object -Unique)).Count -gt 1 })
    $duplicateLeakage = ($groupLeak.Count -gt 0)
  }
  $summary = [ordered]@{
    archive=(Resolve-Path -LiteralPath $ArchivePath).Path
    archive_sha256=(Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash
    archive_contains_images=($archiveImageEntries.Count -gt 0)
    archive_image_file_count=$archiveImageEntries.Count
    archive_listed_image_count=$archiveImageNames.Count
    archive_label_file_count=$labelEntries.Count
    split_seed=$existingSplitReport.seed
    current_split_counts=@{train=84;val=24;test=12}
    fixed_split_file_hashes=@{train=(Get-FileHash (Join-Path $SplitRoot 'train.txt') -Algorithm SHA256).Hash;val=(Get-FileHash (Join-Path $SplitRoot 'val.txt') -Algorithm SHA256).Hash;test=(Get-FileHash (Join-Path $SplitRoot 'test.txt') -Algorithm SHA256).Hash}
    class_id=0
    class_name='upper_arm'
    polygon_rows=$polygonAudit.Count
    changed_labels=(@($imageAudit | Where-Object label_vs_previous -eq 'changed')).Count
    unchanged_labels=(@($imageAudit | Where-Object label_vs_previous -eq 'unchanged')).Count
    missing_cvat_labels=$missingLabels
    extra_cvat_labels=$extraLabels
    invalid_rows=(@($polygonAudit | Where-Object { $_.issues -and $_.issues -notmatch 'self_crossings_' }).Count)
    rows_with_proper_self_crossings=$crossRows.Count
    proper_self_crossing_pairs=(($crossRows | Measure-Object -Property proper_crossings -Sum).Sum)
    duplicate_group_leakage_in_existing_manifest=$duplicateLeakage
    active_labels_modified=$false
    split_files_modified=$false
    staging_directory=$StageRoot
    report_directory=$ReportRoot
    empty_label_files=(@($imageAudit | Where-Object { $_.issues -match 'empty_cvat_label' } | Select-Object -ExpandProperty filename))
    edge_touching_images=(@($imageAudit | Where-Object { $_.edge_touching_vertices -gt 0 } | Select-Object -ExpandProperty filename))
    training_ready=($missingLabels.Count -eq 0 -and $extraLabels.Count -eq 0 -and $polygonAudit.Count -gt 0 -and $crossRows.Count -eq 0 -and (@($polygonAudit | Where-Object issues).Count -eq 0) -and (@($imageAudit | Where-Object { $_.issues -match 'empty_cvat_label' }).Count -eq 0) -and -not $duplicateLeakage)
    human_visual_review_required=$true
    note='Marker gaps, anatomical boundaries, clothing leakage, and disconnected visual regions require human inspection in the generated overlays. Missing labels do not fall back to previous annotations.'
  }
  $summary | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $ReportRoot 'cvat_export_audit.json') -Encoding utf8
  $imageAudit | Export-Csv -LiteralPath (Join-Path $ReportRoot 'image_label_pair_audit.csv') -NoTypeInformation -Encoding UTF8
  $polygonAudit | Export-Csv -LiteralPath (Join-Path $ReportRoot 'polygon_geometry_audit.csv') -NoTypeInformation -Encoding UTF8
  $summary | ConvertTo-Json -Depth 7
  Write-Output "Wrote train/val/test overlay galleries and CSV reports to $ReportRoot"
  Write-Output "Staged new CVAT labels without changing active labels: $StageRoot"
} finally {
  $zip.Dispose()
}
