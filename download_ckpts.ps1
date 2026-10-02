# Downloads the SAM ViT-H checkpoint (2.4 GB) into asset/ - too large for GitHub.
$dst = Join-Path $PSScriptRoot "asset\sam_vit_h_4b8939.pth"
if (-not (Test-Path $dst)) {
    Invoke-WebRequest "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile $dst
}
