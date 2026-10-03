# Input video

The demo accepts any local video file through `--input`. A convenient openly licensed sample is the Blender Foundation's Sintel trailer, distributed by W3C:

- Source: <https://media.w3.org/2010/05/sintel/trailer.mp4>
- Project and license: <https://durian.blender.org/about/>
- License: Creative Commons Attribution 3.0 (CC BY 3.0)

The repository intentionally does not include the downloaded video. Download it locally with PowerShell:

```powershell
Invoke-WebRequest `
  -Uri "https://media.w3.org/2010/05/sintel/trailer.mp4" `
  -OutFile data/input.mp4
```

Or use any video you are permitted to process:

```text
python src/main.py --input path/to/your/video.mp4 --output outputs/tracked.mp4
```
