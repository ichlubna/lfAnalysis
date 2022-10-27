# Useful scripts for light field analysis and processing
## Main scripts
- addon.py - Blender addon for rendering of LF dataset
- tensorBlending.py - utilizes tensor cores for blending of the LF images and provides comparison to the per-pixel method
- compressionTests.py - experiments with compression of LF data
## General tools 
generalTools folder:
- basher.py - helper script to run Bash commands
- evaluator.py - compares visual quality between images - PSNR, SSIM, WMAF, VIF - using ffmpeg
- lfreader.py - parses and reads a folder with LF data (images like *column_row.jpg*)
## Cuda kernels 
Cuda kernel codes for GPU processing that are used in the other scripts.
