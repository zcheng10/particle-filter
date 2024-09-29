@echo off

if "%~1"=="" (
    set IMG="test\Image1.jpg"
) else (
    set IMG=%1
)

set RZ_LAMBDA=120
set RZ_THRES=0.98
set RZ_GAUSS=3
build\bin\Release\particle_filter.exe %IMG%