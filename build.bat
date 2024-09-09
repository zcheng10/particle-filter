cmake -DCMAKE_BUILD_TYPE=Release . -B build
:: cd build
:: msbuild particle_filter.sln
cmake --build ./build --config Release

@echo off
:: %~dp0 is the directory of this file
set OCV_BIN=%~dp03rdparty\opencv\x64\vc16\bin
set OCV_PRE=%PATH%
:: echo ocv_pre = %OCV_PRE%;%OCV_BIN%

echo.%PATH% | findstr /I /C:"%OCV_BIN%" 1>nul

if %errorlevel%==0 (
  echo "got zero - %OCV_BIN% already in PATH"
  GOTO :EOF
)

echo "got one - %OCV_BIN% not in PATH" 
set PATH=%OCV_PRE%;%OCV_BIN%