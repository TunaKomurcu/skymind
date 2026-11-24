@echo off
rem MSVC (x64) ortamini kurup verilen komutu calistirir:  scripts\cpp.cmd cmake -S cpp -B build ...
rem CMake ve Ninja'nin  %USERPROFILE%\.local\bin  altinda (uv tool install cmake ninja) oldugunu varsayar.
setlocal
set "PF86=%ProgramFiles(x86)%"
if "%PF86%"=="" set "PF86=C:\Program Files (x86)"
set "VSWHERE=%PF86%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSTMP=%TEMP%\skymind_vsroot.txt"
if exist "%VSTMP%" del "%VSTMP%"
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%VSTMP%"
set "VSROOT="
if exist "%VSTMP%" set /p VSROOT=<"%VSTMP%"
if not defined VSROOT (
  echo MSVC C++ arac zinciri bulunamadi 1>&2
  exit /b 1
)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul || exit /b 1
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
%*
