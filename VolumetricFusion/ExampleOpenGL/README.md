# HeatmapFusion

## Install Instructions
### Install vcpkg in parent directory (using powershell).
```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

### Install C++ dependencies.
Set system environment variable VCPKG_DEFAULT_TRIPLET=x64-windows, or use :x64-windows flag when installing dependencies.
```
.\vcpkg install glad:x64-windows
.\vcpkg install glfw3:x64-windows
```