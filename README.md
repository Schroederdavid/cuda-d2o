# HeavyWater (D2O) v1.0 Documentation

## How to use

Launch the test.exe (/build/test.exe) file. The program will automatically run a [CUDA](https://developer.nvidia.com/cuda-toolkit) program to maximize the usage of the currently installed GPU. This program can be terminated with a keyboard interrupt.

The exe file can be renamed, moved, or copied anywhere on any device, provided that the device has an [NVIDIA Graphics card](https://www.nvidia.com/de-de/geforce/graphics-cards/) as well as an [NVIDIA Graphics Driver](https://www.nvidia.com/download/index.aspx).

## Notes

The program currently only supports one GPU, and will terminate for currently unknown reason when executing on multi-GPU systems.

I am not responsible or liable for any damages caused by the usage of my program. I can also not guarantee the successful execution of the program.
In case execution fails, the source code is openly available for editing in /src/test.cu. It can be compiled with [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/).

## Credits

All Programming done by David Schroeder, as an intern of [NVIDIA Corporation](https://www.nvidia.com/de-de/).

© David Schroeder 2023, Berlin, Germany
[david@tobiasschroeder.de](mailto:david@tobiasschroeder.de)
