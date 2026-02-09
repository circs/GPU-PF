# 3D Multi-GPU Phase field simulation (treadmill)

This code is a multi-GPU version that can simulate dynamics in directional solidification in a dendritic regime. This code was developed by the Karma group of Northeastern University.

Contact Information:
Center for interdisciplinary research on complex systems
Departments of Physics, Northeastern University
Alain Karma    a.karma@northeastern.edu

The code can be used to reproduce the results in the following papers:

1. TSong, Y. et al. Thermal-field effects on interface dynamics and microstructure selection during alloy directional solidification. Acta Materialia 150, 139-152 (2018)
2. Mota, F. L. et al. Influence of macroscopic interface curvature on dendritic patterns during directional solidification of bulk samples: Experimental and phase-field studies. Acta Materialia 250, 118849 (2023).

## Folder Contents

In the 3D_directional_multi_GPU folder, you’ll find:

* Source code files
* Parameters that can get an example run of SCN-0.96wt%Camphor in a spatially extended simulation domain
* Parameters that can retrieve the results from our published work (to be updated)

## Getting Started

1. Compile the code: `nvcc -arch=sm_70 *.cu -o <your_programme_name>`. You should use a compatible sm flag number associated with your CUDA version for your GPU. Some details can be found here: [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

| Fermi† | Kepler† | Maxwell‡ | Pascal | Volta | Turing | Ampere | Ada | Hopper | Blackwell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sm_20 | sm_30 | sm_50 | sm_60 | sm_70 | sm_75 | sm_80 | sm_89 | sm_90 | ??? |
| sm_35 | sm_52 | sm_61 | |sm_72(Xavier) | | | sm_86 | | sm_90a (Thor) |
|sm_37  |sm_53 |  sm_62| | | | | sm_87 (Orin) | | |


† Fermi and Kepler are deprecated from CUDA 9 and 11 onwards;

‡ Maxwell is deprecated from CUDA 11.6 onwards

2. Run the code: `./<your_programme_name> <ngpu> <tlimit> <iter> <niter>`

Where:

* `<your_programme_name>` is the name of your program.
* `<ngpu>` is the cuda device number of GPUs you want to use, separated by no spaces (e.g., 0123 for 4 GPUs).
* `<tlimit>` is the time limit of the code (use a very large value if no time constraint).
* `<iter>` is the starting iteration number for checkpointing (set to 0 if no time constraint).
* `<niter>` is the total number of iterations to be run (set to 1 if no time constraint).

**To specify the desired number of GPUs, modify the NGPU macro in src/macro.h. Currently it is set to 4**

We also include a batch script to run it on a Slurm-based HPC. Please make sure that you load the specific CUDA module before submitting.

## Post script

The output fields of the example code provided can be visualized with a free software, [Paraview](https://www.paraview.org). 

By default, the output fields will be compressed (controlled by the macro parameter COMPRESS in the code). There will be a total of three tar.gz files: `C_Ny96_D270_G12_k010_V60_dx12_W82.tar.gz`, `C_Ny96_D270_G12_k010_V60_dx12_W82.tar.gz`, and `PF_Ny96_D270_G12_k010_V60_dx12_W82.tar.gz` (the file name could be different base on your choice of parameters and prefix in the code). They contain fields of composition fields, the composition profile, and the phase fields respectively. In most cases, double-clicking can unzip the files (on MacOS or other Linux-based OS). In cases where there is no interactive OS, use the terminal command `tar -xvzf filename.tar.gz` to unzip the files.

After unzipping the tar.gz file, there will be a folder containing a series of VTK files that can be loaded directly in ParaView (if using an interactive OS, double-click to open).

## Result
The top view of the $\phi=0$ contour can be seen in the video below:

https://github.com/circs/ICME_NASA/assets/16184398/9b2b9092-11f0-4392-98c8-62ff1797c23c


The video attached below showcases a 3D view of the output phase field contour of $\phi=0$.


https://github.com/circs/ICME_NASA/assets/16184398/b6d4608b-9394-432a-9d7d-c13a49534b2a

The performance benchmark for the Nvidia V100-SXM2 GPU is attached below:

The code can run on a single GPU without producing any error messages, however, it is designed and intended to run on multiple GPUs (more than 2). Some unphysical behavior may occur when it is run on a single GPU.

![multiGPUBM](https://github.com/circs/ICME_NASA/assets/16184398/5486c22e-bdde-4126-9ea4-f8ab419eb963)



## License

This code is free software licensed under the 2-clause BSD Licenses. See the LICENSE.license files in each folder for details. We cordially ask that any published work derived from this code, or utilizing it references the above-mentioned published works.
