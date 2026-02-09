# 3D single-GPU Phase field simulation (treadmill)

This code is used to simulate dynamics in directional solidification in a dendritic regime. This code was developed by the Karma group of Northeastern University.

Contact Information:
Center for interdisciplinary research on complex systems
Departments of Physics, Northeastern University
Alain Karma    a.karma@northeastern.edu

The code can be used to reproduce the results in the following papers:

1. Clarke, A. J. et al. Microstructure selection in thin-sample directional solidification of an Al-Cu alloy: In situ X-ray imaging and phase-field simulations. Acta Materialia 129, 203-216 (2017) 

## Folder Contents

In the 3D_directional_Single_GPU folder, you’ll find:

* Two folders contain source code files of the two variants of the same single GPU code, the treadmill version and the non-treadmill version.
* For the treadmill version, the default parameter is an example run of SCN-0.46wt%Camphor in a spatially extended simulation domain, initiated from a perturbed planar interface which later forms a stable dendrite, with $V=6 \mu m /s, G=12 K/cm$.
* For the non-treadmill version, where we benchmarked with a publicly available code  [PRISMS-PF/alloy-solidification](https://github.com/prisms-center/phaseField/tree/master/applications/alloySolidification) (The 3D version of that code is not public available at the moment) that can solve the same equations. The default parameter is a case in the benchmark run with PRISMS-PF in DSI-R parameters space. 

## Getting Started

1. Compile the code: `nvcc -arch=sm_70 *.cu -o <your_programme_name>`. You should use a compatible sm flag number associated with your CUDA version for your GPU. Some details can be found here: [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

| Fermi† | Kepler† | Maxwell‡ | Pascal | Volta | Turing | Ampere | Ada | Hopper | Blackwell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sm_20 | sm_30 | sm_50 | sm_60 | sm_70 | sm_75 | sm_80 | sm_89 | sm_90 | ??? |
| sm_35 | sm_52 | sm_61 | |sm_72(Xavier) | | | sm_86 | | sm_90a (Thor) |
|sm_37  |sm_53 |  sm_62| | | | | sm_87 (Orin) | | |


† Fermi and Kepler are deprecated from CUDA 9 and 11 onwards;

‡ Maxwell is deprecated from CUDA 11.6 onwards

2. Run the code: `./<your_programme_name> <your_device_number>`

We also include a batch script to run it on a Slurm-based HPC. Please make sure that you load the specific CUDA module before submitting.


## Post script
The output fields of the example treadmill code provided can be visualized with free software, [Paraview](https://www.paraview.org). 

By default, the output fields will be compressed (controlled by the macro parameter COMPRESS in the code). There will be a total of three tar.gz files: `C_Ny96_D270_G12_k010_V60_dx12_W82.tar.gz`, `C_Ny96_D270_G12_k010_V60_dx12_W82.tar.gz`, and `PF_Ny96_D270_G12_k010_V60_dx12_W82.tar.gz` (the file name could be different base on your choice of parameters and prefix in the code). They contain fields of composition fields, the composition profile, and the phase fields respectively. In most cases, double-clicking can unzip the files (on MacOS or other Linux-based OS). In cases where there is no interactive OS, use the terminal command `tar -xvzf filename.tar.gz` to unzip the files.

After unzipping the tar.gz file, there will be a folder containing a series of VTK files that can be loaded directly in ParaView (if using an interactive OS, double-click to open).

## Result
The top view of the $\phi=0$ contour can be seen in the video below.


https://github.com/circs/ICME_NASA/assets/16184398/231864ed-b17f-4db6-8aff-324a70605635


Here is a 3D view of the $\phi=0$ contour:


https://github.com/circs/ICME_NASA/assets/16184398/1c6973fb-5480-4bd7-822d-3b2f43341630


This example case ran on a single Nvidia V100 GPU for around 15 hours. If you do not wish to run the whole process, you can change the TOTALTIME on line 71 accordingly.


For the non-treadmill code, we have contracted with the main developer of PRISMS-PF and worked together to get a perfect benchmark between the two codes. Here is the comparison of the $\phi=0$ contours at y = 1/3Ly (over the primary dendrite tip) of both codes at two discretizations, dx=1.2W and dx=0.8W:


https://github.com/circs/ICME_NASA/assets/16184398/3ff11664-efd5-4809-92a0-45aa8d8892e0



Here is a 3D side view of the $\phi=0$ contours for different codes at different spatial discretization. (GPU code: $dx=1.2W, dt=5e-4\tau$, $dx=0.8W, dt=5e-4\tau$; PRISMS-PF-3D: $dx=1.2W, dt=5e-4\tau$, $dx=0.8W, dt=2e-4\tau$, the time step is slightly different between each frame for GPU code and PRISMS-PF, thus caused the offset later)

https://github.com/circs/ICME_NASA/assets/16184398/aeb35fe0-bdac-4bde-83cf-9e750303f860

## License

This code is free software licensed under the 2-clause BSD Licenses. See the LICENSE.license files in each folder for details. We cordially ask that any published work derived from this code, or utilizing it references the above-mentioned published works.
