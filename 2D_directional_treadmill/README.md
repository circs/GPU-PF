# 2D Directional solidification (treadmill)

This code is a single-GPU code that can simulate the dynamics of directional solidification in a dendritic regime using the phase field method. This code was developed by the Karma group of Northeastern University.

Contact Information:
Center for interdisciplinary research on complex systems
Departments of Physics, Northeastern University
Alain Karma    a.karma@northeastern.edu

The code can be used to reproduce the results in the following papers:

1. Karma, A. Phase-field formulation for quantitative modeling of alloy solidification. Phys Rev Lett 87, 115701 (2001)
2. Echebarria, B., Folch, R., Karma, A. & Plapp, M. Quantitative phase-field model of alloy solidification. Phys Rev E Stat Nonlin Soft Matter Phys 70, 061604 (2004)



## Folder Contents

In the 2D_directional_GPU folder, you’ll find:

* Source code files
* Postscript folder contains Python script files that can read and visualize the fields or contours of the outputed binary files.
* DSI-R Parameters that can retrieve the results of several stable dendrites starts from a slightly perturbed planar interface.

## Result
The video below depicts the contours of the phase fields for $\phi=0$ 



https://github.com/circs/ICME_NASA/assets/16184398/d666d194-d981-4c2d-b336-700cc833558e




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


## License

This code is free software licensed under the 2-clause BSD Licenses. See the LICENSE.license files in each folder for details. We cordially ask that any published work derived from this code, or utilizing it references the above-mentioned published works.
