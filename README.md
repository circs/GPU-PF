# GPU-PF Repository

Welcome to the GPU-PF repository. This repository is designed to share our research codes that were developed in Karma Group at Northeastern.

## Repository Contents

In the GPU_PF repository, you’ll find:

* Source code files
* Documentation
* Sample datasets
* Parameters that can retrieve the results from our published work 

There are three folders containing separate codes with different models. Up until now (Feb-22-2024), there are four folders/codes, consisting of: 
* The single-GPU code for 2D directional solidification phase field simulation;
* The single-GPU code for 3D directional solidification phase field simulation; 
* The multi-GPU variant of that code that can be computationally capable of simulating a spatially extended system; 

## Getting Started

0. Generating a new SSH key and adding it to the ssh-agent if you haven't, you can refer to the GitHub documentation: [Connect with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
1. Clone the Repo:
You can clone this private repository into your local machine using your GitHub account.
`git clone https://github.com/circs/ICME_NASA.git`
2. Navigate into the Directory:
After successful cloning, navigate into the ICME_NASA directory.
cd GPU-PF


## Running the Simulation

Please review the detailed instructions available in the README.md in each folder for running different solidification simulations.


## License

These projects/codes are free software licensed under the 2-clause BSD Licenses. See the LICENSE.license files in each folder for details. We cordially ask that any published work derived from this code, or utilizing it references the license-mentioned published works.
