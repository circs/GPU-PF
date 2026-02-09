/* ---------------------------------------------------------------
*
* This code is developed by CIRCS group of Northeastern University.
*
* Contact Information:
* Center for interdisciplinary research on complex systems
* Departments of Physics, Northeastern University
* Alain Karma    a.karma (at) northeastern.edu
*
* we cordially ask that any published work derived 
* from this code, or utilizing it references the following published works: 
* 
* 1) Song, Y. et al. Thermal-field effects on interface dynamics and microstructure selection during alloy directional solidification. Acta Materialia 150, 139-152 (2018)
* 
* 2) Mota, F. L. et al. Influence of macroscopic interface curvature on dendritic patterns during directional solidification of bulk samples: Experimental and phase-field studies. Acta Materialia 250, 118849 (2023).
--------------------------------------------------------------- */



#ifndef GPU_MANAGER_H_
#define GPU_MANAGER_H_

// C++ headers
// Project headers
#include "macro.h"

struct GpuManager_Param {

  // int NGPU;           // # of GPUs
  int DEVICES[NGPU];  // Device index
  int nx[NGPU];       // Device Nx
  int SizeGrid[NGPU]; // The grid size of devices
  int SizeCopy[NGPU]; // Size for copying from device to host
  int AcmNx[NGPU];    // Accumulated Nx

  int h_Host2Dev[NGPU]; // Host memory address index for transfer data from host to device
  int h_Dev2Host[NGPU]; // Host memory address index for transfer data from device to host
  int d_Dev2Host[NGPU]; // Device memory address index for transfer data from device to host

  int BLOCK_X[NGPU];
  int BLOCK_Y[NGPU];
  int BLOCK_Z[NGPU];

  int BLOCK_Nx; // for the large domain
  int BLOCK_Ny;
  int BLOCK_Nz;

  int maxThreadsPerBlock;
};

class GpuManager {
public:
  GpuManager() {
    dv = new GpuManager_Param;
  }
  ~GpuManager() {
    delete dv;
  }

  GpuManager_Param *dv;
  GpuManager_Param *CopyParam() { return dv; }

  // functions to set Multi-GPU config
  int Setup(char *GPU_list);
  int EnableP2P();
  int DisableP2P();
  int IsUnifiedAddressing();
  int DisplayDeviceProperties();

  int GpuBlock();
};

#endif
