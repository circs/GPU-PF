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



#ifndef GPU_KERNELS_H_
#define GPU_KERNELS_H_

// C++ headers
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
// Project headers
#include "gpu-manager.h"
#include "macro.h"
#include "param-manager.h"

__global__ void setup_kernel(unsigned long long seed, curandState *States, int AcmNx);

__global__ void Init(REAL *P1, REAL *P2,
                     REAL *U1, REAL *U2,
                     signed char *Phase,
                     curandState *States,
                     Constants *Ct,
                     Variables *Vb,
                     int AcmNx);


__global__ void Compute_P(REAL *Pcurr, REAL *Pnext,
                          REAL *Ucurr,
                          REAL *F,
                          signed char *Phase,
                          curandState *States,
                          Constants *Ct,
                          Variables *Vb,
                          int DevNx, int AcmNx,
                          REAL *Tsbox);

__global__ void Compute_U(REAL *Pcurr, REAL *Pnext,
                          REAL *Ucurr, REAL *Unext,
                          REAL *F,
                          Constants *Ct,
                          Variables *Vb,
                          int DevNx, int AcmNx);

#if (InnerBCL > 0)
__global__ void Compute_P_BCin(REAL *Pcurr, REAL *Pnext,
                               REAL *Ucurr,
                               REAL *F,
                               signed char *Phase,
                               curandState *state,
                               Constants *Ct,
                               Variables *Vb,
                               int DevNx, int AcmNx,
                               REAL *Tsbox);

__global__ void Compute_U_BCin(REAL *Pcurr, REAL *Pnext,
                               REAL *Ucurr, REAL *Unext,
                               REAL *F,
                               Constants *Ct,
                               Variables *Vb,
                               int DevNx, int AcmNx);
#endif

__global__ void PullBack(REAL *Pcurr, REAL *Pnext,
                         REAL *Ucurr, REAL *Unext,
                         signed char *Phase, signed char *TempPha,
                         Constants *Ct,
                         Variables *Vb,
                         int DevNx, int AcmNx);

__global__ void Boundary(REAL *Field, int DevNx, bool if_last);

__global__ void Boundary_Pha(signed char *Field, int DevNx);

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Processing //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
__global__ void GetXsl_YZ(REAL *P, signed char *Phase,
                          REAL *Xmax1, REAL *Xmax2,
                          int DevNx, int AcmNx);

__global__ void GetXtip(REAL *Xmax1, REAL *Xmax2, Variables *Vb);

#endif
