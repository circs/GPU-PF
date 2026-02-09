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



/*
 *  In fields-ecolution, there is a struct "InterParam" (iP) and a
 *  class "FieldsEvolution" (fe). One can cite GPU kernel parameters
 *  such as dim3 and cuda stream from iP. The main part of the code
 *  is constructed within fe.
 *
 */

#ifndef FIELDS_EVOLUTION_H_
#define FIELDS_EVOLUTION_H_

// C++ headers
#include "time.h"
#include <curand.h>
#include <curand_kernel.h>
// Project headers
#include "gpu-kernels.h"
#include "gpu-manager.h"
#include "macro.h"
#include "param-manager.h"

struct InterParam // Internal parameters
{

  int HostGrid;

  int SizeHalo;

  cudaStream_t stm[NGPU];

  dim3 Threads[NGPU];
  dim3 Blocks[NGPU];

  dim3 ThreadsYZ;
  dim3 BlocksYZ;
  dim3 OneThread;
  dim3 OneBlock;

};

class FieldsEvolution {

public:
  FieldsEvolution();
  ~FieldsEvolution();

  /****************************
   *          Fields           *
   ****************************/

  REAL *h_Psi;
  REAL *h_U;
  signed char *h_Phase;

  REAL *Psi1[NGPU], *Psi2[NGPU];
  REAL *U1[NGPU], *U2[NGPU];
  REAL *Unext[NGPU], *Ucurr[NGPU];
  REAL *Pnext[NGPU], *Pcurr[NGPU];
  REAL *Phi[NGPU];
  REAL *Xmax1[NGPU], *Xmax2[NGPU];
  signed char *Phase[NGPU], *TempPha[NGPU];
  curandState *States[NGPU];

  Constants *DevCt[NGPU];
  Variables *DevVb[NGPU];

  // TFC pointers
  REAL *h_Temperature; // Host
  // REAL *h_Tsbox;
  REAL *h_Dphidt_Nx;
  REAL *h_Tcurr;
  REAL *h_Tnext;

  REAL *Tsbox[NGPU]; // Device
  REAL *Dphidt[NGPU];
  REAL *Dphidt_Nx[NGPU];
  REAL *Dphidt_Layer[NGPU]; // for Summation of dphi
  REAL *Tcurr[NGPU], *Tnext[NGPU];

  REAL *Buffer;
  signed char *Buffer_Pha;

  /****************************
  ****************************/

  // gpu-manager
  GpuManager *gm;
  GpuManager_Param *dv; // device parameters, such as DEV_Nx
  int SetupGpu(char *GPU_list);

  // param-manager
  ParamManager *pm;
  Constants *Ct;
  Variables *Vb;
  Variables *Vbb;
  int SetupParam(bool ifbreak, int tlimit_src = 9999999,
                 int iter_src = 0, int niter_src = 0);

  // Memory
  int AllocateGpuMem();
  int FreeGpuMem();

  // Internal param
  InterParam iP;
  int InitInterParam();

  // functions for Initilization
  int CopyParamHost2Dev(int order);  // order=3: copy Ct and Vb; 2: only Vb; 1: only Ct
  int CopyFieldsDev2Host(int order); // order=1:Curr , 2:Next)
  int CopyFieldsHost2Dev();

  int InitStates(unsigned long long seed); // for noise

#if (CutZplane != 0)
  int theFirst2DRecord();
#endif
  int theFirstGrainRecord();

  int ExchangeHalo();
  int CalculateTip(); // Calculate tip and update Vb in both host and dev


  int CalculateLenpull(); // for Vp oscillation
  /****************************
  **   Dependent Functions   **
  *****************************/

  int theFirstOutput();

  // Init fields
  int InitfromFunc(unsigned long long seed);
  int InitfromFile(unsigned long long seed);
  int InitfromBreak(unsigned long long seed);

  // Loops
  int MainTimeLoop(clock_t StartTime);
};

#endif
