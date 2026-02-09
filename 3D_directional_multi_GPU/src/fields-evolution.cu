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

#include "fields-evolution.h"
// C++ & CUDA headers
#include "math.h"
#include "time.h"
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
// Project headers
#include "gpu-kernels.h"
#include "gpu-manager.h"
#include "io-functions.h"
#include "macro.h"
#include "param-manager.h"

/****************************
*****************************
*  Independent Functions  	*
*  do not cite functions 	*
*  within the same class	*
*****************************
*****************************/

FieldsEvolution::FieldsEvolution()
    : dv(NULL), Ct(NULL), Vb(NULL) {

  gm = new GpuManager();
  pm = new ParamManager();

  Vbb = new Variables();

  // Host memory allocation: pinned memory. If use cudaHostAlloc here, device 0 will be taken even if not use it
  // cudaHostAlloc((void **)&h_Psi, (Nx+2)*(Ny+2)*(Nz+2)*sizeof(REAL), 0);
  // cudaHostAlloc((void **)&h_U, (Nx+2)*(Ny+2)*(Nz+2)*sizeof(REAL), 0);
  // cudaHostAlloc((void **)&h_Phase, (Nx+2)*(Ny+2)*(Nz+2)*sizeof(signed char), 0);

  h_Psi = (REAL *)malloc((Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(REAL));
  h_U = (REAL *)malloc((Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(REAL));
  h_Phase = (signed char *)malloc((Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(signed char));
}

FieldsEvolution::~FieldsEvolution() {

  delete gm;
  delete pm;

  delete Vbb;

  // free pinned host memory
  // cudaFreeHost(h_Psi);
  // cudaFreeHost(h_U);
  // cudaFreeHost(h_Phase);

  free(h_Psi);
  free(h_U);
  free(h_Phase);
}

int FieldsEvolution::SetupGpu(char *GPU_list) {
  gm->Setup(GPU_list);
  gm->DisplayDeviceProperties();
  gm->EnableP2P();
  gm->IsUnifiedAddressing();
  gm->GpuBlock();

  dv = gm->CopyParam();

  return 0;
}

int FieldsEvolution::SetupParam(bool ifbreak, int tlimit_src,
                                int iter_src, int niter_src) {
  pm->InitConstants();
  pm->InitVariables(tlimit_src,
                    iter_src, niter_src);

  Ct = pm->CopyConstants();
  Vb = pm->CopyVariables();

  // if starts from break, then copy variables from binary file
  if (ifbreak) {
    memcpy(Vbb, Vb, sizeof(Variables)); // Vbb is the temp copy of Vb
    ReadParam(Vb, 0);

    // Make IterOutFields etc. changeable after break
    Vb->IterOutFields = Vbb->IterOutFields;
    Vb->IterOutSvg = Vbb->IterOutSvg;
    Vb->IterOutTip = Vbb->IterOutTip;

    // Make tlimit changeavle after break
    Vb->tlimit = Vbb->tlimit;
  }

  return 0;
}

int FieldsEvolution::AllocateGpuMem() {

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);

    cudaMalloc((void **)&Psi1[i], dv->SizeGrid[i] * sizeof(REAL));
    cudaMalloc((void **)&Psi2[i], dv->SizeGrid[i] * sizeof(REAL));

    cudaMalloc((void **)&U1[i], dv->SizeGrid[i] * sizeof(REAL));
    cudaMalloc((void **)&U2[i], dv->SizeGrid[i] * sizeof(REAL));

    Ucurr[i] = U1[i];
    Unext[i] = U2[i];

    Pcurr[i] = Psi1[i];
    Pnext[i] = Psi2[i];

    cudaMalloc((void **)&Phi[i], dv->SizeGrid[i] * sizeof(REAL));

    cudaMalloc((void **)&Xmax1[i], (Ny + 2) * (Nz + 2) * sizeof(REAL));
    cudaMalloc((void **)&Xmax2[i], (Ny + 2) * (Nz + 2) * sizeof(REAL));

    cudaMalloc((void **)&Phase[i], dv->SizeGrid[i] * sizeof(signed char));
    cudaMalloc((void **)&TempPha[i], dv->SizeGrid[i] * sizeof(signed char));

    cudaMalloc((void **)&States[i], dv->SizeGrid[i] * sizeof(curandState));

    cudaMalloc((void **)&DevCt[i], sizeof(Constants));
    cudaMalloc((void **)&DevVb[i], sizeof(Variables));
  }

  return 0;
}

int FieldsEvolution::FreeGpuMem() {

  gm->DisableP2P();
  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);

    cudaFree(Psi1[i]);
    cudaFree(Psi2[i]);

    cudaFree(U1[i]);
    cudaFree(U2[i]);

    cudaFree(Phi[i]);
    cudaFree(Xmax1[i]);
    cudaFree(Xmax2[i]);

    cudaFree(Phase[i]);
    cudaFree(TempPha[i]);

    cudaFree(States[i]);

    cudaFree(DevCt[i]);
    cudaFree(DevVb[i]);
  }

  return 0;
}

int FieldsEvolution::InitInterParam() {
  iP.HostGrid = (Nx + 2) * (Ny + 2) * (Nz + 2);

  iP.SizeHalo = (Ny + 2) * (Nz + 2);

  // Set CUDA stream
  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    cudaStreamCreate(&iP.stm[i]);
  }

  // Set dim3
  for (int i = 0; i < NGPU; i++) {
    iP.Threads[i] = dim3(dv->BLOCK_X[i], dv->BLOCK_Y[i], dv->BLOCK_Z[i]);
    iP.Blocks[i] = dim3((dv->nx[i] + 2) / dv->BLOCK_X[i], (Ny + 2) / dv->BLOCK_Y[i], (Nz + 2) / dv->BLOCK_Z[i]);
  }

  iP.ThreadsYZ = dim3(dv->BLOCK_Y[0], dv->BLOCK_Z[0]);
  iP.BlocksYZ = dim3((Ny + 2) / dv->BLOCK_Y[0], (Nz + 2) / dv->BLOCK_Z[0]);
  iP.OneThread = dim3(1);
  iP.OneBlock = dim3(1);

  return 0;
}

int FieldsEvolution::CopyParamHost2Dev(int order) // order=3: copy Ct and Vb; 2: only Vb; 1: only Ct
{
  bool copyCt, copyVb;
  if (order == 1) {
    copyCt = true;
    copyVb = false;
  } else if (order == 2) {
    copyCt = false;
    copyVb = true;
  } else if (order == 3) {
    copyCt = true;
    copyVb = true;
  }

  for (int i = 0; i < NGPU; i++) {
    if (copyCt) {
      cudaMemcpyAsync(DevCt[i], Ct, sizeof(Constants), cudaMemcpyDefault, iP.stm[i]);
    }
    if (copyVb) {
      cudaMemcpyAsync(DevVb[i], Vb, sizeof(Variables), cudaMemcpyDefault, iP.stm[i]);
    }
  }

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    cudaStreamSynchronize(iP.stm[i]);
  }

  return 0;
}

int FieldsEvolution::CopyFieldsDev2Host(int order) // order==1:Curr , 2:Next
{

  for (int i = 0; i < NGPU; i++) {
    if (order == 1) // Curr
    {
      cudaMemcpyAsync(h_Psi + dv->h_Dev2Host[i], Pcurr[i] + dv->d_Dev2Host[i], dv->SizeCopy[i] * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
      cudaMemcpyAsync(h_U + dv->h_Dev2Host[i], Ucurr[i] + dv->d_Dev2Host[i], dv->SizeCopy[i] * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
      cudaMemcpyAsync(h_Phase + dv->h_Dev2Host[i], Phase[i] + dv->d_Dev2Host[i], dv->SizeCopy[i] * sizeof(signed char), cudaMemcpyDefault, iP.stm[i]);
    } else if (order == 2) // Next
    {
      cudaMemcpyAsync(h_Psi + dv->h_Dev2Host[i], Pnext[i] + dv->d_Dev2Host[i], dv->SizeCopy[i] * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
      cudaMemcpyAsync(h_U + dv->h_Dev2Host[i], Unext[i] + dv->d_Dev2Host[i], dv->SizeCopy[i] * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
      cudaMemcpyAsync(h_Phase + dv->h_Dev2Host[i], Phase[i] + dv->d_Dev2Host[i], dv->SizeCopy[i] * sizeof(signed char), cudaMemcpyDefault, iP.stm[i]);
    }
  }

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    cudaStreamSynchronize(iP.stm[i]);
  }

  return 0;
}

int FieldsEvolution::CopyFieldsHost2Dev() {
  for (int i = 0; i < NGPU; i++) {
    cudaMemcpyAsync(Pnext[i], h_Psi + dv->h_Host2Dev[i], dv->SizeGrid[i] * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
    cudaMemcpyAsync(Unext[i], h_U + dv->h_Host2Dev[i], dv->SizeGrid[i] * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
    cudaMemcpyAsync(Phase[i], h_Phase + dv->h_Host2Dev[i], dv->SizeGrid[i] * sizeof(signed char), cudaMemcpyDefault, iP.stm[i]);

    cudaMemcpyAsync(Pcurr[i], Pnext[i], dv->SizeGrid[i] * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
    cudaMemcpyAsync(Ucurr[i], Unext[i], dv->SizeGrid[i] * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
    cudaMemcpyAsync(TempPha[i], Phase[i], dv->SizeGrid[i] * sizeof(signed char), cudaMemcpyDefault, iP.stm[i]);
  }

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    cudaStreamSynchronize(iP.stm[i]);
  }

  return 0;
}

int FieldsEvolution::InitStates(unsigned long long seed) {

  for (int i = 0; i < NGPU; i++) // Setup noise
  {
    cudaSetDevice(dv->DEVICES[i]);

    setup_kernel<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(seed, States[i], dv->AcmNx[i]);
  }

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    cudaStreamSynchronize(iP.stm[i]);
  }

  return 0;
}

#if (CutZplane != 0)
int FieldsEvolution::theFirst2DRecord() {
  // int d2indx=1;

  printf("time = 0 s\t"); // current time

  // to record when the ouput is written
  FILE *D2IndxT = fopen(Ct->D2Time, "w");
  fprintf(D2IndxT, "Index \tTime[s] \txoffs[dx] \n");
  fprintf(D2IndxT, "%d \t%g \t%g \n", 0, 0., Vb->xoffs / dx);
  fclose(D2IndxT);
  //

  WriteZ2Dfield(h_Psi, h_U, h_Phase,
                *Ct, *Vb,
                Zloc, 0, Ct->X2DMAX);

#if (CutZplane > 0)                         // need to be updated ???
  int d2outiter = (int)(niter / CutZplane); // output 2D field every d2outiter iterations.
#endif

  return 0;
}
#endif

int FieldsEvolution::theFirstGrainRecord() {

  // make a file for grain tips informations
  FILE *OutFileGBTips = fopen(Vb->FileNameGBTips, "w");

  fprintf(OutFileGBTips, "Time[s] \t");      // 1
  fprintf(OutFileGBTips, "x1_{Tip}[dx] \t"); // 2
  fprintf(OutFileGBTips, "Delta1 \t");       // 3
  fprintf(OutFileGBTips, "Omega1 \t");       // 4
  fprintf(OutFileGBTips, "x2_{Tip}[dx] \t"); // 5
  fprintf(OutFileGBTips, "Delta2 \t");       // 6
  fprintf(OutFileGBTips, "Omega2 \t");       // 7
  fprintf(OutFileGBTips, "\n");
  fclose(OutFileGBTips);

  // make a file for a GB location
  FILE *OutFileGBL = fopen(Vb->FileNameGBL, "w");

  fprintf(OutFileGBL, "# dx_microns=%g \n", Ct->dx_microns);                                     // 0
  fprintf(OutFileGBL, "# Y_max = %d [dx] = %g [microns] \n", Ny + 1, (Ny + 1) * Ct->dx_microns); // 0
  fprintf(OutFileGBL, "# Grain1 : (alpha,beta) = (%g,%g) [degrees]  \n", AngleA1, AngleB1);      // 0
  fprintf(OutFileGBL, "# Grain2 : (alpha,beta) = (%g,%g) [degrees]  \n", AngleA2, AngleB2);      // 0

  // first line
  fprintf(OutFileGBL, "Time[s] \t");        // 1
  fprintf(OutFileGBL, "x[microns] \t");     // 2
  fprintf(OutFileGBL, "GB1 [microns] \t");  // 3
  fprintf(OutFileGBL, "GB2 [microns] \t");  // 4
  fprintf(OutFileGBL, "Ymax [microns] \t"); // 4
  fprintf(OutFileGBL, "\n");
  fclose(OutFileGBL);

  return 0;
}

int FieldsEvolution::ExchangeHalo() {
  for (int i = 0; i < NGPU - 1; i++) // Exchange halo
  {
    cudaMemcpyAsync(Pnext[i] + (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2), Pnext[i + 1] + (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
    cudaMemcpyAsync(Pnext[i + 1], Pnext[i] + dv->nx[i] * (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);

    cudaMemcpyAsync(Unext[i] + (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2), Unext[i + 1] + (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
    cudaMemcpyAsync(Unext[i + 1], Unext[i] + dv->nx[i] * (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);

    cudaMemcpyAsync(Phase[i] + (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2), Phase[i + 1] + (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(signed char), cudaMemcpyDefault, iP.stm[i]);
    cudaMemcpyAsync(Phase[i + 1], Phase[i] + dv->nx[i] * (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(signed char), cudaMemcpyDefault, iP.stm[i]);
  }

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    cudaStreamSynchronize(iP.stm[i]);
  }

  return 0;
}

int FieldsEvolution::CalculateTip() // Calculate tip and update Vb in both host and dev
                                    // Vb in host will be overwrited by this function
{

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    GetXsl_YZ<<<iP.BlocksYZ, iP.ThreadsYZ, 0, iP.stm[i]>>>(Pcurr[i], Phase[i],
                                                           Xmax1[i], Xmax2[i],
                                                           dv->nx[i], dv->AcmNx[i]);
    GetXtip<<<iP.OneBlock, iP.OneThread, 0, iP.stm[i]>>>(Xmax1[i], Xmax2[i], DevVb[i]);
  }

  REAL xtip1_Max = 0.;
  REAL xtip2_Max = 0.;
  int xtip1_Dev = 0;
  int xtip2_Dev = 0;
  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    cudaStreamSynchronize(iP.stm[i]);
    cudaMemcpy(Vbb, DevVb[i], sizeof(Variables), cudaMemcpyDefault);
    if (Vbb->xtip1 > xtip1_Max) {
      xtip1_Max = Vbb->xtip1;
      xtip1_Dev = i;
    } // Find the device in which the tip1 is
    if (Vbb->xtip2 > xtip2_Max) {
      xtip2_Max = Vbb->xtip2;
      xtip2_Dev = i;
    } // Find the device in which the tip2 is
  }

  // cudaSetDevice(DEVICES[xtip_Dev]);
  // GetRtip<<<NumOneBlock,SizeOneBlock>>>(Xmax[xtip_Dev],Psicurr[xtip_Dev],Parameters[xtip_Dev]); //***Need to find the reason. Find R tip

  cudaMemcpy(Vb, DevVb[xtip1_Dev], sizeof(Variables), cudaMemcpyDefault);
  cudaMemcpy(Vbb, DevVb[xtip2_Dev], sizeof(Variables), cudaMemcpyDefault); // xtip2 might be in different device
  Vb->xtip2 = Vbb->xtip2;
  Vb->ytip2 = Vbb->ytip2;
  Vb->ztip2 = Vbb->ztip2;

  if (Vb->xtip1 > Vb->xtip2) // update xtip in host
  {
    Vb->xtip = Vb->xtip1;
    Vb->ytip = Vb->ytip1;
    Vb->ztip = Vb->ztip1;
  } else {
    Vb->xtip = Vb->xtip2;
    Vb->ytip = Vb->ytip2;
    Vb->ztip = Vb->ztip2;
  }

  CopyParamHost2Dev(2); // Updata parameters in devices

  return 0;
}

int FieldsEvolution::CalculateLenpull() {

  REAL vptemp = Ct->Vp;

  // update pulling velocity for the next time step

#if (OSC_Velocity != WITHOUT)
  if (Vb->iter * Ct->dt * Ct->Tau0_sec > OSC_t0) {
    Vb->OSCVamp = 0.;
    // printf("Vb->iter = %i, OSC_t0 = %g, time = %g\n",Vb->iter, OSC_t0,Vb->iter*Ct->dt*Ct->Tau0_sec);

#if (OSC_Velocity == CONST_V)
    {
      Vb->OSCVamp = Ct->OSCVamp0;
    }
#endif

#if (OSC_Velocity == LINEAR)
    {
      Vb->OSCVamp = Ct->OSCVamp0 / (TOTALTIME - OSC_t0) * Ct->Tau0_sec * (Vb->iter * Ct->dt - OSC_t0 / Ct->Tau0_sec);
    }
#endif

#if (OSC_Velocity == SINOSC)
    {
      Vb->OSCVamp = Ct->OSCVamp0 * sin(2. * PI * (Vb->iter * Ct->dt * Ct->Tau0_sec - OSC_t0) / OSC_Period);
    }
#endif

#if (OSC_Velocity == STEPLIKE)
    {
      REAL OSCTime = Vb->iter * Ct->dt * Ct->Tau0_sec - OSC_t0; // time after turning on the Vp oscillation

      // +OSCVamp0 at the first half period; change "<="" to ">" for -OSCVamp0 at the first half
      if ((OSCTime - (Vb->OSCNstep - 1) * OSC_Period) <= 0.5 * OSC_Period) {
#if (if_Vamp_Up)
        Vb->OSCVamp = Ct->OSCVamp0 * OSC_Vamp_Up;
#else
        Vb->OSCVamp = Ct->OSCVamp0;
#endif
      } else {
        Vb->OSCVamp = -Ct->OSCVamp0;
      }

      if (OSCTime > Vb->OSCNstep * OSC_Period) {
        Vb->OSCNstep = Vb->OSCNstep + 1;
      }
    }
#endif

#if (OSC_tk != 0)
    if (Vb->iter * Ct->dt * Ct->Tau0_sec > OSC_tk) {
      Vb->OSCVamp = 0.;
    }
#endif

    vptemp += Vb->OSCVamp;
  }
#endif

  Vb->Lenpull += Ct->dt * vptemp;

  return 0;
}

/****************************
*****************************
*    Dependent Functions   	*
*	 Cite functions within  *
* 	 the same class		    *
*****************************
*****************************/

int FieldsEvolution::theFirstOutput() {
  InitTipFile(Ct->TipFileName);
  OutputTipFile(Ct, Vb); // output tipfile at t=0

  InitIndexFile(Ct, Vb);

  // 2D record
#if (CutZplane != 0)
  theFirst2DRecord();
#endif
  // Grain record
  theFirstGrainRecord();

  OutputParameters(*dv, *Ct, *Vb);

  // WriteFields(h_Psi, h_U, h_Phase, *Ct, *Vb, 0, 1);
  WriteFields(h_Psi, h_U, h_Phase, h_Temperature, *Ct, *Vb, 0, 0);

  return 0;
}

int FieldsEvolution::InitfromFunc(unsigned long long seed) {

  printf("\nInitializing fields...\n");

  CopyParamHost2Dev(3);

  InitStates(seed);

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);

    Init<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Psi1[i], Psi2[i],
                                                        U1[i], U2[i],
                                                        Phase[i],
                                                        States[i],
                                                        DevCt[i],
                                                        DevVb[i],
                                                        dv->AcmNx[i]);
  }

  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(dv->DEVICES[i]);
    cudaStreamSynchronize(iP.stm[i]);
  }

  CopyFieldsDev2Host(2); //!!! temp

  return 0;
}

int FieldsEvolution::InitfromFile(unsigned long long seed) // if InitfromFile, need to get Delta from the last run
{
  printf("\nInitializing fields from files...\n");

  // Vbb->iter = Vb->iter;
  // Vbb->niter = Vb->niter;
  // Vbb->xint = Vbb->xtip*dx;
  // Vbb->x0 = Vbb->xint-(1.-Vbb->Delta)*Vbb->lT;
  // Vbb->xoffs = Vb->xoffs;
  // Vbb->xprev = Vb->xprev;

  // memcpy(Vb, Vbb, sizeof(Variables));

  InitStates(seed);

#if (INITfromFILE == fromDAT)
  IO_InitfromFile_DAT(h_Psi, h_U, h_Phase, h_Temperature, *Ct);
  CopyFieldsHost2Dev();

  ReadParam(Vbb, 1);

  Vb->xtip = Vbb->xtip;
  Vb->xint = Vb->xtip * dx;
  Vb->Delta = Vbb->Delta;
  Vb->x0 = Vb->xint - (1. - Vb->Delta) * Vbb->lT;
  Vb->Delta = 1. - (Vb->xtip * dx - Vb->x0 + Vb->xoffs - Ct->Vp * Ct->dt * Vb->iter) / Vb->lT;

#endif

#if (INITfromFILE == fromVTK)
  // Delta and d0trace (if TFC) are updated in IO_InitFromFile_VTK
  IO_InitFromFile_VTK(h_Psi, h_U, h_Phase, h_Temperature, *Ct, Vb); // At this point (1-Delta0) is stored into Vb->x0

  CopyFieldsHost2Dev();

  CopyParamHost2Dev(3); // copy both Ct and Vb

  CalculateTip(); // Updated Vb in both host and dev

  Vb->xint = Vb->xtip * dx;
  Vb->x0 = Vb->xint - Vb->x0 * Vb->lT;
  Vb->Delta = 1. - (Vb->xtip * dx - Vb->x0 + Vb->xoffs - Ct->Vp * Ct->dt * Vb->iter) / Vb->lT;

#endif

  CopyParamHost2Dev(3); // copy both Ct and Vb

  CalculateTip(); // Updated Vb in both host and dev

  return 0;
}

int FieldsEvolution::InitfromBreak(unsigned long long seed) {
  printf("\nRead binary fields from break. (iter=%d) \n", Vb->iter);

  CopyParamHost2Dev(3); // copy both Ct and Vb

  InitStates(seed);

  IO_InitfromBreak(h_Psi, h_U, h_Phase, h_Temperature, *Ct);

  CopyFieldsHost2Dev();

  CalculateTip(); // Updated Vb in both host and dev

  return 0;
}

int FieldsEvolution::MainTimeLoop(clock_t StartTime) {

  // for break
  clock_t tBegin = StartTime;
  clock_t tNow;
  REAL CompTime = 0.;
  printf("\n The time limit for this run is %f hour(s).\n", Vb->tlimit / 3600.);

  // Vb->iter = 0;
  // Vb->niter = 8000;

  CopyParamHost2Dev(2); // copy only Vb

  while (Vb->iter < Vb->niter) {

    Vb->iter = Vb->iter + 1;
    CalculateLenpull(); // update Vb->Lenpull and Vb->OSCVamp

    CopyParamHost2Dev(2); // Param synchronized
                          // Between two synchronized points, can only change one type of Param.

#if (InnerBCL > 0) //
    if (Vb->xoffs < (1 + InnerBCL / 10.) * POSITION_0 / Ct->W_microns) {

      //////////////////////
      // Compute Psi next //
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);

        Compute_P_BCin<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Pcurr[i], Pnext[i],
                                                                      Ucurr[i],
                                                                      Phi[i],
                                                                      Phase[i],
                                                                      States[i],
                                                                      DevCt[i], DevVb[i],
                                                                      dv->nx[i], dv->AcmNx[i],
                                                                      Tsbox[i]);

        Boundary<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Pnext[i], dv->nx[i], i == (NGPU - 1));
        Boundary_Pha<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Phase[i], dv->nx[i]);
      }

      ////////////////////////////
      // Exchange Psi next halo // (w/o boundary at first). Compute_U needs this updated information
      for (int i = 0; i < NGPU; i++) {
        cudaStreamSynchronize(iP.stm[i]);
      }
      for (int i = 0; i < NGPU - 1; i++) {
        cudaMemcpyAsync(Pnext[i] + (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2), Pnext[i + 1] + (Ny + 2) * (Nz + 2),
                        iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);

        cudaMemcpyAsync(Pnext[i + 1], Pnext[i] + dv->nx[i] * (Ny + 2) * (Nz + 2),
                        iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
      }

      ////////////////////
      // Compute U next //
      for (int i = 0; i < NGPU; i++) {
        cudaStreamSynchronize(iP.stm[i]);
      }
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);

#if (LatentH > 0) // Updating Dphit, since no T calculation in Compute_U, can put it in this loop
        Dphit<<<iP.BlocksX[i], iP.ThreadsX[i], 0, iP.stm[i]>>>(Pcurr[i], Pnext[i], Dphidt[i], DevCt[i], dv->nx[i]);
#endif

        Compute_U_BCin<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Pcurr[i], Pnext[i],
                                                                      Ucurr[i], Unext[i],
                                                                      Phi[i],
                                                                      DevCt[i],
                                                                      DevVb[i],
                                                                      dv->nx[i], dv->AcmNx[i]);

        Boundary<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Unext[i], dv->nx[i], false);
      }

      ///////////////////
      // Exchange halo //
      for (int i = 0; i < NGPU; i++) {
        cudaStreamSynchronize(iP.stm[i]);
      }
      ExchangeHalo();
    } else
#endif // end of #if(InnerBCL>0)

    {

      //////////////////////
      // Compute Psi next //
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);

        Compute_P<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Pcurr[i], Pnext[i],
                                                                 Ucurr[i],
                                                                 Phi[i],
                                                                 Phase[i],
                                                                 States[i],
                                                                 DevCt[i], DevVb[i],
                                                                 dv->nx[i], dv->AcmNx[i],
                                                                 Tsbox[i]);

        Boundary<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Pnext[i], dv->nx[i], i == (NGPU - 1));
        Boundary_Pha<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Phase[i], dv->nx[i]);
      }

      ////////////////////////////
      // Exchange Psi next halo // (w/o boundary at first). Compute_U needs this updated information
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);
        cudaStreamSynchronize(iP.stm[i]);
      }
      for (int i = 0; i < NGPU - 1; i++) {
        cudaMemcpyAsync(Pnext[i] + (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2), Pnext[i + 1] + (Ny + 2) * (Nz + 2),
                        iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);

        cudaMemcpyAsync(Pnext[i + 1], Pnext[i] + dv->nx[i] * (Ny + 2) * (Nz + 2),
                        iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
      }

      ////////////////////
      // Compute U next //
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);
        cudaStreamSynchronize(iP.stm[i]);
      }
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);

#if (LatentH > 0)                                                                                                                                                // Updating Dphit, since no T calculation in Compute_U, can put it in this loop
        Dphit<<<iP.BlocksSum[i], iP.ThreadsSum[i], iP.smemSize, iP.stm[i]>>>(Pcurr[i], Pnext[i], Dphidt_Layer[i], DevCt[i], dv->nx[i], iP.nLayer, iP.LayerSize); //
        Dphit2<<<iP.BlocksX[i], iP.ThreadsX[i], 0, iP.stm[i]>>>(Dphidt[i], Dphidt_Layer[i], dv->nx[i], iP.nLayer);
#endif

        Compute_U<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Pcurr[i], Pnext[i],
                                                                 Ucurr[i], Unext[i],
                                                                 Phi[i],
                                                                 DevCt[i],
                                                                 DevVb[i],
                                                                 dv->nx[i], dv->AcmNx[i]);

        Boundary<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Unext[i], dv->nx[i], false);
      }

      ///////////////////
      // Exchange halo //
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);
        cudaStreamSynchronize(iP.stm[i]);
      }
      ExchangeHalo();
    } // end of if (Vb->xoffs < (1+InnerBCL/10.)*POSITION_0/Ct->W_microns)

    // Swap poniter
    for (int i = 0; i < NGPU; i++) // Exchange pointer
    {
      cudaSetDevice(dv->DEVICES[i]);

      Buffer = Pnext[i];
      Pnext[i] = Pcurr[i];
      Pcurr[i] = Buffer;
      Buffer = Unext[i];
      Unext[i] = Ucurr[i];
      Ucurr[i] = Buffer;
    }

    ////////////////////////////
    //     Pulling Back       //
    ////////////////////////////
#if (PBFRAME == LAB) // If the frame is in LAB, then the tip position will not be updated
    int off = 1;
#if (OSC_Velocity == WITHOUT)
    if ((Ct->Vp * Ct->dt * Vb->iter - Vb->xoffs) > dx)
#else
    if ((Vb->Lenpull - Vb->xoffs) > dx)
#endif
#endif
#if (PBFRAME == TIP)

      CalculateTip();                          // Param synchronized
    int off = (int)(Vb->xtip - Vb->xint / dx); // Pull back
    if (off)
#endif
    {
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);

        // Updated xoffs in devices
        PullBack<<<iP.Blocks[i], iP.Threads[i], 0, iP.stm[i]>>>(Pcurr[i], Pnext[i],
                                                                Ucurr[i], Unext[i],
                                                                Phase[i], TempPha[i],
                                                                DevCt[i],
                                                                DevVb[i],
                                                                dv->nx[i], dv->AcmNx[i]);
      }

      for (int i = 0; i < NGPU; i++) // Exchange pointer
      {
        cudaSetDevice(dv->DEVICES[i]);
        cudaStreamSynchronize(iP.stm[i]);

        Buffer = Pnext[i];
        Pnext[i] = Pcurr[i];
        Pcurr[i] = Buffer;
        Buffer = Unext[i];
        Unext[i] = Ucurr[i];
        Ucurr[i] = Buffer;

        Buffer_Pha = TempPha[i];
        TempPha[i] = Phase[i];
        Phase[i] = Buffer_Pha;
        // Phase[i] = TempPha[i]; // *** why not swap?
      }

      for (int i = 0; i < NGPU - 1; i++) // Exchange halo
      {
        cudaMemcpyAsync(Pcurr[i] + (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2), Pcurr[i + 1] + (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
        cudaMemcpyAsync(Ucurr[i] + (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2), Ucurr[i + 1] + (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(REAL), cudaMemcpyDefault, iP.stm[i]);
        cudaMemcpyAsync(Phase[i] + (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2), Phase[i + 1] + (Ny + 2) * (Nz + 2), iP.SizeHalo * sizeof(signed char), cudaMemcpyDefault, iP.stm[i]);
      }
      for (int i = 0; i < NGPU; i++) {
        cudaSetDevice(dv->DEVICES[i]);
        cudaStreamSynchronize(iP.stm[i]);
      }

      Vb->xoffs += off * dx; // Updated xoffs in host
      // Npull=Npull+1;

      // output 2D or NOUTFIELDS
      bool ifCutzplane = false;
      bool ifNOUTFIELDS = false;
#if (CutZplane < 0)
      ifCutzplane = (Vb->xoffs > Vb->numoutput_z * POSITION_0 / Ct->W_microns / (-1. * CutZplane));
#endif
#if (NOUTFIELDS < 0)
      ifNOUTFIELDS = (Vb->xoffs > Vb->numoutput_f * POSITION_0 / Ct->W_microns / (-1. * NOUTFIELDS));
#endif

      if (ifCutzplane || ifNOUTFIELDS) {
        CopyFieldsDev2Host(1); // copy mem only once for both CutZplane and NOUTFIELDS

#if (CutZplane < 0)
        if (ifCutzplane) {
          // to record when the ouput is written
          FILE *D2IndxT = fopen(Ct->D2Time, "a");
          fprintf(D2IndxT, "%d \t%g \t%g \n", Vb->numoutput_z, Vb->iter * Ct->dt * Ct->Tau0_sec, Vb->xoffs / dx);
          fclose(D2IndxT);

          // write field (2D)
          WriteZ2Dfield(h_Psi, h_U, h_Phase,
                        *Ct, *Vb,
                        Zloc, Vb->numoutput_z, Ct->X2DMAX);

          // output every Lsol (2D)
          Vb->numoutput_z = Vb->numoutput_z + 1; // update numoutput
        }
#endif

#if (NOUTFIELDS < 0)
        if (ifNOUTFIELDS) {
          WriteFields(h_Psi, h_U, h_Phase, h_Temperature, *Ct, *Vb, Vb->numoutput_f, 0);
          OutputIndexFile(Ct, Vb);
          Vb->numoutput_f = Vb->numoutput_f + 1;
        }
#endif
      }

    } // End of pulling back

    //////////////////////
    //  Output Tip data //
    //////////////////////

    if (Vb->iter % Vb->IterOutTip == 0 || Vb->iter >= Vb->niter) // update tip data at the final step
    {
#if (OSC_Velocity == WITHOUT)
      Vb->Delta = 1. - (Vb->xtip * dx - Vb->x0 + Vb->xoffs - Ct->Vp * Ct->dt * Vb->iter) / Vb->lT;
#else
      Vb->Delta = 1. - (Vb->xtip * dx - Vb->x0 + Vb->xoffs - Vb->Lenpull) / Vb->lT;
#endif
      Vb->Omega = 1. / (1. - Ct->kcoeff) * (1. - Ct->kcoeff / (Ct->kcoeff + (1. - Ct->kcoeff) * Vb->Delta));
      Vb->Vel = (Vb->xprev > 0.) ? (Vb->xtip * dx + Vb->xoffs - Vb->xprev) / (Vb->IterOutTip * Ct->dt) : 0.;
      Vb->xprev = Vb->xtip * dx + Vb->xoffs; // update xprev after Vel is calculated

      // Numerical test on Delta
#if (Thermaltau == 0 && IQY != 0 && IQZ != 0 && LateralG == 0)
      if (Vb->Delta < 0. || Vb->Delta > 1.) {
        Vb->iter = Vb->niter * 2; // end this run
        printf("Looks like something went wrong.... (review your parameters, dx, W/d0, etc.).\n");
      }
#endif

      OutputTipFile(Ct, Vb);

      //================================================
      // search Grain Boundary
      //================================================

#if (OSC_Velocity == WITHOUT)
      Vb->Delta1 = 1. - (Vb->xtip1 * dx - Vb->x0 + Vb->xoffs - Ct->Vp * Ct->dt * Vb->iter) / Vb->lT;
      Vb->Delta2 = 1. - (Vb->xtip2 * dx - Vb->x0 + Vb->xoffs - Ct->Vp * Ct->dt * Vb->iter) / Vb->lT;
#else
      Vb->Delta1 = 1. - (Vb->xtip1 * dx - Vb->x0 + Vb->xoffs - Vb->Lenpull) / Vb->lT;
      Vb->Delta2 = 1. - (Vb->xtip2 * dx - Vb->x0 + Vb->xoffs - Vb->Lenpull) / Vb->lT;
#endif

      Vb->Omega1 = 1. / (1. - Ct->kcoeff) * (1. - Ct->kcoeff / (Ct->kcoeff + (1. - Ct->kcoeff) * Vb->Delta1));
      Vb->Omega2 = 1. / (1. - Ct->kcoeff) * (1. - Ct->kcoeff / (Ct->kcoeff + (1. - Ct->kcoeff) * Vb->Delta2));

      OutputGrainTip(Ct, Vb);
    } // End of output tip

    ///////////////////////////////
    // Output SVG and (or) Movie //
    ///////////////////////////////
#if (NOUTSVG > 0)
    if (Vb->iter % Vb->IterOutSvg == 0) {
      CopyFieldsDev2Host(1);
      WriteFields(h_Psi, h_U, h_Phase, h_Temperature, *Ct, *Vb, int(Vb->iter * Ct->dt * Ct->Tau0_sec), 1);
    }
#endif

#if (NOUTFIELDS > 0)
    if (Vb->iter % Vb->IterOutFields == 0) {
      CopyFieldsDev2Host(1);
      WriteFields(h_Psi, h_U, h_Phase, h_Temperature, *Ct, *Vb, int(Vb->iter * Ct->dt * Ct->Tau0_sec), 0);
      OutputIndexFile(Ct, Vb);
    }
#endif

    // for debug !!!
    /*
            CopyFieldsDev2Host(1);
            WriteFields(h_Psi, h_U, h_Phase, h_Temperature, *Ct, *Vb, Vb->iter, 0);
            char FileName[256];
            FILE *OutFile;
            sprintf(FileName,"m_Dphit_binary.%d.dat",Vb->iter);
            OutFile=fopen(FileName,"w");

            for(int i=0; i<Nx+2; i++) {
                    REAL d = h_Dphidt_Nx[i];
                    fwrite((char*)&d,sizeof(REAL),1,OutFile);
            }
            fclose(OutFile);
    */

    ///////////////////////////////
    // 	   Break check point     //
    ///////////////////////////////
    tNow = clock();
    CompTime = (tNow - tBegin) / CLOCKS_PER_SEC;
    if (CompTime > Vb->tlimit) {
      CopyFieldsDev2Host(1);

      WriteBinaryFiles(h_Psi, h_U, h_Phase, h_Temperature, *Ct);
      WriteParam(Vb, 0);

      // Save iteration information at check point
      FILE *CHKDiscovery = fopen("Break_info.txt", "a");
      fprintf(CHKDiscovery, "%g \t%d \t%d \t%d \n", CompTime, Vb->tlimit, Vb->iter, Vb->niter);
      fclose(CHKDiscovery);

      printf("Binary fields have been written at break point (iter=%d,niter=%d) \n", Vb->iter, Vb->niter);

      break;
    }
  }

  // Final output
  if (Vb->iter >= Vb->niter && Vb->iter < (Vb->niter + 10)) // if quit unexpectedly, then don't write the final files
  {
    CopyFieldsDev2Host(1);
    WriteFields(h_Psi, h_U, h_Phase, h_Temperature, *Ct, *Vb, IndexFINAL, 1);
    WriteFields(h_Psi, h_U, h_Phase, h_Temperature, *Ct, *Vb, IndexFINAL, 0);
    OutputIndexFile(Ct, Vb);

    // Write Vb to Final_Variables.dat
    WriteParam(Vb, 1);

    // Save iteration information at final point
    FILE *CHKDiscovery = fopen("Break_info.txt", "a");
    fprintf(CHKDiscovery, "%g \t%d \t%d \t%d \n", CompTime, Vb->tlimit, Vb->iter, Vb->niter);
    fclose(CHKDiscovery);

#if (COMPRESS) // Compress final files
    CompressFiles(*Ct);
#endif
  }

  return 0;
}