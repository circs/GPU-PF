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



#include "gpu-manager.h"

// C++ & CUDA headers
#include <fstream>
#include <iostream>
// Project headers
#include "macro.h"

// Internal function
int AutoBlockSize(int *Bloc, int DEV_Nx);

// Setup dv->DEVICES, dv->nx, dv->SizeGrid
int GpuManager::Setup(char *GPU_list) {
  // NGPU = strlen(GPU_list);  			// # of GPU requested
  int AVAIL_GPU; // # of available GPU
  cudaGetDeviceCount(&AVAIL_GPU);
  if (NGPU > AVAIL_GPU) {
    printf("Invalid number of GPUs specified: %i is greater than "
           "the total number of available GPUs (%i)\n",
           NGPU, AVAIL_GPU);
    exit(1);
  }
  printf("Run with %i DEVICES\n", NGPU);

  // dv->DEVICES
  // dv->DEVICES = (int*)malloc(NGPU*sizeof(int)) ;
  for (int i = 0; i < NGPU; i++) {
    dv->DEVICES[i] = GPU_list[i] - '0'; // get the list of device numbers
  }

  // dv->nx
  // dv->nx = (int*)malloc(NGPU*sizeof(int)) ;
  int Quotient, Remainder;
  Remainder = Nx % NGPU;
  Quotient = (Nx - Remainder) / NGPU;
  for (int i = 0; i < NGPU; i++) {
    if (i < Remainder) {
      dv->nx[i] = Quotient + 1;
    } else {
      dv->nx[i] = Quotient;
    }
  }

  // dv->SizeGrid
  // dv->SizeGrid = (int*)malloc(NGPU*sizeof(int)) ;
  for (int i = 0; i < NGPU; i++) {
    dv->SizeGrid[i] = (dv->nx[i] + 2) * (Ny + 2) * (Nz + 2);
  }

  // Mem Index
  // dv->SizeCopy 	= (int*)malloc(NGPU*sizeof(int)) ;
  // dv->AcmNx 		= (int*)malloc(NGPU*sizeof(int)) ;
  // dv->h_Host2Dev 	= (int*)malloc(NGPU*sizeof(int)) ;
  // dv->h_Dev2Host 	= (int*)malloc(NGPU*sizeof(int)) ;
  // dv->d_Dev2Host 	= (int*)malloc(NGPU*sizeof(int)) ;

  int Temp1 = 0;
  int Temp2 = 0;
  int Temp3 = 0;
  for (int i = 0; i < NGPU; i++) {
    dv->h_Host2Dev[i] = Temp1;
    Temp1 += dv->nx[i] * (Ny + 2) * (Nz + 2);
    dv->h_Dev2Host[i] = Temp2;

    if (i == 0) {
      Temp2 = (dv->nx[0] + 1) * (Ny + 2) * (Nz + 2);
      dv->d_Dev2Host[0] = 0;
    } 
    else {
      Temp2 += dv->nx[i] * (Ny + 2) * (Nz + 2);
      dv->d_Dev2Host[i] = (Ny + 2) * (Nz + 2);
    }

    if (i == 0 || i == (NGPU - 1)) {
      dv->SizeCopy[i] = (dv->nx[i] + 1) * (Ny + 2) * (Nz + 2);
    } else {
      dv->SizeCopy[i] = dv->nx[i] * (Ny + 2) * (Nz + 2);
    }

    dv->AcmNx[i] = Temp3;
    Temp3 += dv->nx[i];
  }

  return 0;
}

int GpuManager::EnableP2P() // Check the compatibility of P2P communication. If supported, enable P2P.
{
  int peer_access_available = 0;

  if (NGPU > 1) {
    for (int i = 0; i < (NGPU - 1); i++) {
      cudaSetDevice(dv->DEVICES[i]);
      peer_access_available = 0;
      cudaDeviceCanAccessPeer(&peer_access_available, dv->DEVICES[i], dv->DEVICES[i + 1]);
      printf("> Direct peer access from GPU%d to GPU%d: %s \n", dv->DEVICES[i], dv->DEVICES[i + 1], (peer_access_available ? "enabled" : "not support"));
      cudaDeviceEnablePeerAccess(dv->DEVICES[i + 1], 0);

      cudaSetDevice(dv->DEVICES[i + 1]);
      peer_access_available = 0;
      cudaDeviceCanAccessPeer(&peer_access_available, dv->DEVICES[i + 1], dv->DEVICES[i]);
      printf("> Direct peer access from GPU%d to GPU%d: %s \n", dv->DEVICES[i + 1], dv->DEVICES[i], (peer_access_available ? "enabled" : "not support"));
      cudaDeviceEnablePeerAccess(dv->DEVICES[i], 0);
    }
  }

  return 0;
}

int GpuManager::DisableP2P() // Since P2P occupies memory on device, need to disable P2P at the end.
{
  if (NGPU > 1) {
    for (int i = 0; i < (NGPU - 1); i++) {
      cudaSetDevice(dv->DEVICES[i]);
      cudaDeviceDisablePeerAccess(dv->DEVICES[i + 1]);

      cudaSetDevice(dv->DEVICES[i + 1]);
      cudaDeviceDisablePeerAccess(dv->DEVICES[i]);
    }
  }

  return 0;
}

int GpuManager::IsUnifiedAddressing() // Check the compatibility of UVA.
{
  cudaDeviceProp prop[NGPU];

  for (int i = 0; i < NGPU; i++) {
    cudaGetDeviceProperties(&prop[i], dv->DEVICES[i]);
    printf("> GPU%d: %s %s unified addressing\n", dv->DEVICES[i], prop[i].name,
           (prop[i].unifiedAddressing ? "support" : "not support"));
  }

  return 0;
}

int GpuManager::DisplayDeviceProperties() {
  int Ndev = dv->DEVICES[0]; // Display the property of dv->DEVICES[0]

  cudaDeviceProp deviceProp;
  memset(&deviceProp, 0, sizeof(deviceProp));
  if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, Ndev)) {
    printf("==============================================================");
    printf("\nDevice Name \t %s ", deviceProp.name);
    printf("\nDevice Index\t %d ", Ndev);
    printf("\n==============================================================");
    printf("\nTotal Global Memory                  \t %ld KB", (long int)(deviceProp.totalGlobalMem / 1024));
    printf("\nShared memory available per block    \t %ld KB", (long int)(deviceProp.sharedMemPerBlock / 1024));
    printf("\nNumber of registers per thread block \t %d", deviceProp.regsPerBlock);
    printf("\nWarp size in threads             \t %d", deviceProp.warpSize);
    printf("\nMemory Pitch                     \t %ld bytes", (long int)(deviceProp.memPitch));
    printf("\nMaximum threads per block        \t %d", deviceProp.maxThreadsPerBlock);
    printf("\nMaximum Thread Dimension (block) \t %d * %d * %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("\nMaximum Thread Dimension (grid)  \t %d * %d * %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("\nTotal constant memory            \t %ld bytes", (long int)(deviceProp.totalConstMem));
    printf("\nCompute capability		    	\t %d.%d", deviceProp.major, deviceProp.minor);
    printf("\nClock rate                       \t %d KHz", deviceProp.clockRate);
    printf("\nTexture Alignment                \t %ld bytes", (long int)(deviceProp.textureAlignment));
    printf("\nDevice Overlap                   \t %s", deviceProp.deviceOverlap ? "Allowed" : "Not Allowed");
    printf("\nNumber of Multi processors       \t %d", deviceProp.multiProcessorCount);
    printf("\n==============================================================\n");

    dv->maxThreadsPerBlock = deviceProp.maxThreadsPerBlock; // record the Maximum threads per block
  } else {
    printf("\nCould not get properties for device %d.....\n", Ndev);
  }

  return 0;
}

int GpuManager::GpuBlock() {
  // dv->BLOCK_X = (int*)malloc(NGPU*sizeof(int)) ;
  // dv->BLOCK_Y = (int*)malloc(NGPU*sizeof(int)) ;
  // dv->BLOCK_Z = (int*)malloc(NGPU*sizeof(int)) ;

  // dv->BLOCK_XYZ[0]=dv->BLOCK_X;
  // dv->BLOCK_XYZ[1]=dv->BLOCK_Y;
  // dv->BLOCK_XYZ[2]=dv->BLOCK_Z;

  int B[3] = {1, 1, 1};
  for (int i = 0; i < NGPU; i++) {
    B[0] = 1;
    B[1] = 1;
    B[2] = 1;
    AutoBlockSize(B, dv->nx[i]);
    dv->BLOCK_X[i] = B[0];
    dv->BLOCK_Y[i] = B[1];
    dv->BLOCK_Z[i] = B[2];
  }

  // for the large domain
  B[0] = 1;
  B[1] = 1;
  B[2] = 1;
  AutoBlockSize(B, Nx);
  dv->BLOCK_Nx = B[0];
  dv->BLOCK_Ny = B[1];
  dv->BLOCK_Nz = B[2];

  return 0;
}

int AutoBlockSize(int *Bloc, int DEV_Nx) // ??? need to revise to improve the performance
{
  for (int bx = 1; bx <= BSIZEMAX; bx++) {
    if (((DEV_Nx + 2.) / bx) == ((DEV_Nx + 2) / bx)) {
      for (int by = 1; by <= BSIZEMAX && bx * by <= BLOCKMAX; by++) {
        if (((Ny + 2.) / by) == ((Ny + 2) / by)) {
          for (int bz = 1; bz <= BSIZEMAX && bx * by * bz <= BLOCKMAX; bz++) {
            if (((Nz + 2.) / bz) == ((Nz + 2) / bz)) {
              if (bx * by * bz == Bloc[0] * Bloc[1] * Bloc[2]) {
                int VAR = (Bloc[0] - Bloc[1]) * (Bloc[0] - Bloc[1]) + (Bloc[0] - Bloc[2]) * (Bloc[0] - Bloc[2]) + (Bloc[2] - Bloc[1]) * (Bloc[2] - Bloc[1]);
                int var = (bx - by) * (bx - by) + (bx - bz) * (bx - bz) + (bz - by) * (bz - by);
                if (var < VAR) {
                  Bloc[0] = bx;
                  Bloc[1] = by;
                  Bloc[2] = bz;
                }
              } else if (bx * by * bz > Bloc[0] * Bloc[1] * Bloc[2]) {
                Bloc[0] = bx;
                Bloc[1] = by;
                Bloc[2] = bz;
              }
            }
          }
        }
      }
    }
  }

  return 0;
}
