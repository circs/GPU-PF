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



// C++ headers
#include "time.h"
#include <fstream> // for printf("%d\n", );
#include <iostream>

// Project headers
#include "fields-evolution.h"
#include "macro.h"

// Normal initialization
int BeginfromInit(char *GPU_list, unsigned long long seed, int tlimit = 9999999) {
  clock_t StartTime = clock();

  FieldsEvolution *fe;
  fe = new FieldsEvolution();

  fe->SetupGpu(GPU_list);
  fe->SetupParam(false, tlimit); // if starts from break: false
  fe->AllocateGpuMem();
  fe->InitInterParam();


#if (INITfromFILE)
  fe->InitfromFile(seed);
#else
  fe->InitfromFunc(seed);
#endif

  fe->theFirstOutput();
  fe->MainTimeLoop(StartTime);

  fe->FreeGpuMem();

  delete fe;
  return 0;
}

// Continue the program from break point
int BeginfromBreak(char *GPU_list, int tlimit,
                   int iter, int niter, unsigned long long seed) {
  clock_t StartTime = clock();

  FieldsEvolution *fe;
  fe = new FieldsEvolution();

  fe->SetupGpu(GPU_list);
  fe->SetupParam(true, tlimit, iter, niter); // if starts from break: true
  fe->AllocateGpuMem();
  fe->InitInterParam();


  fe->InitfromBreak(seed);
  fe->MainTimeLoop(StartTime);

  fe->FreeGpuMem();

  delete fe;
  return 0;
}

int TestBreak(char *GPU_list, unsigned long long seed, int tlimit = 9999999) {
  clock_t StartTime = clock();

  FieldsEvolution *fe;
  fe = new FieldsEvolution();

  fe->SetupGpu(GPU_list);
  fe->SetupParam(tlimit);
  fe->AllocateGpuMem();
  fe->InitInterParam();

  fe->InitfromBreak(seed);
  // fe->InitfromFile();
  fe->theFirstOutput();
  fe->MainTimeLoop(StartTime);

  fe->FreeGpuMem();
  delete fe;
  return 0;
}

int main(int argc, char **argv) {
  // random number seed
#if (RSEED == 0)
  unsigned long long seed = time(NULL);
#else
  unsigned long long seed = RSEED;
#endif
  printf("The random seed is %llu\n", seed);

  // parse command line
  char *GPU_list = argv[1];

  if (argc > 2) {
    int tlimit = atoi(argv[2]);
    int iter = atoi(argv[3]);
    int niter = atoi(argv[4]);

    if (iter == 0) {
      BeginfromInit(GPU_list, seed, tlimit);
    } else {
      BeginfromBreak(GPU_list, tlimit, iter, niter, seed);
    }
  } else // without parse command line control
  {
    BeginfromInit(GPU_list, seed);
    // TestBreak(GPU_list,seed);
  }

  return 0;
}
