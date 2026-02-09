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



#ifndef IO_FUNCTIONS_H_
#define IO_FUNCTIONS_H_

// C++ headers
#include <iostream>
// Project headers
#include "gpu-manager.h"
#include "macro.h"
#include "param-manager.h"

int IO_InitfromFile_DAT(REAL *P, REAL *U, signed char *Phase, REAL *Temperature, Constants Ct);

int IO_InitFromFile_VTK(REAL *P, REAL *U, signed char *Phase, REAL *Temperature, Constants Ct, Variables *Vb);

int IO_InitfromBreak(REAL *P, REAL *U, signed char *Phase, REAL *Temperature, Constants Ct);

// When (NOUTFIELDS<0), output IndexTime.dat file
int InitIndexFile(Constants *Ct, Variables *Vb);
int OutputIndexFile(Constants *Ct, Variables *Vb);

int InitTipFile(char *TipFileName);

int OutputTipFile(Constants *Ct, Variables *Vb);

int OutputGrainTip(Constants *Ct, Variables *Vb);

void OutputParameters(GpuManager_Param dv, Constants Ct, Variables Vb);

void WriteParam(Variables *Vb, int ifFinal);

void ReadParam(Variables *Vb, int ifFinal);

void WriteFields(REAL *P, REAL *U, signed char *Phase, REAL *Temperature,
                 Constants Ct, Variables Vb,
                 int index, int SVG);

void WriteBinaryFiles(REAL *P, REAL *U, signed char *Phase, REAL *Temperature, Constants Ct);

#if (CutZplane != 0)
void WriteZ2Dfield(REAL *P, REAL *U,
                   signed char *Phase,
                   Constants Ct,
                   Variables Vb,
                   int zplane, int d2num, int Xmax2Df);
#endif

#if (COMPRESS)
int CompressFiles(Constants Ct);
#endif

#endif