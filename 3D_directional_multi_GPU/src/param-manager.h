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



#ifndef PARAM_MANAGER_H_
#define PARAM_MANAGER_H_

// Project headers
#include "gpu-manager.h"
#include "macro.h"

struct Constants // 38 in total
{

  int IterPull;
  char OutputPrefix[LENMAX];
  char TipFileName[LENMAX];
  char IndexFileName[LENMAX]; // for IndexTIme.dat file

  REAL sqrt2;
  REAL PI;
  REAL dt;
  REAL kcoeff;
  REAL omk;
  REAL opk;
  REAL Eps4;
  REAL D;
  REAL Lambda;

  REAL r11[2];
  REAL r12[2];
  REAL r13[2];

  REAL r21[2];
  REAL r22[2];
  REAL r23[2];

  REAL r31[2];
  REAL r32[2];
  REAL r33[2];

  REAL Vp;
  // REAL lT;
  REAL lT0;
  REAL lT1;

  REAL W_microns;
  REAL dx_microns;
  REAL Tau0_sec;

  // local heat
  REAL Hamp;
  REAL slht;
  REAL flht;

  // for thermal drift
  REAL Tdtau;
  REAL Tddzt;

  // for 2D record
#if (CutZplane != 0)
  int X2DMAX;
  char D2Time[128];
#endif


  // For oscillatory pulling velocity
#if (OSC_Velocity)
  REAL OSCVamp0; // The amplitude of Vp oscillation [W]
#endif
};

struct Variables {

  int tlimit; // for break

  int iter;
  int niter;

  REAL lT;

  REAL xint;
  double xoffs;
  REAL x0;
  REAL xprev;

  REAL xtip;
  REAL ytip;
  REAL ztip;

  REAL xtip1;
  REAL ytip1;
  REAL ztip1;

  REAL xtip2;
  REAL ytip2;
  REAL ztip2;

  REAL RadY;
  REAL RadZ;

  REAL Vel;
  REAL Delta;
  REAL Omega;

  // for grain boundaries
  int jGBy;

  // for outputs
  int IterOutFields;
  int IterOutSvg;
  int IterOutTip;

  // for Grain tip
  char FileNameGBTips[LENMAX];
  char FileNameGBL[LENMAX];

  REAL Delta1;
  REAL Delta2;
  REAL Omega1;
  REAL Omega2;

  // for output fields
  int numoutput_f; // for NOUTFIELDS < 0
  int numoutput_z; // for CutZplane < 0


  // For oscillatory pulling velocity
  double Lenpull; // pulled length [W]
#if (OSC_Velocity)
  REAL OSCVamp;
  int OSCNstep; // for step-like forcing
#endif
};

class ParamManager {
public:
  ParamManager() {
    Ct = new Constants;
    Vb = new Variables;
  }
  ~ParamManager() {
    delete Ct;
    delete Vb;
  }

  Constants *Ct;
  int InitConstants();
  Constants *CopyConstants() { return Ct; }

  Variables *Vb;
  int InitVariables(int tlimit_src,
                    int iter_src, int niter_src);
  Variables *CopyVariables() { return Vb; }
};

#endif
