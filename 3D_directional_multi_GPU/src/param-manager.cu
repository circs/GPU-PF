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



#include "param-manager.h"

// C++ & CUDA headers
#include <fstream>
#include <iostream>
#include <sstream>
// Project headers
#include "gpu-manager.h"
#include "macro.h"

int ParamManager::InitConstants() {
  // Constants
  REAL PI = 4. * atan(1.);
  REAL a1 = 5. * sqrt(2.) / 8.;
  REAL a2 = 47. / 75.;
  // ------------------------------------------------------
  // ---------------- SIMULATION PARAMETERS ---------------
  // ------------------------------------------------------
  // ------------------------ Alloy -----------------------
  REAL c0 = COMPOSITION;     // Nominal composition, unitC
  REAL m = LIQSLOPE;         // |Liquidus slope|, K/unitC
  REAL Diff = DIFFUSION;     // Diffusion, micron^2/s
  REAL Gamma = GIBBSTHOMSON; // Gibbs thomson coefficient, K.m
  REAL kcoeff = PARTITION;   // Partition coefficient
  REAL Eps4 = ANISOTROPY;
  // ----------------------- Process ----------------------
  REAL Vpull = VELOCITY;                                       // Pulling speed, microns/s
                                                               // rotation angles
  REAL Aalpha[2] = {AngleA1 * PI / 180., AngleA2 * PI / 180.}; // angle 1
  REAL Abeta[2] = {AngleB1 * PI / 180., AngleB2 * PI / 180.};  // angle 2
  REAL Agamma[2] = {AngleC1 * PI / 180., AngleC2 * PI / 180.}; // angle 3
  // ------------------------------------------------------
  REAL mc0 = m * c0;             // |Liquidus slope|*Nominal composition, K
  REAL DT0 = mc0 / kcoeff - mc0; // Solidification range, K
  REAL d0 = Gamma / DT0 * 1.e6;  // Capillarity length @ T0, microns
  // -------------- Non-dimensional parameters ------------
  REAL D = a1 * a2 * E;
  REAL Lambda = a1 * E;
  REAL Vp = Vpull * d0 / Diff * a1 * a2 * E * E; //*** why?
  REAL W_microns = E * d0;                       // [microns]
  REAL dx_microns = W_microns * dx;              // [microns]
  REAL Tau0_sec = Vp / Vpull * W_microns;        // [seconds]
  REAL lT0 = DT0 / GRAD0 * 1.e6 / (E * d0);
  REAL lT1 = DT0 / GRAD1 * 1.e6 / (E * d0);
  // REAL lT = lT0;
  // ------------------- Output ---------------------------
#if (TIME0 > 0)
  sprintf(Ct->OutputPrefix, "%s_D%d_G%dto%d_k0%d_V%d_dx%d_W%d", PREFIX, (int)(DIFFUSION), (int)(GRAD0 / 100), (int)(GRAD1 / 100), (int)(PARTITION * 100), (int)(VELOCITY * 10), (int)(dx * 10), (int)(E));
#else
  sprintf(Ct->OutputPrefix, "%s_D%d_G%d_k0%d_V%d_dx%d_W%d", PREFIX, (int)(DIFFUSION), (int)(GRAD0 / 100), (int)(PARTITION * 100), (int)(VELOCITY * 10), (int)(dx * 10), (int)(E));
#endif

  sprintf(Ct->TipFileName, "%s.tip.dat", Ct->OutputPrefix);
  sprintf(Ct->IndexFileName, "%s_IndexTime.dat", Ct->OutputPrefix);

  // --------------------------------------------------------------
  // Making output iterations multiples of IterPull=dx/vp/dt,
  // dt adjuted for pulling back every round number of iterations,
  // in order to avoid spurious results oscillations due to
  // non-synchronization between pull-back and output frequencies
  // --------------------------------------------------------------
  REAL dt = dt0;
  if (dt > Kdt * dx * dx / 6. / D) {
    dt = Kdt * dx * dx / 6. / D;
  }
  int IterPull = int(dx / Vp / dt + 1);
  dt = dx / Vp / IterPull;
  IterPull = int(dx / Vp / dt);

  // 38 constants
  Ct->IterPull = IterPull;

  Ct->sqrt2 = sqrt(2.);
  Ct->PI = 4. * atan(1.);
  Ct->dt = dt;
  Ct->kcoeff = kcoeff;
  Ct->omk = 1. - kcoeff;
  Ct->opk = 1. + kcoeff;
  Ct->Eps4 = Eps4;
  Ct->D = D;
  Ct->Lambda = Lambda;

  // rotation angles - grain 1

  // grain 1
  Ct->r11[0] = cos(Aalpha[0]) * cos(Agamma[0]) + sin(Aalpha[0]) * sin(Abeta[0]) * sin(Agamma[0]);
  Ct->r12[0] = sin(Aalpha[0]) * cos(Agamma[0]) - cos(Aalpha[0]) * sin(Abeta[0]) * sin(Agamma[0]);
  Ct->r13[0] = cos(Abeta[0]) * sin(Agamma[0]);

  Ct->r21[0] = -1. * sin(Aalpha[0]) * cos(Abeta[0]);
  Ct->r22[0] = cos(Aalpha[0]) * cos(Abeta[0]);
  Ct->r23[0] = sin(Abeta[0]);

  Ct->r31[0] = -1. * cos(Aalpha[0]) * sin(Agamma[0]) + sin(Aalpha[0]) * sin(Abeta[0]) * cos(Agamma[0]);
  Ct->r32[0] = -1. * sin(Aalpha[0]) * sin(Agamma[0]) - cos(Aalpha[0]) * sin(Abeta[0]) * cos(Agamma[0]);
  Ct->r33[0] = cos(Abeta[0]) * cos(Agamma[0]);

  // grain 2
  Ct->r11[1] = cos(Aalpha[1]) * cos(Agamma[1]) + sin(Aalpha[1]) * sin(Abeta[1]) * sin(Agamma[1]);
  Ct->r12[1] = sin(Aalpha[1]) * cos(Agamma[1]) - cos(Aalpha[1]) * sin(Abeta[1]) * sin(Agamma[1]);
  Ct->r13[1] = cos(Abeta[1]) * sin(Agamma[1]);

  Ct->r21[1] = -1. * sin(Aalpha[1]) * cos(Abeta[1]);
  Ct->r22[1] = cos(Aalpha[1]) * cos(Abeta[1]);
  Ct->r23[1] = sin(Abeta[1]);

  Ct->r31[1] = -1. * cos(Aalpha[1]) * sin(Agamma[1]) + sin(Aalpha[1]) * sin(Abeta[1]) * cos(Agamma[1]);
  Ct->r32[1] = -1. * sin(Aalpha[1]) * sin(Agamma[1]) - cos(Aalpha[1]) * sin(Abeta[1]) * cos(Agamma[1]);
  Ct->r33[1] = cos(Abeta[1]) * cos(Agamma[1]);

  Ct->Vp = Vp;
  // Ct->lT=lT;
  Ct->lT0 = lT0;
  Ct->lT1 = lT1;

  Ct->W_microns = W_microns;
  Ct->dx_microns = dx_microns;
  Ct->Tau0_sec = Tau0_sec;

  Ct->slht = TIMELH;
  Ct->flht = 0.;
  Ct->Hamp = ampLH;

  // for thermal drift
  Ct->Tdtau = Thermaltau / Tau0_sec;  // [tau_0]
  Ct->Tddzt = Thermaldzt / W_microns; // [W]

  // for 2D record
#if (CutZplane != 0)
  Ct->X2DMAX = (Xout2DMAX == 0) ? Nx + 2 : (int)(POSITION_0 / Ct->W_microns / dx + XoutMAX); // output 2D field along the x axis.
  sprintf(Ct->D2Time, "%s_IndexTimeFor2D.dat", Ct->OutputPrefix);
#endif


#if (OSC_Velocity)
  Ct->OSCVamp0 = OSC_Vamp * d0 / Diff * a1 * a2 * E * E; // The amplitude of Vp oscillation [W]
#endif

  return 0;
}

int ParamManager::InitVariables(int tlimit_src,
                                int iter_src, int niter_src) {
  REAL TotalTime = TOTALTIME / Ct->Tau0_sec; // [/Tau0]
  REAL Delta0 = UNDERCOOL_0;                 // Initial supercooling

  Vb->tlimit = tlimit_src; // Default tlimit = 9999999

  Vb->iter = iter_src; // Default iter = 0

  Vb->lT = Ct->lT0; // new changed

  // niter multiple of IterPull
  Vb->niter = int(TotalTime / Ct->dt);
  Vb->niter = Vb->niter / Ct->IterPull;
  Vb->niter = (Vb->niter + 1) * Ct->IterPull;

  Vb->xoffs = 0.;
  Vb->xprev = 0.;

  Vb->xint = POSITION_0 / Ct->W_microns;
#if (LateralG > 0)
  Vb->x0 = Vb->xint - 0.25 * dx * LateralG - (1. - Delta0) * Vb->lT; // Initial interface position [/W]
#else
  Vb->x0 = Vb->xint - (1. - Delta0) * Vb->lT; // Initial interface position [/W]
#endif
  TotalTime = Vb->niter * Ct->dt;

  Vb->xtip = Vb->xint / dx;
  Vb->ytip = 0.;
  Vb->ztip = 0.;
  Vb->xtip1 = Vb->xint / dx;
  Vb->ytip1 = 0.;
  Vb->ztip1 = 0.;
  Vb->xtip2 = Vb->xint / dx;
  Vb->ytip2 = 0.;
  Vb->ztip2 = 0.;

  Vb->RadY = 0.;
  Vb->RadZ = 0.;

  Vb->Vel = 0.;
  Vb->Delta = 1. - (Vb->xtip * dx - Vb->x0 + Vb->xoffs - Ct->Vp * Ct->dt * Vb->iter) / Vb->lT;
  Vb->Omega = 1. / (1. - Ct->kcoeff) * (1. - Ct->kcoeff / (Ct->kcoeff + (1. - Ct->kcoeff) * Vb->Delta));

  Vb->jGBy = (int)(GByloc);

  //  IterOutFields multiple of IterPull
  if (NOUTFIELDS > 0) {
    Vb->IterOutFields = Vb->niter / NOUTFIELDS;
    Vb->IterOutFields = Vb->IterOutFields / Ct->IterPull;
    Vb->IterOutFields = Vb->IterOutFields * Ct->IterPull;
    if (Vb->IterOutFields == 0)
      Vb->IterOutFields = Ct->IterPull;
  }

  //  IterOutSvg multiple of IterPull
  if (NOUTSVG > 0) {
    Vb->IterOutSvg = Vb->niter / NOUTSVG;
    Vb->IterOutSvg = Vb->IterOutSvg / Ct->IterPull;
    Vb->IterOutSvg = Vb->IterOutSvg * Ct->IterPull;
    if (Vb->IterOutSvg == 0)
      Vb->IterOutSvg = Ct->IterPull;
  }

  if (NOUTTIP > 0) {
    Vb->IterOutTip = Vb->niter / NOUTTIP;
    Vb->IterOutTip = Vb->IterOutTip / Ct->IterPull;
    Vb->IterOutTip = Vb->IterOutTip * Ct->IterPull;
    if (Vb->IterOutTip == 0)
      Vb->IterOutTip = Ct->IterPull;
  }

  // for grain tip
  sprintf(Vb->FileNameGBTips, "%s.GBtips.dat", Ct->OutputPrefix);
  sprintf(Vb->FileNameGBL, "%s.GBL.dat", Ct->OutputPrefix);

  Vb->Delta1 = Vb->Delta2 = Vb->Delta;
  Vb->Omega1 = Vb->Omega2 = Vb->Omega;

  Vb->numoutput_f = 1; // for NOUTFIELDS < 0
  Vb->numoutput_z = 1; // for CutZplane < 0


  // For oscillatory pulling velocity
  Vb->Lenpull = 0.; // pulled length [W]
#if (OSC_Velocity)
  Vb->OSCVamp = ((int)(OSC_t0) == 0) ? Ct->OSCVamp0 : 0.;
  Vb->OSCNstep = 1; // for step-like forcing
#endif

  return 0;
}
