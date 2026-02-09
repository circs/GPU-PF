/*---------------------------------------------------------------
 *
 * This code is developed by CIRCS group of Northeastern University.
 *
 * Contact Information:
 * 
 * Center for interdisciplinary research on complex systems
 * Departments of Physics, Northeastern University
 * 
 * Alain Karma    a.karma (at) northeastern.edu
 *
 *
 * we cordially ask that any published work derived 
 * from this code, or utilizing it references the following published works: 
 * 
 * 1)Clarke, A. J. et al. Microstructure selection in thin-sample directional solidification of an Al-Cu alloy: In situ X-ray imaging and phase-field simulations. \
 * Acta Materialia 129, 203-216 (2017) 
 *--------------------------------------------------------------- */

// With oscillatory pulling velocity

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// -----------------------
//		SIMULATION ID
// -----------------------
#define PREFIX    "Ny48" // Prefix of Output File Name
#define Ny        256     // Dimension Y
#define Nz        256     // Dimension Z
#define INIT_FILE ""

#define E 154.1666667 // E=W/d0

// -----------------------
//		EXPERIMENT
// -----------------------
// Process
#define VELOCITY (6.)         // micron/s
#define GRAD0    (12.5 * 100) // K/m
#define GRAD1    (12.5 * 100) // K/m
#define TIME0    0            // h
#define TIME1    0            // h
// changed
#define TIMELH 0  // s
#define ampLH  0. // Amplitude by local heat
// a starting time for the local heat
// from this time, the local temperature on a tip is chaging during t~ rho/Vp
// changed

// Alloy
#define PARTITION    (0.1)
#define DIFFUSION    (270.)     // micron^2/s
#define COMPOSITION  (0.46)     // UnitCompo
#define LIQSLOPE     (1.365)    // K/UnitCompo
#define GIBBSTHOMSON (6.478e-8) // K.m
#define ANISOTROPY   (0.011)

// -----------------------
//		SIMULATION
// -----------------------
// Dimensions
#define TOTALTIME (2000.) // [seconds]
#define Nx        718     // Dimension X

// Boundary conditions
#define BOUND_COND_Y NOFLUX
#define BOUND_COND_Z NOFLUX

// effect of wall
#define WALLEFFECT WITHOUT
#define WALLSLOPE  1.
// wall slope

// effect of wall
#define AngleA 0 // angle 1, polar angle [degree]
#define AngleB 0 // angel 2, azimuthal angle [degree]
// wall slope

// Discretization

#define dx  1.2 // Grid size [W]
#define dt0 1.  // Time step (input, may be reajusted)
#define Kdt .9  // Time step adjustment criterion (dt<=Kdt*dx*dx*dx/6/D)

// Noise - if conserved noise, the Fnoise value is the unit of microns^3.
#define NOISE      FLAT
#define Fnoise     (0.01)
#define tmax_NOISE 0 // (int) in minutes

// New impose: Thermal drift
#define Theramltau 0    // [s], + switch: if >0, switch on, if =0 switch off
#define Thermaldzt (0.) // [microns]

// Computational
#define REAL         float
#define PBFRAME      TIP
#define ANTITRAPPING 1

// Oscillatory pulling velocity
// #define OSC_Velocity		  CONST_V		// It can be WITHOUT, SIN, CONST_V or LINEAR
// #define OSC_Onset             (200.)		// The onset of imposing oscillatory Vp [s]
// #define OSC_Amp               (0.1)			// The amplitude of oscillatory pulling velocity [micron]
// #define OSC_Period            (200.)			// The period of oscillatory pulling velocity [s]

#define OSC_Velocity WITHOUT // It can be WITHOUT, SINOSC, CONST_V, LINEAR or STEPLIKE
#define OSC_Vamp     (2.)    // Same as OSC_Amp [micron/s]
#define if_Vamp_Up   0       // set to 1 if the upper osc amplitude is different from the lower (only for steplike)
#define OSC_Vamp_Up  (0.6)   // the upper osc amplitude [OSC_Vamp]
#define OSC_t0       (200.)  // Same as OSC_Onset [s]
#define OSC_tk       (800)   // Time to kill OSCVamp [s]; Set 0 if no OSC_tk
#define OSC_Period   (24.)   // The Vp oscillation period [s]

// -----------------------
//	 INITIAL CONDITIONS
// -----------------------
#define UNDERCOOL_0 (0.)
#define POSITION_0  (900. / W_microns) // [W]
#define IfTemporal  0                  // set to impose an artificial lateral temperature difference, for initialzing a quarter cell
#define MIXd        0                  // [microns]

#define IQY 1  // Wavenumber of the initial perturbation /Y
#define IQZ 1  // Wavenumber of the initial perturbation /Z
#define AMP 2. // [dx]

// Input File
#define INITfromFILE  0
#define INITfromFILE2 0 // Init files from a second src. Works w/o mirror and multiply. The initial conditions are read based on the 1st src
#define INIT_FILE2    ""

#define FROM_X       0
#define MIRROR_Y     0
#define MIRROR_Z     0
#define MULTIPLY_Y   1
#define MULTIPLY_Z   1
#define CUTONEFOURTH 0
#define PERIODIC_Y   0 // When MULTIPLY_Y>1, multiply the domain periodically

// -----------------------
//		  OUTPUT
// -----------------------
#define NOUTFIELDS  50
#define NOUTTIP     2000
#define NOUTSVG     0
#define COMPRESS    1
#define OUTINDEXSEC 1
#define SUMMARIZE   1

#define AMPSEARCH  0 // how many positions to search (if AMPSEARCH>0, recording)
#define MAXINTFNUM 0 // max cross-section?

///////// GPU //////////////////
// Blocks
#define BSIZEMAX 64
#define BLOCKMAX 512
// Mapping function
#define STRIDE       ((Ny + 2) * (Nz + 2))
#define WIDTH        (Nz + 2)
#define pos(x, y, z) (STRIDE * (x) + WIDTH * (y) + (z))
///////// FLAGS ////////////////
#define WITHOUT  0
#define NOFLUX   1
#define PERIODIC 2
#define ANTISYM  3 // Applies only on Z+
#define HELICAL  4 // Applies only on Y
// for noise
#define FLAT     1
#define GAUSSIAN 2
#define CONSERVE 3
// for frame
#define LAB 0
#define TIP 1
// for wall slope
#define WSLOPE  1 // for thin sample at the interface
#define NzSLOPE 2
#define N0SLOPE 3
// for Vp oscillations
#define SINOSC   1
#define CONST_V  2
#define LINEAR   3
#define STEPLIKE 4
///////// CONSTANTS /////////////
#define LENMAX     256
#define IndexFINAL 999999
////////////////////////////////
#define Npol         10
#define DECIMALS_P   "%.2f "
#define DECIMALS_C   "%.3f "
#define DECIMALS_SVG "%.6f "
#define XoutMIN      0
#define XoutMAX      Nx
////////////////////////////////

// ----------------------------
// Parameters copied on the GPU
struct Constants {
  REAL sqrt2;
  REAL PI;
  REAL dt;
  REAL kcoeff;
  REAL omk;
  REAL opk;
  REAL Eps4;
  REAL D;
  REAL Lambda;

  REAL Alpha;
  REAL sAlpha;
  REAL cAlpha;
  REAL sAlpha2;
  REAL cAlpha2;
  REAL s2Alpha;
  REAL c2Alpha;
  REAL cAlphasAlpha;

  REAL Beta;
  REAL sBeta;
  REAL cBeta;
  REAL sBeta2;
  REAL cBeta2;
  REAL s2Beta;
  REAL c2Beta;
  REAL cBetasBeta;

  REAL Vp;
  REAL lT;
  REAL lT0;
  REAL lT1;

  REAL W_microns;
  REAL dx_microns;
  REAL Tau0_sec;
  REAL xint;
  REAL x0;

  REAL Xtip;
  REAL Ytip;
  REAL Ztip;
  REAL RadY;
  REAL RadZ;

  double xoffs;
  int iter;

  // changed
  // local heat
  REAL Hamp;
  REAL slht;
  REAL flht;
  // changed

  // for thermal drift
  REAL Tdtau;
  REAL Tddzt;
  //

  // For oscillatory pulling velocity
#if (OSC_Velocity)
  REAL OSCVamp0;
  REAL OSCVamp;
  int OSCNstep; // for step-like forcing
#endif

  double Lenpull; // pulled length, use for Δ calculation
};

////////////////////////////
// GPU computing kernels
////////////////////////////
__global__ void Init(REAL *P1, REAL *P2, REAL *U1, REAL *U2, Constants *Param, curandState *state);
//__global__ void Compute_P(REAL *Pcurr,REAL *Pnext,REAL *F,REAL *Cucurr,REAL *Jn,Constants *Param,curandState *state, REAL OSC_Vp);
__global__ void Compute_P(REAL *Pcurr, REAL *Pnext, REAL *F, REAL *Cucurr, REAL *Jn, Constants *Param, curandState *state);
__global__ void Compute_U(REAL *Pcurr, REAL *Pnext, REAL *F, REAL *Cucurr, REAL *Cunext, REAL *Jn, Constants *Param);
__global__ void PullBack(REAL *P, REAL *U, REAL *Pnext, REAL *Cunext, Constants *Param);
__global__ void BC(REAL *Pcurr, REAL *Pnext, REAL *Cucurr, REAL *Cunext);
__global__ void setup_kernel(unsigned long long seed, curandState *state);
__global__ void GetXsl_YZ(REAL *P, REAL *Xyz);
__global__ void GetXtip(REAL *Xyz, Constants *Param);
__global__ void GetRtip(REAL *Xyz, REAL *P, Constants *Param);
__device__ REAL PolInt(REAL *XA, REAL *YA, REAL X);
__device__ REAL RootBis(REAL *xp, REAL *yp);

////////////////////////////
// CPU I/O function
////////////////////////////
void OutputParameters(char Prefix[LENMAX], REAL dt, int niter, int IterOutFields, int *Bloc);
void OutputCompTime(char Prefix[LENMAX], REAL time);
void InitFromFile(REAL *P, REAL *U, Constants *Param);
void WriteFields(char Prefix[LENMAX], int index, REAL *P, REAL *U, Constants *Param, int SVG);
#if (AMPSEARCH > 0)
void WriteAmplitude(char Prefix[LENMAX], int searchloc, REAL timing, REAL *P, int outnum);
#endif

////////////////////////////
// Cuda Device Management
////////////////////////////
void DisplayDeviceProperties(int Ndev);
void GetMemUsage(int *Array, int Num);
int GetFreeDevice(int Num);
void AutoBlockSize(int *Bloc);

////////////////////////////////////////////////
//              Main CPU program              //
////////////////////////////////////////////////
int main(int argc, char **argv) {
  clock_t begin = clock();

  // Attributing a GPU device
  int CudaDevice = 0;
  if (argc > 1) {
    CudaDevice = atoi(argv[1]);
  } else {
    // Looking for a free device
    int Ndevices = -1;
    cudaGetDeviceCount(&Ndevices);
    if (Ndevices > 1) {
      CudaDevice = GetFreeDevice(Ndevices);
      if (CudaDevice < 0) {
        return 1;
      }
    }
  }
  cudaSetDevice(CudaDevice);
  DisplayDeviceProperties(CudaDevice);

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
  REAL Vpull = VELOCITY;           // Pulling speed, microns/s
  REAL Alpha = AngleA * PI / 180.; // polar angle
  REAL Beta = AngleB * PI / 180.;  // azimuthal angle
  // ------------------------------------------------------
  REAL mc0 = m * c0;             // |Liquidus slope|*Nominal composition, K
  REAL DT0 = mc0 / kcoeff - mc0; // Solidification range, K
  REAL d0 = Gamma / DT0 * 1.e6;  // Capillarity length @ T0, microns
  // -------------- Non-dimensional parameters ------------
  REAL D = a1 * a2 * E;
  REAL Lambda = a1 * E;
  REAL Vp = Vpull * d0 / Diff * a1 * a2 * E * E;

#if (OSC_Velocity)
  //	REAL OSC_Vp=Vp;
  REAL OSCVamp0 = OSC_Vamp * d0 / Diff * a1 * a2 * E * E;
#endif

  REAL W_microns = E * d0;                // [microns]
  REAL dx_microns = W_microns * dx;       // [microns]
  REAL Tau0_sec = Vp / Vpull * W_microns; // [seconds]
  REAL lT0 = DT0 / GRAD0 * 1.e6 / (E * d0);
  REAL lT1 = DT0 / GRAD1 * 1.e6 / (E * d0);
  REAL lT = lT0;
  // REAL lD=D/Vp;
  // --------------- Initial Conditions -------------------
  REAL Delta0 = UNDERCOOL_0; // Initial supercooling
  REAL xint = POSITION_0;    // Initial interface position [/W]
  REAL x0 = xint - (1. - Delta0) * lT;
  // ----------------- Computational ----------------------
  REAL TotalTime = TOTALTIME / Tau0_sec; // [/Tau0]
  REAL dt = dt0;
  if (dt > Kdt * dx * dx / 6. / D) {
    dt = Kdt * dx * dx / 6. / D;
  }
  // ------------------- Output ---------------------------
  char OutputPrefix[LENMAX];

#if (TIME0 > 0)
  sprintf(OutputPrefix, "%s_D%d_G%dto%d_k0%d_V%d_dx%d_W%d", PREFIX, (int)(DIFFUSION), (int)(GRAD0 / 100), (int)(GRAD1 / 100), (int)(PARTITION * 100), (int)(VELOCITY * 10), (int)(dx * 10), (int)(E));
#else
  sprintf(OutputPrefix, "%s_D%d_G%d_k0%d_V%d_dx%d_W%d", PREFIX, (int)(DIFFUSION), (int)(GRAD0 / 100), (int)(PARTITION * 100), (int)(VELOCITY * 10), (int)(dx * 10), (int)(E));
#endif

  // --------------------------------------------------------------
  // Making output iterations multiples of IterPull=dx/vp/dt,
  // dt adjuted for pulling back every round number of iterations,
  // in order to avoid spurious results oscillations due to
  // non-synchronization between pull-back and output frequencies
  // --------------------------------------------------------------
  int IterPull = int(dx / Vp / dt + 1);
  dt = dx / Vp / IterPull;
  IterPull = int(dx / Vp / dt);

  // niter multiple of IterPull
  int niter = int(TotalTime / dt);
  niter = niter / IterPull;
  niter = (niter + 1) * IterPull;
  TotalTime = niter * dt;
  // ------------------------------------------------------
  int IterOutFields = niter;
#if (NOUTFIELDS > 0)
  IterOutFields = niter / NOUTFIELDS;
  IterOutFields = IterOutFields / IterPull;
  IterOutFields = IterOutFields * IterPull;
  if (IterOutFields == 0)
    IterOutFields = IterPull;
  int iPullFields = IterOutFields / IterPull;
  if (iPullFields == 0)
    iPullFields = 1;
#endif
    // ------------------------------------------------------
#if (NOUTSVG > 0)
  int IterOutSvg = niter / NOUTSVG;
#endif
  // ------------------------------------------------------

  // -------------------------------------------
  // -------------- CPU Memory -----------------
  // -------------------------------------------
  int iter = 0;
  int Npull = 0;
  REAL xoffs = 0.;
  REAL xtip = xint / dx, ytip = 0., ztip = 0.;

  // Arrays
  size_t SizeGrid = (Nx + 2) * (Ny + 2) * (Nz + 2);
  REAL *h_Psi = (REAL *)malloc(SizeGrid * sizeof(REAL));
  REAL *h_U = (REAL *)malloc(SizeGrid * sizeof(REAL));

  // ----------- h_Parameters storage -----------
  Constants h_Parameters[1];

  (*h_Parameters).sqrt2 = sqrt(2.);
  (*h_Parameters).PI = 4. * atan(1.);
  (*h_Parameters).dt = dt;
  (*h_Parameters).kcoeff = kcoeff;
  (*h_Parameters).omk = 1. - kcoeff;
  (*h_Parameters).opk = 1. + kcoeff;
  (*h_Parameters).Eps4 = Eps4;
  (*h_Parameters).D = D;
  (*h_Parameters).Lambda = Lambda;

  (*h_Parameters).Alpha = Alpha;
  (*h_Parameters).sAlpha = sin(Alpha);
  (*h_Parameters).cAlpha = cos(Alpha);
  (*h_Parameters).sAlpha2 = sin(Alpha) * sin(Alpha);
  (*h_Parameters).cAlpha2 = cos(Alpha) * cos(Alpha);
  (*h_Parameters).s2Alpha = sin(2. * Alpha);
  (*h_Parameters).c2Alpha = cos(2. * Alpha);
  (*h_Parameters).cAlphasAlpha = cos(Alpha) * sin(Alpha);

  (*h_Parameters).Beta = Beta;
  (*h_Parameters).sBeta = sin(Beta);
  (*h_Parameters).cBeta = cos(Beta);
  (*h_Parameters).sBeta2 = sin(Beta) * sin(Beta);
  (*h_Parameters).cBeta2 = cos(Beta) * cos(Beta);
  (*h_Parameters).s2Beta = sin(2. * Beta);
  (*h_Parameters).c2Beta = cos(2. * Beta);
  (*h_Parameters).cBetasBeta = cos(Beta) * sin(Beta);

  (*h_Parameters).Vp = Vp;
  (*h_Parameters).lT = lT;
  (*h_Parameters).lT0 = lT0;
  (*h_Parameters).lT1 = lT1;

  (*h_Parameters).W_microns = W_microns;
  (*h_Parameters).dx_microns = dx_microns;
  (*h_Parameters).Tau0_sec = Tau0_sec;
  (*h_Parameters).xint = xint;
  (*h_Parameters).x0 = x0;

  (*h_Parameters).xoffs = xoffs;
  (*h_Parameters).iter = iter;
  (*h_Parameters).Xtip = xtip;
  (*h_Parameters).Ytip = ytip;
  (*h_Parameters).Ztip = ztip;
  (*h_Parameters).RadY = 0.;
  (*h_Parameters).RadZ = 0.;

  // changed
  (*h_Parameters).slht = TIMELH;
  (*h_Parameters).flht = 0.;
  (*h_Parameters).Hamp = ampLH;
  // changed

  // for thermal drift
  (*h_Parameters).Tdtau = Theramltau / Tau0_sec;  // [tau_0]
  (*h_Parameters).Tddzt = Thermaldzt / W_microns; // [W]
                                                  //

  // For oscillatory pulling velocity
#if (OSC_Velocity)
  (*h_Parameters).OSCVamp0 = OSCVamp0;
  (*h_Parameters).OSCVamp = ((int)(OSC_t0) == 0) ? OSCVamp0 : 0.;
  (*h_Parameters).OSCNstep = 1; // for step-like forcing
#endif
  (*h_Parameters).Lenpull = 0.;

  // -------------------------------------------
  // -------------- GPU Memory -----------------
  // -------------------------------------------
  Constants *Parameters;
  cudaMalloc((void **)&Parameters, sizeof(Constants));

  REAL *Psi1, *U1, *Psi2, *U2;
  cudaMalloc((void **)&Psi1, SizeGrid * sizeof(REAL));
  cudaMalloc((void **)&Psi2, SizeGrid * sizeof(REAL));
  cudaMalloc((void **)&U1, SizeGrid * sizeof(REAL));
  cudaMalloc((void **)&U2, SizeGrid * sizeof(REAL));

  REAL *Unext, *Ucurr, *Ubuff;
  Ucurr = U1;
  Unext = U2;
  Ubuff = NULL;

  REAL *Psinext, *Psicurr, *Psibuff;
  Psicurr = Psi1;
  Psinext = Psi2;
  Psibuff = NULL;

  REAL *Phi;
  cudaMalloc((void **)&Phi, SizeGrid * sizeof(REAL));

  // for conserved noise
  REAL *Jnoise; // x,*Jnoisey,*Jnoisez;
  cudaMalloc((void **)&Jnoise, SizeGrid * sizeof(REAL));
  // cudaMalloc((void**)&Jnoisey,SizeGrid*sizeof(REAL));
  // cudaMalloc((void**)&Jnoisez,SizeGrid*sizeof(REAL));

  REAL *Xmax;
  cudaMalloc((void **)&Xmax, (Ny + 2) * (Nz + 2) * sizeof(REAL));

  curandState *devStates;
  cudaMalloc((void **)&devStates, SizeGrid * sizeof(curandState));

  // -------------- GPU BLOCKS -----------------
  int B[3] = {1, 1, 1};
  AutoBlockSize(B);
  const int BLOCK_SIZE_X = B[0];
  const int BLOCK_SIZE_Y = B[1];
  const int BLOCK_SIZE_Z = B[2];

  dim3 SizeBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 NumBlocks((Nx + 2) / BLOCK_SIZE_X, (Ny + 2) / BLOCK_SIZE_Y, (Nz + 2) / BLOCK_SIZE_Z);

  dim3 SizeBlockYZ(BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 NumBlocksYZ((Ny + 2) / BLOCK_SIZE_Y, (Nz + 2) / BLOCK_SIZE_Z);

  dim3 SizeOneBlock(1);
  dim3 NumOneBlock(1);

  // ------------- Initializations -------------
  setup_kernel<<<NumBlocks, SizeBlock>>>(time(NULL), devStates);
#if (INITfromFILE)
  printf("Initializing fields from files...\n");
  cudaMemcpy(Parameters, h_Parameters, sizeof(Constants), cudaMemcpyHostToDevice);
  InitFromFile(h_Psi, h_U, h_Parameters);
  cudaMemcpy(U1, h_U, SizeGrid * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(U2, h_U, SizeGrid * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(Psi1, h_Psi, SizeGrid * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(Psi2, h_Psi, SizeGrid * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(Parameters, h_Parameters, sizeof(Constants), cudaMemcpyHostToDevice);

  // At this point (1-Delta0) is stored into (*h_Parameters).x0
  GetXsl_YZ<<<NumBlocksYZ, SizeBlockYZ>>>(Psicurr, Xmax);
  GetXtip<<<NumOneBlock, SizeOneBlock>>>(Xmax, Parameters);
  cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);
  xtip = (*h_Parameters).Xtip;
  ytip = (*h_Parameters).Ytip;
  ztip = (*h_Parameters).Ztip;
  xint = xtip * dx;
  (*h_Parameters).xint = xint;
  x0 = xint - ((*h_Parameters).x0) * lT;
  (*h_Parameters).x0 = x0;
  cudaMemcpy(Parameters, h_Parameters, sizeof(Constants), cudaMemcpyHostToDevice);
#else
  printf("Initializing fields.\n");
  cudaMemcpy(Parameters, h_Parameters, sizeof(Constants), cudaMemcpyHostToDevice);
  Init<<<NumBlocks, SizeBlock>>>(Psi1, Psi2, U1, U2, Parameters, devStates);
  cudaMemcpy(h_U, Ucurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Psi, Psicurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);
#endif

  // -----------------
  // Tip data tracking
  // -----------------
  REAL Vel = 0., xprev = 0., RhoY = 0., RhoZ = 0.;
  REAL Delta = 1. - (xtip * dx - x0 + xoffs - Vp * dt * iter) / lT;
  REAL Omega = 1. / (1. - kcoeff) * (1. - kcoeff / (kcoeff + (1. - kcoeff) * Delta));

  // Make IterOutTip multiple of IterPull
  int IterOutTip = niter / NOUTTIP;
  IterOutTip = IterOutTip / IterPull;
  IterOutTip = IterOutTip * IterPull;
  if (IterOutTip == 0)
    IterOutTip = IterPull;

  // Output Tip /time
  char TipFileName[LENMAX];
  sprintf(TipFileName, "%s.tip.dat", OutputPrefix);
  FILE *TipF = fopen(TipFileName, "w");
  fprintf(TipF, "(1)Time[s] \t");           // 1
  fprintf(TipF, "(2)Delta \t");             // 2
  fprintf(TipF, "(3)Omega \t");             // 3
  fprintf(TipF, "(4)V_{Tip}[micron/s] \t"); // 4
  fprintf(TipF, "(5)x_{Tip} \t");           // 5
  fprintf(TipF, "(6)y_{Tip} \t");           // 6
  fprintf(TipF, "(7)z_{Tip} \t");           // 7
  fprintf(TipF, "(8)R_{Tip}/y \t");         // 8
  fprintf(TipF, "(9)R_{Tip}/z \t");         // 9
  fprintf(TipF, "(10)x_{Tip}/N_X \t");      // 10
  fprintf(TipF, "(11)x_{Tip}-x_L \t");      // 11
  fprintf(TipF, "(12)x_{offset} \t");       // 12
  fprintf(TipF, "(13)GradT[K/m] \t");       // 13

#if (OSC_Velocity)
  fprintf(TipF, "(14)V_p[micron/s] \t");     // 14
  fprintf(TipF, "(15)V_{OSC}[micron/s] \t"); // 15
  fprintf(TipF, "(16)V_{tot}[micron/s] \t"); // 16
#endif
  fprintf(TipF, "(17)TipPosition[micron] \t"); // 17

  fprintf(TipF, "\n");
  fclose(TipF);

#if (AMPSEARCH > 0)
  char FileNameAmp[LENMAX];
  FILE *OutFileAmp;

  int tip_dx = (int)(POSITION_0 * W_microns / dx_microns);
  int searchstep = (int)((tip_dx - 1) / AMPSEARCH);

  for (int sintf = searchstep; sintf < tip_dx; sintf += searchstep) {
    sprintf(FileNameAmp, "Amp_%s.%d.dat", OutputPrefix, sintf);
    OutFileAmp = fopen(FileNameAmp, "w");
    fprintf(OutFileAmp, "#dx = %g microns \n", dx_microns);                          // notification
    fprintf(OutFileAmp, "#SearchLoc = %d [dx] (Tip_x ~ %d [dx])\n", sintf, tip_dx);  // notification
    fprintf(OutFileAmp, "#Loc ~ %d [dx] (from the tip position)\n", tip_dx - sintf); // notification
    fprintf(OutFileAmp, "#time[s] \t");                                              // #1 time
    for (int j = 0; j < MAXINTFNUM; j++) {
      fprintf(OutFileAmp, "Interf_{y%d}[dx] \t", j); // # for the Y intersection
    }
    for (int k = 0; k < MAXINTFNUM; k++) {
      fprintf(OutFileAmp, "Interf_{z%d}[dx] \t", k); // # for the Z intersection
    }
    fprintf(OutFileAmp, "\n");
    fclose(OutFileAmp);
  }

#endif

  // ----------------------------
  OutputParameters(OutputPrefix, dt, niter, IterOutFields, B);
  // return 1;
  //  ----------------------------

  cudaFuncSetCacheConfig(Compute_P, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(Compute_U, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(BC, cudaFuncCachePreferL1);

  WriteFields(OutputPrefix, 0, h_Psi, h_U, h_Parameters, 1);
#if (NOUTFIELDS > 0)
  WriteFields(OutputPrefix, 0, h_Psi, h_U, h_Parameters, 0);
#endif
  // return 1;

  // ----------------------------
  //			Time loop
  // ----------------------------

  // temporary
  // niter=3;

  for (iter = 1; iter <= niter; iter = iter + 1) {

    // Oscillatory pulling velocity
    /*
#if(OSC_Velocity!=WITHOUT)

#if(OSC_Velocity==SIN)

    if(iter*dt*Tau0_sec>OSC_Onset) OSC_Vp=(Vpull + OSC_Amp*sin(2*PI*(iter*dt*Tau0_sec-OSC_Onset)/OSC_Period))*d0/Diff*a1*a2*E*E;

#elif(OSC_Velocity==CONST_V)

    if(iter*dt*Tau0_sec>OSC_Onset) OSC_Vp=(Vpull + OSC_Amp)*d0/Diff*a1*a2*E*E;

#elif(OSC_Velocity==LINEAR)

    if(iter*dt*Tau0_sec>OSC_Onset) OSC_Vp=(Vpull + OSC_Amp*(iter*dt*Tau0_sec/OSC_Onset-1.))*d0/Diff*a1*a2*E*E;
#endif

#endif
     */
    ////

    //////////////////////
    // Fields Evolution
    //////////////////////
    //		Compute_P<<<NumBlocks,SizeBlock>>>(Psicurr,Psinext,Phi,Ucurr,Jnoise,Parameters,devStates,OSC_Vp);
    Compute_P<<<NumBlocks, SizeBlock>>>(Psicurr, Psinext, Phi, Ucurr, Jnoise, Parameters, devStates);
#if (WALLEFFECT > 0)
    BC<<<NumBlocks, SizeBlock>>>(Psicurr, Psinext, Ucurr, Unext);
#endif
    Compute_U<<<NumBlocks, SizeBlock>>>(Psicurr, Psinext, Phi, Ucurr, Unext, Jnoise, Parameters);
    Psibuff = Psinext;
    Psinext = Psicurr;
    Psicurr = Psibuff;
    Ubuff = Unext;
    Unext = Ucurr;
    Ucurr = Ubuff;

    //////////////////
    // Pulling Back
    //////////////////
#if (PBFRAME == LAB)
    int off = 1;
    if ((Vp * iter * dt - xoffs) > dx)
#endif
#if (PBFRAME == TIP)
      GetXsl_YZ<<<NumBlocksYZ, SizeBlockYZ>>>(Psicurr, Xmax);
    GetXtip<<<NumOneBlock, SizeOneBlock>>>(Xmax, Parameters);
    cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);
    xtip = (*h_Parameters).Xtip;
    int off = (int)(xtip - xint / dx);
    // int off=(int)(xtip-POSITION_0/dx);	// This line can fix the pullback position but may cause instability at the start
    if (off)
#endif
    {
      PullBack<<<NumBlocks, SizeBlock>>>(Psicurr, Ucurr, Psinext, Unext, Parameters);
      Psibuff = Psinext;
      Psinext = Psicurr;
      Psicurr = Psibuff;
      Ubuff = Unext;
      Unext = Ucurr;
      Ucurr = Ubuff;
      xoffs += off * dx;
      Npull = Npull + 1;

      ////////////////////////
      // Output Movie files
      ////////////////////////
#if (NOUTFIELDS > 0)
      if (Npull % iPullFields == 0) {
        GetXsl_YZ<<<NumBlocksYZ, SizeBlockYZ>>>(Psicurr, Xmax);
        GetXtip<<<NumOneBlock, SizeOneBlock>>>(Xmax, Parameters);
        cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_U, Ucurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Psi, Psicurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);
        int indx = Npull / iPullFields;
#if (OUTINDEXSEC)
        indx = int(iter * dt * Tau0_sec);
#endif
        WriteFields(OutputPrefix, indx, h_Psi, h_U, h_Parameters, 0);
      }
#endif
    }

    //////////////////////
    // Output Tip data
    //////////////////////
    if (iter % IterOutTip == 0) {
      GetXsl_YZ<<<NumBlocksYZ, SizeBlockYZ>>>(Psicurr, Xmax);
      GetXtip<<<NumOneBlock, SizeOneBlock>>>(Xmax, Parameters);
      GetRtip<<<NumOneBlock, SizeOneBlock>>>(Xmax, Psicurr, Parameters);
      cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);

      xtip = (*h_Parameters).Xtip;
      ytip = (*h_Parameters).Ytip;
      ztip = (*h_Parameters).Ztip;
      RhoY = (*h_Parameters).RadY;
      RhoZ = (*h_Parameters).RadZ;

      REAL Zero = 0.;
      lT = (*h_Parameters).lT;

#if (OSC_Velocity == WITHOUT)
      Delta = 1. - (xtip * dx - x0 + xoffs - Vp * dt * iter) / lT;
#else
      Delta = 1. - (xtip * dx - x0 + xoffs - (*h_Parameters).Lenpull) / lT;
#endif
      Omega = 1. / (1. - kcoeff) * (1. - kcoeff / (kcoeff + (1. - kcoeff) * Delta));
      Vel = ((xprev != 0.) ? (xtip * dx + xoffs - xprev) / (IterOutTip * dt) : 1. / Zero);
      xprev = xtip * dx + xoffs;

      TipF = fopen(TipFileName, "a");
      fprintf(TipF, "%g \t", iter * dt * Tau0_sec);       //  1:	Time		[s]
      fprintf(TipF, "%g \t", Delta);                      //  2:	Delta		[-]
      fprintf(TipF, "%g \t", Omega);                      //  3:	Omega		[-]
      fprintf(TipF, "%g \t", Vel * W_microns / Tau0_sec); //  4:	Vel_Tip		[micron/s]
      fprintf(TipF, "%g \t", xtip);                       //  5:	xtip
      fprintf(TipF, "%g \t", ytip);                       //  6:	ytip
      fprintf(TipF, "%g \t", ztip);                       //  7:	ztip
      fprintf(TipF, "%g \t", RhoY * W_microns);           //  8:	R/y
      fprintf(TipF, "%g \t", RhoZ * W_microns);           //  9:	R/z
      fprintf(TipF, "%g \t", xtip / Nx);                  // 10:	xtip/Nx
      fprintf(TipF, "%g \t", Delta * lT * W_microns);     // 11:	xtip-xL
      fprintf(TipF, "%g \t", xoffs);                      // 12:	xoffs

      REAL Grad = GRAD0;
#if (TIME0 > 0)
      Grad = (iter * dt * Tau0_sec / 60. < TIME0) ? GRAD0 / 100. : (iter * dt * Tau0_sec / 60. > TIME1) ? GRAD1 / 100.
                                                                                                        : (GRAD0 + (GRAD1 - GRAD0) / (TIME1 - TIME0) * (iter * dt * Tau0_sec / 60. - TIME0)) / 100.;
#endif
      fprintf(TipF, "%g \t", Grad); // 13:	GradT		[K/m]

      // check pulling velocity
#if (OSC_Velocity)
      fprintf(TipF, "%g \t", (*h_Parameters).Vp * W_microns / Tau0_sec);                             // 14:	pulling velocity [micron/s]
      fprintf(TipF, "%g \t", (*h_Parameters).OSCVamp * W_microns / Tau0_sec);                        // 15:	oscillating velocity [micron/s]
      fprintf(TipF, "%g \t", ((*h_Parameters).Vp + (*h_Parameters).OSCVamp) * W_microns / Tau0_sec); // 16:	total pulling velocity [micron/s]
      fprintf(TipF, "%g \t", (xtip * dx - xint + xoffs - (*h_Parameters).Lenpull) * W_microns);      // 17: Tip position
#else
      fprintf(TipF, "%g \t", (xtip * dx - xint + xoffs - Vp * dt * iter * W_microns)); // 17: Tip position
#endif

      fprintf(TipF, "\n");
      fclose(TipF);

      // Numerical test on Delta

#if (Theramltau == 0 && IQY != 0 && IQZ != 0)
      if (Delta < 0. || Delta > 1.) {
        iter = niter + 1;
        printf("Looks like something went wrong.... (review your parameters, dx, W/d0, etc.).\n");
      }
#endif

      //================================================
      // Max amplitude search
      //================================================
#if (AMPSEARCH > 0)
      cudaMemcpy(h_Psi, Psicurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);

      for (int sintf = searchstep; sintf < tip_dx; sintf += searchstep) {
        WriteAmplitude(OutputPrefix, sintf, (iter * dt * Tau0_sec), h_Psi, MAXINTFNUM);
      }
#endif
    }

    //////////////////////
    // Output Save data
    //////////////////////
#if (NOUTSVG > 0)
    if (iter % IterOutSvg == 0) {
      GetXsl_YZ<<<NumBlocksYZ, SizeBlockYZ>>>(Psicurr, Xmax);
      GetXtip<<<NumOneBlock, SizeOneBlock>>>(Xmax, Parameters);
      cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_U, Ucurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_Psi, Psicurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);
      WriteFields(OutputPrefix, int(iter * dt * Tau0_sec), h_Psi, h_U, h_Parameters, 1);
    }
#endif

  } // end main time loop
  //---------------------

  BC<<<NumBlocks, SizeBlock>>>(Psicurr, Psinext, Ucurr, Unext);
  Psibuff = Psinext;
  Psinext = Psicurr;
  Psicurr = Psibuff;
  Ubuff = Unext;
  Unext = Ucurr;
  Ucurr = Ubuff;

  ///////////////////////////
  // Writing final results
  ///////////////////////////
  GetXsl_YZ<<<NumBlocksYZ, SizeBlockYZ>>>(Psicurr, Xmax);
  GetXtip<<<NumOneBlock, SizeOneBlock>>>(Xmax, Parameters);
  cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_U, Ucurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Psi, Psicurr, SizeGrid * sizeof(REAL), cudaMemcpyDeviceToHost);
  WriteFields(OutputPrefix, IndexFINAL, h_Psi, h_U, h_Parameters, 1);
#if (NOUTFIELDS > 0)
  WriteFields(OutputPrefix, (int)(TOTALTIME), h_Psi, h_U, h_Parameters, 0);
#endif

  clock_t end = clock();
  REAL CompTime = (end - begin) / CLOCKS_PER_SEC;
  OutputCompTime(OutputPrefix, CompTime);

#if (SUMMARIZE)
  /////////////////////////////////////
  // Summarize final results to file
  /////////////////////////////////////
  FILE *SummaryF = fopen("Summary.dat", "a");
  fprintf(SummaryF, "#File \t");
  fprintf(SummaryF, "Nx \tNy \tNz \t");
  fprintf(SummaryF, "DimX[microns] \tDimY[microns] \tDimZ[microns] \t");
  fprintf(SummaryF, "Delta \t");
  fprintf(SummaryF, "Omega \t");
  fprintf(SummaryF, "k \t");
  fprintf(SummaryF, "D[microns^2/s] \t");
  fprintf(SummaryF, "V[microns/s] \t");
  fprintf(SummaryF, "G_0[K/m] \t");
  fprintf(SummaryF, "G_1[K/m] \t");
  fprintf(SummaryF, "W/d_0 \t");
  fprintf(SummaryF, "dx/W \t");
  fprintf(SummaryF, "dx[microns] \t");
  fprintf(SummaryF, "time[s] \t");
  fprintf(SummaryF, "time_GPU[s] \t");
  fprintf(SummaryF, "\n");
  /////////////////////////////////////
  fprintf(SummaryF, "%s \t", OutputPrefix);
  fprintf(SummaryF, "%d \t%d \t%d \t", Nx, Ny, Nz);
  fprintf(SummaryF, "%g \t%g \t%g \t", Nx * dx_microns, Ny * dx_microns, Nz * dx_microns);
  fprintf(SummaryF, "%g \t", Delta);
  fprintf(SummaryF, "%g \t", Omega);
  fprintf(SummaryF, "%g \t", PARTITION);
  fprintf(SummaryF, "%g \t", DIFFUSION);
  fprintf(SummaryF, "%g \t", VELOCITY);
  fprintf(SummaryF, "%g \t", GRAD0);
  fprintf(SummaryF, "%g \t", GRAD1);
  fprintf(SummaryF, "%g \t", E);
  fprintf(SummaryF, "%g \t", dx);
  fprintf(SummaryF, "%g \t", dx_microns);
  fprintf(SummaryF, "%d \t", (int)(TOTALTIME));
  fprintf(SummaryF, "%d \t", (int)(CompTime));
  fprintf(SummaryF, "\n");
  /////////////////////////////////////
  fclose(SummaryF);
#endif

#if (COMPRESS)
  ///////////////////////////////////////
  // Compress results to *.tar.gz files
  ///////////////////////////////////////
  printf(" Compressing *%s* files ... ", OutputPrefix);
  char Command[1024];
  sprintf(Command, "tar -czf PF_%s.tar.gz PF_%s*vtk Psi_%s*Final*vtk Compo_%s*Final*vtk %s*Param* %s*tip*", OutputPrefix, OutputPrefix, OutputPrefix, OutputPrefix, OutputPrefix, OutputPrefix);
  system(Command);
  sprintf(Command, "rm PF_%s*vtk", OutputPrefix);
  system(Command);
  sprintf(Command, "tar -czf C_%s.tar.gz C_%s*vtk Psi_%s*Final*vtk Compo_%s*Final*vtk %s*Param* %s*tip*", OutputPrefix, OutputPrefix, OutputPrefix, OutputPrefix, OutputPrefix, OutputPrefix);
  system(Command);
  sprintf(Command, "rm C_%s*vtk", OutputPrefix);
  system(Command);
  sprintf(Command, "tar -czf CompoX_%s.tar.gz CompoX_%s*dat %s*Param* %s*tip*", OutputPrefix, OutputPrefix, OutputPrefix, OutputPrefix);
  system(Command);
  sprintf(Command, "rm CompoX_%s*dat", OutputPrefix);
  system(Command);
#if (AMPSEARCH > 0)
  sprintf(Command, "tar -czf Amp_%s.tar.gz Amp_%s*dat %s*Param* %s*tip*", OutputPrefix, OutputPrefix, OutputPrefix, OutputPrefix);
  system(Command);
  sprintf(Command, "rm Amp_%s*dat", OutputPrefix);
  system(Command);
  printf("done! \n");
#endif
#endif

  cudaFree(Phi);
  cudaFree(Jnoise);
  cudaFree(Psi1);
  cudaFree(Psi2);
  cudaFree(U1);
  cudaFree(U2);
  cudaFree(Parameters);
  cudaFree(devStates);

  free(h_Psi);
  free(h_U);

  return EXIT_SUCCESS;
}
//------------------------ End Main Program ------------------------

void OutputParameters(char Prefix[LENMAX], REAL dt, int niter, int IterOutFields, int *Bloc) {
  // ------------------------------------------------------
  REAL a1 = 5. * sqrt(2.) / 8.;
  REAL a2 = 47. / 75.;
  // ------------------------------------------------------
  REAL mc0 = LIQSLOPE * COMPOSITION;   // |Liquidus slope|*Nominal composition, K
  REAL DT0 = mc0 / PARTITION - mc0;    // Solidification range, K
  REAL d0 = GIBBSTHOMSON / DT0 * 1.e6; // Capillarity length @ T0, microns
  REAL lTherm0 = DT0 / GRAD0 * 1.e6;   // Thermal length, microns
#if (TIME0 > 0)
  REAL lTherm1 = DT0 / GRAD1 * 1.e6; // Thermal length, microns
#endif
  // -------------- Non-dimensional parameters ------------
  REAL D = a1 * a2 * E;
  REAL Lambda = a1 * E;
  REAL Vp = VELOCITY * d0 / DIFFUSION * a1 * a2 * E * E;
  REAL W_microns = E * d0;                   // [microns]
  REAL dx_microns = W_microns * dx;          // [microns]
  REAL Tau0_sec = Vp / VELOCITY * W_microns; // [seconds]
  // ----------------------------------------------------
  //		Output to screen
  // ----------------------------------------------------
  printf("\n----------------------------------------");
  printf("\n         SIMULATION PARAMETERS");
  printf("\n----------------------------------------\n");
  printf(" Vp    = %g microns/s\n", VELOCITY);
#if (TIME0 > 0)
  printf(" G     = %g to %g K/m\n", GRAD0, GRAD1);
  printf(" (at t = %d to %d h)\n", TIME0, TIME1);
#else
  printf(" G     = %g K/m\n", GRAD0);
#endif
  printf("\n");
  printf(" m     = %g K/UnitC\n", LIQSLOPE);
  printf(" c0    = %g UnitC\n", COMPOSITION);
  printf(" D     = %g microns^2/s\n", DIFFUSION);
  printf(" Gamma = %g Km\n", GIBBSTHOMSON);
  printf(" k     = %g\n", PARTITION);
  printf(" Eps4  = %g\n", ANISOTROPY);
#if ((AngleA != 0) || (AngleB != 0))
  printf("\n");
  printf(" Anisotropy rotation\n");
  printf(" Alpha = %d degrees\n", AngleA);
  printf(" Beta  = %d degrees\n", AngleB);
  printf("\n");
#endif
#if (TIME0 > 0)
  printf(" lT    = %g to %g microns\n", lTherm0, lTherm1);
  printf(" (at t = %d to %d min)\n", TIME0, TIME1);
#else
  printf(" lT    = %g microns\n", lTherm0);
#endif
  printf(" lD    = %g microns\n", DIFFUSION / VELOCITY);
#if (MIXd)
  printf(" limited lD = %d microns\n", MIXd);
#endif
  printf(" d0    = %g microns\n", d0);
  // changed
#if (TIMELH > 0)
  printf(" Local heating was on \n");
  printf(" from %i s with %g amplitude \n", TIMELH, ampLH);
#endif

#if (OSC_Velocity)

  printf(" \n");
  printf(" Imposed an oscillation of V_p \n");
  printf(" from t             = %g s\n", OSC_t0);
  printf(" Oscillation Amp    = %g [microns/s]\n", OSC_Vamp);
  printf("                    = %g \n", OSC_Vamp * d0 / DIFFUSION * a1 * a2 * E * E);

  if (OSC_Velocity == SINOSC || OSC_Velocity == STEPLIKE) {
    printf(" Oscillation period = %g [s]\n", OSC_Period);
  }

#endif

  // changed
  printf("-------------- DIMENSIONS --------------\n");
  printf(" Dimension/X = %g microns\n", Nx * dx_microns);
  printf(" Dimension/Y = %g microns\n", Ny * dx_microns);
  printf(" Dimension/Z = %g microns\n", Nz * dx_microns);
  printf(" Total time  = %g seconds\n", TOTALTIME);
  printf("------------ DIMENSIONLESS -------------\n");
  printf(" W/d0 = %g\n", E);
  printf(" W    = %g microns\n", W_microns);
  printf(" Tau0 = %g seconds\n", Tau0_sec);
  printf("----------------------------------------\n");
  printf(" D          = %g\n", D);
  printf(" Vp         = %g\n", Vp);
  printf(" lD         = %g\n", D / Vp);
  printf(" Lambda     = %g\n", Lambda);
  printf(" Total time = %g \n", niter * dt);
  printf("--------- INITIAL CONDITIONS -----------\n");
#if (!INITfromFILE)
  printf("Delta_0 = %g\n", UNDERCOOL_0);
  printf("x_int_0 = %g W\n", POSITION_0);
  printf("        = %g dx\n", POSITION_0 / dx);
  printf("        = %g microns\n", POSITION_0 * W_microns);
#if (IQY != 0 && IQZ != 0)
#if (IQY < 0 || IQZ < 0)
  printf("Initial random perturbation \n");
#else
  printf("Initial sine perturbation \n");
  printf("Wave nb /Y = %g \n", (REAL)(IQY));
  printf("Wave nb /Z = %g \n", (REAL)(IQZ));
#endif
  printf("Amplitude  = %g dx\n", AMP);
#endif
#else
  printf("From files *%s* \n", INIT_FILE);
#endif
  printf("------------ COMPUTATIONAL -------------\n");
  printf(" Nx+2 = %d\n", Nx + 2);
  printf(" Ny+2 = %d\n", Ny + 2);
  printf(" Nz+2 = %d\n", Nz + 2);
  printf(" dx = %g\n", dx);
  printf("    = %g microns\n", dx_microns);
  printf(" dt = %g\n", dt);
  printf("    = %g seconds\n", dt * Tau0_sec);
#if (BOUND_COND_Y == NOFLUX)
  printf(" Boundary Conditions /y : No-Flux\n");
#endif
#if (BOUND_COND_Y == PERIODIC)
  printf(" Boundary Conditions /y : Periodic\n");
#endif
#if (BOUND_COND_Y == HELICAL)
  printf(" Boundary Conditions /y : Helical\n");
#endif
#if (BOUND_COND_Z == NOFLUX)
  printf(" Boundary Conditions /z : No-Flux\n");
#endif
#if (BOUND_COND_Z == PERIODIC)
  printf(" Boundary Conditions /z : Periodic\n");
#endif
#if (BOUND_COND_Z == ANTISYM)
  printf(" Boundary Conditions /z : No-Flux (bottom), Anti-symmetric (top)\n");
#endif

#if (WALLEFFECT == WSLOPE)
  printf(" WALLSLOPE: %g \n", WALLSLOPE);
#endif
#if (WALLEFFECT == NzSLOPE)
  printf(" WALLSLOPE: %g at max Nz\n", WALLSLOPE);
#endif
#if (WALLEFFECT == N0SLOPE)
  printf(" WALLSLOPE: %g at min Nz\n", WALLSLOPE);
#endif

#if (NOISE != WITHOUT)
#if (NOISE == FLAT)
  printf(" Noise: Flat distribution\n");
  printf("        Amplitude = %g\n", Fnoise);
#endif
#if (NOISE == GAUSSIAN)
  printf(" Noise: Gaussian distribution\n");
  printf("        Amplitude = %g\n", Fnoise);
#endif
#if (NOISE == CONSERVE)
  printf(" Noise: Conserved Noise\n");
  printf("        Amplitude = %g microns^3\n", Fnoise);
  printf("        Fu0/dx_microns^3 = %g \n", Fnoise / (dx_microns * dx_microns * dx_microns));
#endif
#endif
  printf("----------------- GPU ------------------\n");
  printf("Block size /X = %d\n", Bloc[0]);
  printf("Block size /Y = %d\n", Bloc[1]);
  printf("Block size /Z = %d\n", Bloc[2]);
  printf("Thread/Block  = %d\n", Bloc[0] * Bloc[1] * Bloc[2]);
  printf("Number of Blocks /X = %g\n", ((Nx + 2.) / Bloc[0]));
  printf("Number of Blocks /Y = %g\n", ((Ny + 2.) / Bloc[1]));
  printf("Number of Blocks /Z = %g\n", ((Nz + 2.) / Bloc[2]));
  printf("Number of Blocks    = %d\n", ((Nx + 2) / Bloc[0] * (Ny + 2) / Bloc[1] * (Nz + 2) / Bloc[2]));
  printf("----------------------------------------\n");
  printf(" Time step input dt = %g\n", dt0);
  if (dt0 > dx * dx / 6. / D)
    printf("                 dt > dx^2/(6*D) ...\n");
  if (dt < dt0)
    printf("       =>        dt=%g\n", dt);
  printf(" Number of iterations: %d\n", niter);
  printf(" Output every %d iterations\n", IterOutFields);
  printf("----------------------------------------\n");
  printf("Pull-back : ");
#if (PBFRAME == LAB)
  printf("Lab frame \n");
#endif
#if (PBFRAME == TIP)
  printf("Tip frame \n");
#endif
  printf("----------------------------------------\n\n");
  // ----------------------------------------------------
  //		Output to file
  // ----------------------------------------------------
  char FileName[LENMAX];
  sprintf(FileName, "%s.Param.txt", Prefix);
  FILE *OutFile;
  OutFile = fopen(FileName, "w");
  fprintf(OutFile, "----------------------------------------");
  fprintf(OutFile, "\n         SIMULATION PARAMETERS");
  fprintf(OutFile, "\n----------------------------------------\n");
  fprintf(OutFile, " Vp    = %g microns/s\n", VELOCITY);
#if (TIME0 > 0)
  fprintf(OutFile, " G     = %g to %g K/m\n", GRAD0, GRAD1);
  fprintf(OutFile, " (at t = %d to %d min)\n", TIME0, TIME1);
#else
  fprintf(OutFile, " G     = %g K/m\n", GRAD0);
#endif
  fprintf(OutFile, "\n");
  fprintf(OutFile, " m     = %g K/UnitC\n", LIQSLOPE);
  fprintf(OutFile, " c0    = %g UnitC\n", COMPOSITION);
  fprintf(OutFile, " D     = %g microns^2/s\n", DIFFUSION);
  fprintf(OutFile, " Gamma = %g Km\n", GIBBSTHOMSON);
  fprintf(OutFile, " k     = %g\n", PARTITION);
  fprintf(OutFile, " Eps4  = %g\n", ANISOTROPY);
#if ((AngleA != 0) || (AngleB != 0))
  fprintf(OutFile, "\n");
  fprintf(OutFile, " Anisotropy rotation\n");
  fprintf(OutFile, " Alpha = %d degrees\n", AngleA);
  fprintf(OutFile, " Beta  = %d degrees\n", AngleB);
  fprintf(OutFile, "\n");
#endif
#if (TIME0 > 0)
  fprintf(OutFile, " lT    = %g to %g microns\n", lTherm0, lTherm1);
  fprintf(OutFile, " (at t = %d to %d min)\n", TIME0, TIME1);
#else
  fprintf(OutFile, " lT    = %g microns\n", lTherm0);
#endif
  fprintf(OutFile, " lD    = %g microns\n", DIFFUSION / VELOCITY);
#if (MIXd)
  fprintf(OutFile, " limited lD = %d microns\n", MIXd);
#endif
  fprintf(OutFile, " d0    = %g microns\n", d0);
// changed
#if (TIMELH > 0)
  fprintf(OutFile, " Local heating was on \n");
  fprintf(OutFile, " from %i s with %g amplitude \n", TIMELH, ampLH);
#endif

#if (OSC_Velocity)

  fprintf(OutFile, " \n");
  fprintf(OutFile, " Imposed an oscillation of V_p \n");
  fprintf(OutFile, " from t          = %g s\n", OSC_t0);
  fprintf(OutFile, " Oscillation Amp = %g [microns/s]\n", OSC_Vamp);
  fprintf(OutFile, "                 = %g \n", OSC_Vamp * d0 / DIFFUSION * a1 * a2 * E * E);

  if (OSC_Velocity == SINOSC || OSC_Velocity == STEPLIKE) {
    fprintf(OutFile, " Oscillation period = %g [s]\n", OSC_Period);
  }

#endif

  // changed
  fprintf(OutFile, "-------------- DIMENSIONS --------------\n");
  fprintf(OutFile, " Dimension/X = %g microns\n", Nx * dx_microns);
  fprintf(OutFile, " Dimension/Y = %g microns\n", Ny * dx_microns);
  fprintf(OutFile, " Dimension/Z = %g microns\n", Nz * dx_microns);
  fprintf(OutFile, " Total time  = %g seconds\n", TOTALTIME);
  fprintf(OutFile, "------------ DIMENSIONLESS -------------\n");
  fprintf(OutFile, " W/d0 = %g\n", E);
  fprintf(OutFile, " W    = %g microns\n", W_microns);
  fprintf(OutFile, " Tau0 = %g seconds\n", Tau0_sec);
  fprintf(OutFile, "----------------------------------------\n");
  fprintf(OutFile, " D          = %g\n", D);
  fprintf(OutFile, " Vp         = %g\n", Vp);
  fprintf(OutFile, " lD         = %g\n", D / Vp);
  fprintf(OutFile, " Lambda     = %g\n", Lambda);
  fprintf(OutFile, " Total time = %g \n", niter * dt);
  fprintf(OutFile, "--------- INITIAL CONDITIONS -----------\n");
#if (!INITfromFILE)
  fprintf(OutFile, "Delta_0    = %g\n", UNDERCOOL_0);
  fprintf(OutFile, "x_int_0    = %g W\n", POSITION_0);
  fprintf(OutFile, "           = %g dx\n", POSITION_0 / dx);
  fprintf(OutFile, "           = %g microns\n", POSITION_0 * W_microns);
#if (IQY != 0 && IQZ != 0)
#if (IQY < 0 || IQZ < 0)
  fprintf(OutFile, "Initial random perturbation \n");
#else
  fprintf(OutFile, "Initial sine perturbation \n");
  fprintf(OutFile, "Wave nb /Y = %g \n", (REAL)(IQY));
  fprintf(OutFile, "Wave nb /Z = %g \n", (REAL)(IQZ));
#endif
  fprintf(OutFile, "Amplitude  = %g dx\n", AMP);
#endif
#else
  fprintf(OutFile, "From files *%s* \n", INIT_FILE);
#endif
  fprintf(OutFile, "------------ COMPUTATIONAL -------------\n");
  fprintf(OutFile, " Nx+2 = %d\n", Nx + 2);
  fprintf(OutFile, " Ny+2 = %d\n", Ny + 2);
  fprintf(OutFile, " Nz+2 = %d\n", Nz + 2);
  fprintf(OutFile, " dx = %g\n", dx);
  fprintf(OutFile, "    = %g microns\n", dx_microns);
  fprintf(OutFile, " dt = %g\n", dt);
  fprintf(OutFile, "    = %g seconds\n", dt * Tau0_sec);
#if (BOUND_COND_Y == NOFLUX)
  fprintf(OutFile, " Boundary Conditions /y : No-Flux\n");
#endif
#if (BOUND_COND_Y == PERIODIC)
  fprintf(OutFile, " Boundary Conditions /y : Periodic\n");
#endif
#if (BOUND_COND_Y == HELICAL)
  fprintf(OutFile, " Boundary Conditions /y : Helical\n");
#endif
#if (BOUND_COND_Z == NOFLUX)
  fprintf(OutFile, " Boundary Conditions /z : No-Flux\n");
#endif
#if (BOUND_COND_Z == PERIODIC)
  fprintf(OutFile, " Boundary Conditions /z : Periodic\n");
#endif
#if (BOUND_COND_Z == ANTISYM)
  fprintf(OutFile, " Boundary Conditions /z : No-Flux (bottom), Anti-symmetric (top)\n");
#endif

#if (WALLEFFECT == WSLOPE)
  fprintf(OutFile, " WALLSLOPE: %g \n", WALLSLOPE);
#endif
#if (WALLEFFECT == NzSLOPE)
  fprintf(OutFile, " WALLSLOPE: %g at max Nz\n", WALLSLOPE);
#endif
#if (WALLEFFECT == N0SLOPE)
  fprintf(OutFile, " WALLSLOPE: %g at min Nz\n", WALLSLOPE);
#endif

#if (NOISE != WITHOUT)
#if (NOISE == FLAT)
  fprintf(OutFile, " Noise: Flat distribution\n");
  fprintf(OutFile, "        Amplitude = %g\n", Fnoise);
#endif
#if (NOISE == GAUSSIAN)
  fprintf(OutFile, " Noise: Gaussian distribution\n");
  fprintf(OutFile, "        Amplitude = %g\n", Fnoise);
#endif
#if (NOISE == CONSERVE)
  fprintf(OutFile, " Noise: Conserved Noise\n");
  fprintf(OutFile, "        Amplitude = %g microns^3\n", Fnoise);
  fprintf(OutFile, "        Fu0/dx_microns^3 = %g \n", Fnoise / (dx_microns * dx_microns * dx_microns));
#endif
#endif
  fprintf(OutFile, "----------------- GPU ------------------\n");
  fprintf(OutFile, "Block size /X = %d\n", Bloc[0]);
  fprintf(OutFile, "Block size /Y = %d\n", Bloc[1]);
  fprintf(OutFile, "Block size /Z = %d\n", Bloc[2]);
  fprintf(OutFile, "Thread/Block  = %d\n", Bloc[0] * Bloc[1] * Bloc[2]);
  fprintf(OutFile, "Number of Blocks /X = %g\n", ((Nx + 2.) / Bloc[0]));
  fprintf(OutFile, "Number of Blocks /Y = %g\n", ((Ny + 2.) / Bloc[1]));
  fprintf(OutFile, "Number of Blocks /Z = %g\n", ((Nz + 2.) / Bloc[2]));
  fprintf(OutFile, "Number of Blocks    = %d\n", ((Nx + 2) / Bloc[0] * (Ny + 2) / Bloc[1] * (Nz + 2) / Bloc[2]));
  fprintf(OutFile, "----------------------------------------\n");
  fprintf(OutFile, " Time step input dt = %g\n", dt0);
  if (dt0 > dx * dx / 6. / D)
    fprintf(OutFile, "                 dt > dx^2/(6*D) ...\n", dt);
  if (dt < dt0)
    fprintf(OutFile, "       =>        dt=%g\n", dt);
  fprintf(OutFile, " Number of iterations: %d\n", niter);
  fprintf(OutFile, " Output every %d iterations\n", IterOutFields);
  fprintf(OutFile, "----------------------------------------\n");
  fprintf(OutFile, "Pull-back : ");
#if (PBFRAME == LAB)
  fprintf(OutFile, "Lab frame \n");
#endif
#if (PBFRAME == TIP)
  fprintf(OutFile, "Tip frame \n");
#endif
  fprintf(OutFile, "----------------------------------------\n\n");
  fclose(OutFile);
  // ----------------------------------------------------
}
void OutputCompTime(char Prefix[LENMAX], REAL time) {
  REAL oneminute = 60.;
  REAL onehour = 60. * oneminute;
  REAL oneday = 24. * onehour;
  int days = (int)(time / oneday);
  int hours = (int)((time - days * oneday) / onehour);
  int minutes = (int)((time - days * oneday - hours * onehour) / oneminute);
  int seconds = (int)(time - days * oneday - hours * onehour - minutes * oneminute);

  char FileName[LENMAX];
  sprintf(FileName, "%s.Param.txt", Prefix);
  FILE *OutFile;
  OutFile = fopen(FileName, "a");
  fprintf(OutFile, "========================================\n");
  fprintf(OutFile, " Simulation time = %ds \n", (int)(time));
  if (minutes) {
    fprintf(OutFile, "                 = ");
    if (days) {
      fprintf(OutFile, "%dd", days);
    }
    if (hours) {
      fprintf(OutFile, "%dh", hours);
    }
    fprintf(OutFile, "%dm%ds \n", minutes, seconds);
  }
  fprintf(OutFile, "========================================\n");
  fclose(OutFile);
}

// Constants
#define sqrt2  (*Param).sqrt2
#define PI     (*Param).PI
#define kcoeff (*Param).kcoeff
#define omk    (*Param).omk
#define opk    (*Param).opk
#define Eps4   (*Param).Eps4
#define D      (*Param).D
#define Lambda (*Param).Lambda
#define dt     (*Param).dt

#define Alpha        (*Param).Alpha
#define sAlpha       (*Param).sAlpha
#define cAlpha       (*Param).cAlpha
#define sAlpha2      (*Param).sAlpha2
#define cAlpha2      (*Param).cAlpha2
#define s2Alpha      (*Param).s2Alpha
#define c2Alpha      (*Param).c2Alpha
#define cAlphasAlpha (*Param).cAlphasAlpha
#define Beta         (*Param).Beta
#define sBeta        (*Param).sBeta
#define cBeta        (*Param).cBeta
#define sBeta2       (*Param).sBeta2
#define cBeta2       (*Param).cBeta2
#define s2Beta       (*Param).s2Beta
#define c2Beta       (*Param).c2Beta
#define cBetasBeta   (*Param).cBetasBeta

#define Vp         (*Param).Vp
#define lT         (*Param).lT
#define lT0        (*Param).lT0
#define lT1        (*Param).lT1
#define W_microns  (*Param).W_microns
#define dx_microns (*Param).dx_microns
#define Tau0_sec   (*Param).Tau0_sec
#define xint       (*Param).xint
#define x0         (*Param).x0
#define Xtip       (*Param).Xtip
#define Ytip       (*Param).Ytip
#define Ztip       (*Param).Ztip
#define RadY       (*Param).RadY
#define RadZ       (*Param).RadZ
// Variables
#define xoffs (*Param).xoffs
#define iter  (*Param).iter
#define IMIN  0
#define IMAX  Nx + 1

// changed
// local heating
#if (TIMELH > 0)
#define slht (*Param).slht
#define flht (*Param).flht
#define Hamp (*Param).Hamp
#endif
// changed

// for thermal drift
#define Tdtau (*Param).Tdtau
#define Tddzt (*Param).Tddzt
//

// for oscillating Vp
#if (OSC_Velocity)
#define OSCVamp  (*Param).OSCVamp
#define OSCVamp0 (*Param).OSCVamp0
#define OSCNstep (*Param).OSCNstep // for step-like forcing
#endif
#define Lenpull (*Param).Lenpull

#if (BOUND_COND_Y == NOFLUX)
#define JMIN 0
#define JMAX Ny + 1
#endif
#if (BOUND_COND_Y == PERIODIC || BOUND_COND_Y == HELICAL)
#define JMIN Ny + 1
#define JMAX 0
#endif

#if (BOUND_COND_Z == NOFLUX || BOUND_COND_Z == ANTISYM)
#define KMIN 0
#define KMAX Nz + 1
#endif
#if (BOUND_COND_Z == PERIODIC)
#define KMIN Nz + 1
#define KMAX 0
#endif

#if (BOUND_COND_Z == ANTISYM) // function must appear at Zmax, i.e. each time KMAX appears
#define SYM(y) (Ny + 1 - (y))
#else
#define SYM(y) (y)
#endif

#if (BOUND_COND_Y == HELICAL) // function must appear at Ymin and Ymax, i.e. each time JMIN or JMAX appears
#define HEL(z) (Nz + 1 - (z))
#else
#define HEL(z) (z)
#endif

/////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Initializations ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
__global__ void Init(REAL *P1, REAL *P2, REAL *U1, REAL *U2, Constants *Param, curandState *state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  //----------------------------
  // Initial interface position
  //----------------------------
  REAL xintyz = xint / dx;
#if (IQY != 0 && IQZ != 0)
#if (IQY < 0 || IQZ < 0)
  //----------------------------------------------
  // Random initial perturbation of amplitude AMP
  //----------------------------------------------
  curandState localState;
  localState = state[pos(i, j, k)];
  REAL ran = curand_uniform_double(&localState);
  state[pos(i, j, k)] = localState;
  xintyz -= AMP * (ran - .5);
#else
  //----------------------------------------------------------------
  // Sine perturbation of amplitude AMP and wavenumbers IQY and IQZ
  //----------------------------------------------------------------
#if (BOUND_COND_Z == ANTISYM)
  xintyz -= AMP * (1. - cos(IQY * 3.14159 * j / Ny / 2.) * cos(IQZ * 3.14159 * k / Nz / 2.));
#else
  xintyz -= AMP * (1. + cos(IQY * 3.14159 * j / Ny) * cos(IQZ * 3.14159 * k / Nz));
#endif
#endif
#endif
  P1[pos(i, j, k)] = P2[pos(i, j, k)] = -(i - xintyz) * dx;

  //-----------------------
  // Initial concentration
  //-----------------------
  REAL phi = tanh(P1[pos(i, j, k)] / sqrt2);
  REAL c = (i < xintyz) ? (0.5 * (opk - omk * phi)) * (1. - (1. - UNDERCOOL_0) * omk) : (0.5 * (opk - omk * phi)) * (kcoeff + omk * (1. - (1. - UNDERCOOL_0)) * exp(-(i - xintyz) * dx * Vp / D));
  U1[pos(i, j, k)] = U2[pos(i, j, k)] = (2. * c - opk + omk * phi) / omk / (opk - omk * phi);
}

void InitFromFile(REAL *P, REAL *U, Constants *Param) {
#define pos_old(x, y, z) ((Ny_old + 2) * (Nz_old + 2) * (x) + (Nz_old + 2) * (y) + (z))
  char buff[15];
  int Nx_old, Ny_old, Nz_old, j0 = 1, k0 = 1;
  REAL *Pold, *Cold, Del0;

  char INIT_FILE_P[LENMAX];
  char INIT_FILE_C[LENMAX];
  sprintf(INIT_FILE_P, "Psi_%s.Final.vtk", INIT_FILE);
  sprintf(INIT_FILE_C, "Compo_%s.Final.vtk", INIT_FILE);

#if (INITfromFILE2)
#define pos_old2(x, y, z) ((Ny_old2 + 2) * (Nz_old2 + 2) * (x) + (Nz_old2 + 2) * (y) + (z))
  int Nx_old2, Ny_old2, Nz_old2;
  REAL *Pold2, *Cold2, Del0_2;

  char INIT_FILE_P2[LENMAX];
  char INIT_FILE_C2[LENMAX];
  sprintf(INIT_FILE_P2, "Psi_%s.Final.vtk", INIT_FILE2);
  sprintf(INIT_FILE_C2, "Compo_%s.Final.vtk", INIT_FILE2);
#endif

  // ===============================================
  // Get Initial Psi values ==> Pold[pos_old(i,j,k)]
  // ===============================================
  std::string line;
  std::ifstream InFileP(INIT_FILE_P);
  if (InFileP.is_open()) {
    getline(InFileP, line); //	""# vtk DataFile Version 3.0\n"
    getline(InFileP, line); //	"Delta %g\n"
    // ========================
    // Get Delta_0
    std::istringstream DeltaInit(line);
    DeltaInit >> buff;
    DeltaInit >> Del0;
    x0 = 1. - Del0; // At this point (1-Delta0) is stored into (*h_Parameters).x0
    // ======================
    getline(InFileP, line); //	"ASCII\n"
    getline(InFileP, line); //	"DATASET STRUCTURED_POINTS\n"
    getline(InFileP, line); //	"DIMENSIONS %d %d %d\n"
    // ========================
    // Get old x,y,z dimensions
    std::istringstream Dimensions(line);
    Dimensions >> buff;
    Dimensions >> Nx_old;
    Nx_old--;
    Nx_old--;
    Dimensions >> Ny_old;
    Ny_old--;
    Ny_old--;
    Dimensions >> Nz_old;
    Nz_old--;
    Nz_old--;
    Pold = (REAL *)malloc((Nx_old + 2) * (Ny_old + 2) * (Nz_old + 2) * sizeof(REAL));
    Cold = (REAL *)malloc((Nx_old + 2) * (Ny_old + 2) * (Nz_old + 2) * sizeof(REAL));
    // ======================
    getline(InFileP, line); //	"ASPECT_RATIO %f %f %f\n"
    getline(InFileP, line); //	"ORIGIN 0 0 0\n"
    getline(InFileP, line); //	"POINT_DATA %d\n"
    getline(InFileP, line); //	"SCALARS Psi double 1\n"
    getline(InFileP, line); //	"LOOKUP_TABLE default\n"
    getline(InFileP, line);
    std::istringstream ValuesP(line);
    for (int k = 0; k < Nz_old + 2; k++) {
      for (int j = 0; j < Ny_old + 2; j++) {
        for (int i = 0; i < Nx_old + 2; i++) {
          ValuesP >> Pold[pos_old(i, j, k)];
        }
      }
    }
    InFileP.close();
  } else
    std::cout << "Unable to open file " << INIT_FILE_P << std::endl;

  // =============================================
  // Get Initial C values ==> Cold[pos_old(i,j,k)]
  // =============================================
  std::ifstream InFileC(INIT_FILE_C);
  if (InFileC.is_open()) {

    getline(InFileC, line); //	"# vtk DataFile Version 3.0\n"
    getline(InFileC, line); //	"Title\n"
    getline(InFileC, line); //	"ASCII\n"
    getline(InFileC, line); //	"DATASET STRUCTURED_POINTS\n"
    getline(InFileC, line); //	"DIMENSIONS %d %d %d\n"
    getline(InFileC, line); //	"ASPECT_RATIO %f %f %f\n"
    getline(InFileC, line); //	"ORIGIN 0 0 0\n"
    getline(InFileC, line); //	"POINT_DATA %d\n"
    getline(InFileC, line); //	"SCALARS Psi double 1\n"
    getline(InFileC, line); //	"LOOKUP_TABLE default\n"
    getline(InFileC, line);

    std::istringstream ValuesC(line);
    for (int k = 0; k < Nz_old + 2; k++) {
      for (int j = 0; j < Ny_old + 2; j++) {
        for (int i = 0; i < Nx_old + 2; i++) {
          ValuesC >> Cold[pos_old(i, j, k)];
        }
      }
    }
    InFileC.close();
  } else
    std::cout << "Unable to open file " << INIT_FILE_C << std::endl;

#if (INITfromFILE2)
  // ===============================================
  // Get Initial Psi values ==> Pold2[pos_old2(i,j,k)]
  // ===============================================
  // std::string line;
  std::ifstream InFileP2(INIT_FILE_P2);
  if (InFileP2.is_open()) {
    getline(InFileP2, line); //	""# vtk DataFile Version 3.0\n"
    getline(InFileP2, line); //	"Delta %g\n"
    // ========================
    // Get Delta_0
    std::istringstream DeltaInit(line);
    DeltaInit >> buff;
    DeltaInit >> Del0_2;
    // x0=1.-Del0_2 ; // use the x0 extracted from the 1st file instead
    // ======================
    getline(InFileP2, line); //	"ASCII\n"
    getline(InFileP2, line); //	"DATASET STRUCTURED_POINTS\n"
    getline(InFileP2, line); //	"DIMENSIONS %d %d %d\n"
    // ========================
    // Get old x,y,z dimensions
    std::istringstream Dimensions(line);
    Dimensions >> buff;
    Dimensions >> Nx_old2;
    Nx_old2--;
    Nx_old2--;
    Dimensions >> Ny_old2;
    Ny_old2--;
    Ny_old2--;
    Dimensions >> Nz_old2;
    Nz_old2--;
    Nz_old2--;
    Pold2 = (REAL *)malloc((Nx_old2 + 2) * (Ny_old2 + 2) * (Nz_old2 + 2) * sizeof(REAL));
    Cold2 = (REAL *)malloc((Nx_old2 + 2) * (Ny_old2 + 2) * (Nz_old2 + 2) * sizeof(REAL));
    // ======================
    getline(InFileP2, line); //	"ASPECT_RATIO %f %f %f\n"
    getline(InFileP2, line); //	"ORIGIN 0 0 0\n"
    getline(InFileP2, line); //	"POINT_DATA %d\n"
    getline(InFileP2, line); //	"SCALARS Psi double 1\n"
    getline(InFileP2, line); //	"LOOKUP_TABLE default\n"
    getline(InFileP2, line);
    std::istringstream ValuesP(line);
    for (int k = 0; k < Nz_old2 + 2; k++) {
      for (int j = 0; j < Ny_old2 + 2; j++) {
        for (int i = 0; i < Nx_old2 + 2; i++) {
          ValuesP >> Pold2[pos_old2(i, j, k)];
        }
      }
    }
    InFileP2.close();
  } else
    std::cout << "Unable to open file " << INIT_FILE_P2 << std::endl;

  // =============================================
  // Get Initial C values ==> Cold2[pos_old2(i,j,k)]
  // =============================================
  std::ifstream InFileC2(INIT_FILE_C2);
  if (InFileC2.is_open()) {

    getline(InFileC2, line); //	"# vtk DataFile Version 3.0\n"
    getline(InFileC2, line); //	"Title\n"
    getline(InFileC2, line); //	"ASCII\n"
    getline(InFileC2, line); //	"DATASET STRUCTURED_POINTS\n"
    getline(InFileC2, line); //	"DIMENSIONS %d %d %d\n"
    getline(InFileC2, line); //	"ASPECT_RATIO %f %f %f\n"
    getline(InFileC2, line); //	"ORIGIN 0 0 0\n"
    getline(InFileC2, line); //	"POINT_DATA %d\n"
    getline(InFileC2, line); //	"SCALARS Psi double 1\n"
    getline(InFileC2, line); //	"LOOKUP_TABLE default\n"
    getline(InFileC2, line);

    std::istringstream ValuesC(line);
    for (int k = 0; k < Nz_old2 + 2; k++) {
      for (int j = 0; j < Ny_old2 + 2; j++) {
        for (int i = 0; i < Nx_old2 + 2; i++) {
          ValuesC >> Cold2[pos_old2(i, j, k)];
        }
      }
    }
    InFileC2.close();
  } else
    std::cout << "Unable to open file " << INIT_FILE_C2 << std::endl;
#endif

  // ======================================
  // Translate
  // Pold[pos_old(i,j,k)] ==> P[pos(i,j,k)]
  // Cold[pos_old(i,j,k)] ==> C[pos(i,j,k)]
  // ======================================
  REAL phi;
  REAL P000, P010, P001, P011, P100, P110, P101, P111;
  REAL C000, C010, C001, C011, C100, C110, C101, C111, c;
  REAL xold, yold, zold, xi, yi, zi; // changed
  REAL p0, p1, c0, c1;               // changed
  int iold, jold, kold, inew, jnew, knew;
  int imax = (Nx <= Nx_old) ? Nx : Nx_old;
  int jmax = (MULTIPLY_Y > 1) ? Ny / MULTIPLY_Y : Ny;
  int kmax = (MULTIPLY_Z > 1) ? Nz / MULTIPLY_Z : (MULTIPLY_Z < -1) ? -Nz / MULTIPLY_Z
                                                                    : Nz;

#if (CUTONEFOURTH)
  j0 = Ny_old / 2;
  k0 = Nz_old / 2;
#endif

#if (INITfromFILE2) // When init from 2 sources, translate the 1st source at first
  jmax = Ny_old;
  int jmax2 = Ny_old2;
#endif

  for (int i = 1; i <= imax; i++) {
    for (int j = 1; j <= jmax; j++) {
      for (int k = 1; k <= kmax; k++) {
        inew = i;
        jnew = (MIRROR_Y) ? jmax + 1 - j : j;
        knew = (MIRROR_Z) ? kmax + 1 - k : k;

        // iold=i;
        // changed
        xold = 1. + (Nx_old - 2.) / (REAL)(imax - 2.) * (i - 1.);
        iold = (int)(xold);
        xi = xold - iold;
        // changed

        yold = (REAL)(j0) + (REAL)(Ny_old - j0 - 1.) / (REAL)(jmax - 2.) * (j - 1.);
        jold = (int)(yold);
        yi = yold - jold;

        zold = (REAL)(k0) + (REAL)(Nz_old - k0 - 1.) / (REAL)(kmax - 2.) * (k - 1.);
        kold = (int)(zold);
        zi = zold - kold;

        P100 = Pold[pos_old(iold + 1, jold, kold)];
        P110 = Pold[pos_old(iold + 1, jold + 1, kold)];
        P101 = Pold[pos_old(iold + 1, jold, kold + 1)];
        P111 = Pold[pos_old(iold + 1, jold + 1, kold + 1)];

        P[pos(inew + 1, jnew, knew)] = P100 + (P110 - P100) * yi + (P101 - P100) * zi + (P111 - P110 - P101 + P100) * yi * zi;
        p1 = P[pos(inew + 1, jnew, knew)];

        P000 = Pold[pos_old(iold, jold, kold)];
        P010 = Pold[pos_old(iold, jold + 1, kold)];
        P001 = Pold[pos_old(iold, jold, kold + 1)];
        P011 = Pold[pos_old(iold, jold + 1, kold + 1)];

        P[pos(inew, jnew, knew)] = P000 + (P010 - P000) * yi + (P001 - P000) * zi + (P011 - P010 - P001 + P000) * yi * zi;
        p0 = P[pos(inew, jnew, knew)];

        P[pos(inew, jnew, knew)] = p0 + (p1 - p0) * xi;

        C000 = Cold[pos_old(iold, jold, kold)];
        C010 = Cold[pos_old(iold, jold + 1, kold)];
        C001 = Cold[pos_old(iold, jold, kold + 1)];
        C011 = Cold[pos_old(iold, jold + 1, kold + 1)];
        c0 = C000 + (C010 - C000) * yi + (C001 - C000) * zi + (C011 - C010 - C001 + C000) * yi * zi;

        C100 = Cold[pos_old(iold + 1, jold, kold)];
        C110 = Cold[pos_old(iold + 1, jold + 1, kold)];
        C101 = Cold[pos_old(iold + 1, jold, kold + 1)];
        C111 = Cold[pos_old(iold + 1, jold + 1, kold + 1)];
        c1 = C100 + (C110 - C100) * yi + (C101 - C100) * zi + (C111 - C110 - C101 + C100) * yi * zi;

        c = c0 + (c1 - c0) * xi;
        phi = tanh(P[pos(inew, jnew, knew)] / sqrt2);
        U[pos(inew, jnew, knew)] = (2. * c - opk + omk * phi) / omk / (opk - omk * phi);
      }
    }
  }
  // ====================================
  // Fill in the x dimension if Nx>Nx_old
  // ====================================
  if (Nx > Nx_old) {
    for (int i = imax + 1; i <= Nx; i++) {
      for (int j = 1; j <= jmax; j++) {
        for (int k = 1; k <= kmax; k++) {
          P[pos(i, j, k)] = P[pos(imax, j, k)];
          U[pos(i, j, k)] = U[pos(imax, j, k)];
        }
      }
    }
  }

#if (INITfromFILE2) // translate the 2nd source
  for (int i = 1; i <= imax; i++) {
    for (int j = 1; j <= jmax2; j++) {
      for (int k = 1; k <= kmax; k++) {
        inew = i;
        jnew = jmax + j;
        knew = (MIRROR_Z) ? kmax + 1 - k : k;

        // iold=i;
        // changed
        xold = 1. + (Nx_old2 - 2.) / (REAL)(imax - 2.) * (i - 1.);
        iold = (int)(xold);
        xi = xold - iold;
        // changed

        yold = (REAL)(j0) + (REAL)(Ny_old2 - j0 - 1.) / (REAL)(jmax2 - 2.) * (j - 1.);
        jold = (int)(yold);
        yi = yold - jold;

        zold = (REAL)(k0) + (REAL)(Nz_old2 - k0 - 1.) / (REAL)(kmax - 2.) * (k - 1.);
        kold = (int)(zold);
        zi = zold - kold;

        P100 = Pold2[pos_old2(iold + 1, jold, kold)];
        P110 = Pold2[pos_old2(iold + 1, jold + 1, kold)];
        P101 = Pold2[pos_old2(iold + 1, jold, kold + 1)];
        P111 = Pold2[pos_old2(iold + 1, jold + 1, kold + 1)];

        P[pos(inew + 1, jnew, knew)] = P100 + (P110 - P100) * yi + (P101 - P100) * zi + (P111 - P110 - P101 + P100) * yi * zi;
        p1 = P[pos(inew + 1, jnew, knew)];

        P000 = Pold2[pos_old2(iold, jold, kold)];
        P010 = Pold2[pos_old2(iold, jold + 1, kold)];
        P001 = Pold2[pos_old2(iold, jold, kold + 1)];
        P011 = Pold2[pos_old2(iold, jold + 1, kold + 1)];

        P[pos(inew, jnew, knew)] = P000 + (P010 - P000) * yi + (P001 - P000) * zi + (P011 - P010 - P001 + P000) * yi * zi;
        p0 = P[pos(inew, jnew, knew)];

        P[pos(inew, jnew, knew)] = p0 + (p1 - p0) * xi;

        C000 = Cold2[pos_old2(iold, jold, kold)];
        C010 = Cold2[pos_old2(iold, jold + 1, kold)];
        C001 = Cold2[pos_old2(iold, jold, kold + 1)];
        C011 = Cold2[pos_old2(iold, jold + 1, kold + 1)];
        c0 = C000 + (C010 - C000) * yi + (C001 - C000) * zi + (C011 - C010 - C001 + C000) * yi * zi;

        C100 = Cold2[pos_old2(iold + 1, jold, kold)];
        C110 = Cold2[pos_old2(iold + 1, jold + 1, kold)];
        C101 = Cold2[pos_old2(iold + 1, jold, kold + 1)];
        C111 = Cold2[pos_old2(iold + 1, jold + 1, kold + 1)];
        c1 = C100 + (C110 - C100) * yi + (C101 - C100) * zi + (C111 - C110 - C101 + C100) * yi * zi;

        c = c0 + (c1 - c0) * xi;
        phi = tanh(P[pos(inew, jnew, knew)] / sqrt2);
        U[pos(inew, jnew, knew)] = (2. * c - opk + omk * phi) / omk / (opk - omk * phi);
      }
    }
  }
#endif

  // ==========
  // Multiply/Z
  // ==========
  int Mz = (MULTIPLY_Z < 0) ? -MULTIPLY_Z : MULTIPLY_Z;
  if (MULTIPLY_Z > 1 || MULTIPLY_Z < -1) {
    for (int i = 1; i <= Nx; i++) {
      for (int j = 1; j <= jmax; j++) {
        for (int iz = 2; iz <= Mz; iz++) {
          // Mirror on y if (MULTIPLY_Z<0 && (iz%4==2 || iz%4==3))
          jnew = (MULTIPLY_Z < 0 && (iz % 4 == 2 || iz % 4 == 3)) ? jmax + 1 - j : j;
          for (int k = 1; k <= kmax; k++) {
            // Mirror on z if(iz%2==0)
            knew = (iz % 2 == 0) ? kmax * (iz - 1) + kmax + 1 - k : kmax * (iz - 1) + k;
            P[pos(i, jnew, knew)] = P[pos(i, j, k)];
            U[pos(i, jnew, knew)] = U[pos(i, j, k)];
          }
        }
      }
    }
  }
  // ==========
  // Multiply/Y
  // ==========
  int My = MULTIPLY_Y;
  if (MULTIPLY_Y > 1) {
    for (int i = 1; i <= Nx; i++) {
      for (int iy = 2; iy <= My; iy++) {
        for (int j = 1; j <= jmax; j++) {
          // Mirror on y if(iy%2==0)
          jnew = (iy % 2 == 0) ? jmax * (iy - 1) + jmax + 1 - j : jmax * (iy - 1) + j;
#if (PERIODIC_Y)
          jnew = jmax * (iy - 1) + j;
#endif
          for (int k = 1; k <= Nz; k++) {
            knew = k;
            P[pos(i, jnew, knew)] = P[pos(i, j, k)];
            U[pos(i, jnew, knew)] = U[pos(i, j, k)];
          }
        }
      }
    }
  }

  // ===================
  // Boundary conditions
  // ===================
  for (int j = 1; j <= Ny; j++) {
    for (int k = 1; k <= Nz; k++) {
      P[pos(IMIN, j, k)] = P[pos(1, j, k)];
      U[pos(IMIN, j, k)] = U[pos(1, j, k)];
      P[pos(IMAX, j, k)] = P[pos(Nx, j, k)];
      U[pos(IMAX, j, k)] = U[pos(Nx, j, k)];
    }
  }
  for (int i = 1; i <= Nx; i++) {
    for (int k = 1; k <= Nz; k++) {
      P[pos(i, JMIN, HEL(k))] = P[pos(i, 1, k)];
      U[pos(i, JMIN, HEL(k))] = U[pos(i, 1, k)];
      P[pos(i, JMAX, HEL(k))] = P[pos(i, Ny, k)];
      U[pos(i, JMAX, HEL(k))] = U[pos(i, Ny, k)];
    }
  }
  for (int i = 1; i <= Nx; i++) {
    for (int j = 1; j <= Ny; j++) {
      P[pos(i, j, KMIN)] = P[pos(i, j, 1)];
      U[pos(i, j, KMIN)] = U[pos(i, j, 1)];
      P[pos(i, SYM(j), KMAX)] = P[pos(i, j, Nz)];
      U[pos(i, SYM(j), KMAX)] = U[pos(i, j, Nz)];
    }
  }
  for (int i = 1; i <= Nx; i++) {
    P[pos(i, JMIN, HEL(KMIN))] = P[pos(i, 1, 1)];
    U[pos(i, JMIN, HEL(KMIN))] = U[pos(i, 1, 1)];
    P[pos(i, JMAX, HEL(KMIN))] = P[pos(i, Ny, 1)];
    U[pos(i, JMAX, HEL(KMIN))] = U[pos(i, Ny, 1)];
    P[pos(i, SYM(JMIN), HEL(KMAX))] = P[pos(i, 1, Nz)];
    U[pos(i, SYM(JMIN), HEL(KMAX))] = U[pos(i, 1, Nz)];
    P[pos(i, SYM(JMAX), HEL(KMAX))] = P[pos(i, Ny, Nz)];
    U[pos(i, SYM(JMAX), HEL(KMAX))] = U[pos(i, Ny, Nz)];
  }
  for (int j = 1; j <= Ny; j++) {
    P[pos(IMIN, j, KMIN)] = P[pos(1, j, 1)];
    U[pos(IMIN, j, KMIN)] = U[pos(1, j, 1)];
    P[pos(IMAX, j, KMIN)] = P[pos(Nx, j, 1)];
    U[pos(IMAX, j, KMIN)] = U[pos(Nx, j, 1)];
    P[pos(IMIN, SYM(j), KMAX)] = P[pos(1, j, Nz)];
    U[pos(IMIN, SYM(j), KMAX)] = U[pos(1, j, Nz)];
    P[pos(IMAX, SYM(j), KMAX)] = P[pos(Nx, j, Nz)];
    U[pos(IMAX, SYM(j), KMAX)] = U[pos(Nx, j, Nz)];
  }
  for (int k = 1; k <= Nz; k++) {
    P[pos(IMIN, JMIN, HEL(k))] = P[pos(1, 1, k)];
    U[pos(IMIN, JMIN, HEL(k))] = U[pos(1, 1, k)];
    P[pos(IMIN, JMAX, HEL(k))] = P[pos(1, Ny, k)];
    U[pos(IMIN, JMAX, HEL(k))] = U[pos(1, Ny, k)];
    P[pos(IMAX, JMIN, HEL(k))] = P[pos(Nx, 1, k)];
    U[pos(IMAX, JMIN, HEL(k))] = U[pos(Nx, 1, k)];
    P[pos(IMAX, JMAX, HEL(k))] = P[pos(Nx, Ny, k)];
    U[pos(IMAX, JMAX, HEL(k))] = U[pos(Nx, Ny, k)];
  }

  P[pos(IMIN, JMIN, HEL(KMIN))] = P[pos(1, 1, 1)];
  U[pos(IMIN, JMIN, HEL(KMIN))] = U[pos(1, 1, 1)];
  P[pos(IMIN, JMAX, HEL(KMIN))] = P[pos(1, Ny, 1)];
  U[pos(IMIN, JMAX, HEL(KMIN))] = U[pos(1, Ny, 1)];
  P[pos(IMAX, JMIN, HEL(KMIN))] = P[pos(Nx, 1, 1)];
  U[pos(IMAX, JMIN, HEL(KMIN))] = U[pos(Nx, 1, 1)];
  P[pos(IMAX, JMAX, HEL(KMIN))] = P[pos(Nx, Ny, 1)];
  U[pos(IMAX, JMAX, HEL(KMIN))] = U[pos(Nx, Ny, 1)];
  P[pos(IMIN, SYM(JMIN), HEL(KMAX))] = P[pos(1, 1, Nz)];
  U[pos(IMIN, SYM(JMIN), HEL(KMAX))] = U[pos(1, 1, Nz)];
  P[pos(IMIN, SYM(JMAX), HEL(KMAX))] = P[pos(1, Ny, Nz)];
  U[pos(IMIN, SYM(JMAX), HEL(KMAX))] = U[pos(1, Ny, Nz)];
  P[pos(IMAX, SYM(JMIN), HEL(KMAX))] = P[pos(Nx, 1, Nz)];
  U[pos(IMAX, SYM(JMIN), HEL(KMAX))] = U[pos(Nx, 1, Nz)];
  P[pos(IMAX, SYM(JMAX), HEL(KMAX))] = P[pos(Nx, Ny, Nz)];
  U[pos(IMAX, SYM(JMAX), HEL(KMAX))] = U[pos(Nx, Ny, Nz)];

  free(Pold);
  free(Cold);
}

__global__ void setup_kernel(unsigned long long seed, curandState *state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  curand_init(seed, pos(i, j, k), 0, &state[pos(i, j, k)]);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Computation /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
#define omp2 (1. - phi * phi)

//__global__ void Compute_P(REAL *Pcurr,REAL *Pnext,REAL *F,REAL *Cucurr,REAL *Jn,Constants *Param,curandState *state, REAL OSC_Vp)
__global__ void Compute_P(REAL *Pcurr, REAL *Pnext, REAL *F, REAL *Cucurr, REAL *Jn, Constants *Param, curandState *state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  REAL psi = Pcurr[pos(i, j, k)];
  REAL phi = F[pos(i, j, k)] = tanh(psi / sqrt2);

  if (i * j * k * (i - Nx - 1) * (j - Ny - 1) * (k - Nz - 1)) //	i.e. (i!=0 && j!=0 && k!=0 && i!=Nx+1 && j!=Ny+1 && k!=Nz+1)
  {
#if (WALLEFFECT > 0)

    bool wcheck = 0;

#if (WALLEFFECT == WSLOPE)

    if ((k - 1) * (k - Nz) == 0) {
      wcheck = 1;
    }

#endif

#if (WALLEFFECT == NzSLOPE)
    if ((k - Nz) == 0) {
      wcheck = 1;
    }
#endif

#if (WALLEFFECT == N0SLOPE)
    if ((k - 1) == 0) {
      wcheck = 1;
    }
#endif

    if (wcheck == 1) {

      Pnext[pos(i, j, k)] = Pnext[pos(i, j, k)];

    } else
#endif
    {

#define dx2      (2. * dx)
#define dxs      (dx * dx)
#define dxs4     (4. * dx * dx)
#define prefac   (1. - 3. * Eps4)
#define prefacs  (prefac * prefac)
#define eps      (4. * Eps4 / (1 - 3. * Eps4))
#define foueps   (4. * eps)
#define sixeps   (16. * eps)
#define sixepssq (16. * eps * eps)

      REAL anis = 0;
      REAL xi = 1;

      REAL phx = ((Pcurr[pos(i + 1, j, k)] - Pcurr[pos(i - 1, j, k)]) / dx2);
      REAL phy = ((Pcurr[pos(i, j + 1, k)] - Pcurr[pos(i, j - 1, k)]) / dx2);
      REAL phz = ((Pcurr[pos(i, j, k + 1)] - Pcurr[pos(i, j, k - 1)]) / dx2);

      REAL phxx = (Pcurr[pos(i + 1, j, k)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(i - 1, j, k)]) / dxs - sqrt2 * phi * phx * phx;
      REAL phyy = (Pcurr[pos(i, j + 1, k)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(i, j - 1, k)]) / dxs - sqrt2 * phi * phy * phy;
      REAL phzz = (Pcurr[pos(i, j, k + 1)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(i, j, k - 1)]) / dxs - sqrt2 * phi * phz * phz;

      REAL phxy = (Pcurr[pos(i + 1, j + 1, k)] - Pcurr[pos(i - 1, j + 1, k)] - Pcurr[pos(i + 1, j - 1, k)] + Pcurr[pos(i - 1, j - 1, k)]) / dxs4 - sqrt2 * phi * phx * phy;
      REAL phyz = (Pcurr[pos(i, j + 1, k + 1)] - Pcurr[pos(i, j - 1, k + 1)] - Pcurr[pos(i, j + 1, k - 1)] + Pcurr[pos(i, j - 1, k - 1)]) / dxs4 - sqrt2 * phi * phy * phz;
      REAL phxz = (Pcurr[pos(i + 1, j, k + 1)] - Pcurr[pos(i - 1, j, k + 1)] - Pcurr[pos(i + 1, j, k - 1)] + Pcurr[pos(i - 1, j, k - 1)]) / dxs4 - sqrt2 * phi * phz * phx;

      REAL dphix = phx * cAlpha - phy * sAlpha;
      REAL dphiy = phx * sAlpha * cBeta + phy * cAlpha * cBeta - phz * sBeta;
      REAL dphiz = phx * sAlpha * sBeta + phy * cAlpha * sBeta + phz * cBeta;

      REAL pxy = dphix * dphiy;
      REAL pxz = dphix * dphiz;
      REAL pyz = dphiy * dphiz;

      REAL px2 = dphix * dphix;
      REAL py2 = dphiy * dphiy;
      REAL pz2 = dphiz * dphiz;
      REAL xnorm = px2 + py2 + pz2;

      REAL dphixx = phxx * cAlpha2 + phyy * sAlpha2 - phxy * s2Alpha;
      REAL dphiyy = (phxx * sAlpha2 + phyy * cAlpha2) * cBeta2 + phzz * sBeta2 + 2. * cBeta * (sAlpha * (phxy * cAlpha * cBeta - phxz * sBeta) - phyz * cAlpha * sBeta);
      REAL dphizz = (phxx * sAlpha2 + phyy * cAlpha2) * sBeta2 + phzz * cBeta2 + 2. * sBeta * (sAlpha * (phxy * cAlpha * sBeta + phxz * cBeta) + phyz * cAlpha * cBeta);

      REAL sumnn = dphixx + dphiyy + dphizz;

      //		if(fabs(omp2)>1.e-10)
      if (omp2 > 0.) {
        REAL xnorm2 = xnorm * xnorm;
        REAL xnorm3 = xnorm2 * xnorm;

        // rotate
        REAL dphixy = cAlpha * (cBeta * (phxx * sAlpha + phxy * cAlpha) - phxz * sBeta) - sAlpha * (cBeta * (phxy * sAlpha + phyy * cAlpha) - phyz * sBeta);

        REAL dphiyz = sBeta * cBeta * (phxx * sAlpha2 + phyy * cAlpha2 - phzz) + phxy * s2Alpha * sBeta * cBeta + c2Beta * (phxz * sAlpha + phyz * cAlpha);

        REAL dphixz = cAlpha * (sBeta * (phxx * sAlpha + phxy * cAlpha) + cBeta * phxz) - sAlpha * (sBeta * (phxy * sAlpha + phyy * cAlpha) + phyz * cBeta);

        REAL px4 = px2 * px2;
        REAL px6 = px2 * px4;
        REAL py4 = py2 * py2;
        REAL py6 = py2 * py4;
        REAL pz4 = pz2 * pz2;
        REAL pz6 = pz2 * pz4;
        REAL px2y2 = px2 * py2;
        REAL px4y2 = px4 * py2;
        REAL px2z2 = px2 * pz2;
        REAL px4z2 = px4 * pz2;
        REAL py2z2 = py2 * pz2;
        REAL py4x2 = py4 * px2;
        REAL py4z2 = py4 * pz2;
        REAL pz4x2 = pz4 * px2;
        REAL pz4y2 = pz4 * py2;
        REAL px2y2z2 = px2 * py2 * pz2;

        REAL frac = (px4 + py4 + pz4) / xnorm2;
        xi = 1. + eps * frac;
        REAL fouepsxi = foueps * xi;

#define t1x (dphixx * (px4y2 + 4. * py4x2 - py6 + px4z2 + 6. * px2y2z2 - pz4y2 - py4z2 + 4. * pz4x2 - pz6))
#define t1y (dphiyy * (py4z2 + 4. * pz4y2 - pz6 + py4x2 + 6. * px2y2z2 - px4z2 - pz4x2 + 4. * px4y2 - px6))
#define t1z (dphizz * (pz4x2 + 4. * px4z2 - px6 + pz4y2 + 6. * px2y2z2 - py4x2 - px4y2 + 4. * py4z2 - py6))

        REAL termx = px2y2 - py4 - pz4 + px2z2;
        REAL termy = px2y2 - px4 - pz4 + py2z2;
        REAL termz = py2z2 - px4 - py4 + px2z2;

#define t2x  (dphixx * px2 * termx * termx)
#define t2y  (dphiyy * py2 * termy * termy)
#define t2z  (dphizz * pz2 * termz * termz)
#define t1xy (dphixy * pxy * (-2. * px2y2 - px2z2 - py2z2 + pz4))
#define t1xz (dphixz * pxz * (-2. * px2z2 - px2y2 - py2z2 + py4))
#define t1yz (dphiyz * pyz * (-2. * py2z2 - px2y2 - px2z2 + px4))
#define t2xy (2. * dphixy * pxy * termx * termy)
#define t2xz (2. * dphixz * pxz * termx * termz)
#define t2yz (2. * dphiyz * pyz * termy * termz)

        anis = fouepsxi * (t1x + t1y + t1z) / xnorm3 + sixeps * xi * (t1xy + t1xz + t1yz) / xnorm3 + sixepssq * ((t2x + t2y + t2z) / xnorm2 + (t2xy + t2xz + t2yz) / xnorm2) / xnorm3;
      }

      // ----
      // dpdt
      // ----

#if (TIME0 > 0)
      lT = (iter * dt * Tau0_sec < TIME0) ? lT0 : (iter * dt * Tau0_sec > TIME1) ? lT1
                                                                                 : lT0 + (lT1 - lT0) / (TIME1 - TIME0) * (iter * dt * Tau0_sec - TIME0);
#endif

      // REAL temp=(i*dx+xoffs-x0-Vp*dt*iter)/lT;

      // Oscillatory pulling velocity
      /*
#if(OSC_Velocity==WITHOUT)
  REAL Vp_temp=Vp;
#else
  REAL Vp_temp=OSC_Vp;
#endif
       */

      ///////
#if (Theramltau == 0)
      // no thermal effect
#if (OSC_Velocity == WITHOUT)
#if (IfTemporal)
      REAL temp = (i * dx + 0.5 * j + 0.5 * k + xoffs - x0 - Vp * dt * iter) / lT;
#else
      REAL temp = (i * dx + xoffs - x0 - Vp * dt * iter) / lT;
      // REAL temp=(i*dx+xoffs-x0-Lenpull)/lT;
#endif

#else

      /*
      if (i==2 && j==2 && k ==2)
      {
          printf("iter = %d, Vp = %g, OSC amp= %g (%g), Vp+OSCVamp = %g\n", iter, Vp, OSCVamp,OSCVamp0, Vp+OSCVamp);
      }
       */

      // REAL temp=(i*dx+xoffs-x0-(Vp+OSCVamp)*dt*iter)/lT;

      REAL temp = (i * dx + xoffs - x0 - Lenpull) / lT;

      // printf("Vp = %g, OSC amp= %g (%g), Vp+OSCVamp = %g\n", Vp, OSCVamp,OSCVamp0, Vp+OSCVamp);
#endif

#else

      // thermal drift effect: temp = temp + Tddzt * (1- exp( -1.*iter*dt/Tdtau ) )
      REAL temp = (i * dx + xoffs - x0 - Vp * dt * iter + Tddzt * (1. - exp(-1. * iter * dt / Tdtau))) / lT;

#endif

      ///////

      REAL Tau = (temp > 1.) ? kcoeff : (1. - omk * temp);

      Pnext[pos(i, j, k)] = psi + dt * (xi * xi * prefacs * sumnn + sqrt2 * phi - sqrt2 * omp2 * Lambda * (Cucurr[pos(i, j, k)] + temp) + anis * prefacs) / (xi * xi * prefacs) / (Tau);

#if (WALLEFFECT == WSLOPE)
      if (k == 2) {
        Pnext[pos(i, j, 1)] = Pnext[pos(i, j, k)] - WALLSLOPE * dx;
      } else if (k == (Nz - 1)) {
        Pnext[pos(i, j, Nz)] = Pnext[pos(i, j, k)] - WALLSLOPE * dx;
      }
#endif
#if (WALLEFFECT == NzSLOPE)

      if (k == (Nz - 1)) {
        Pnext[pos(i, j, Nz)] = Pnext[pos(i, j, k)] - WALLSLOPE * dx;
      }
#endif

#if (WALLEFFECT == N0SLOPE)
      if (k == 2) {
        Pnext[pos(i, j, 1)] = Pnext[pos(i, j, k)] - WALLSLOPE * dx;
      }
#endif
    }

#if (NOISE != WITHOUT)
#if (tmax_NOISE > 0)
    if (iter * dt * Tau0_sec < 60. * tmax_NOISE)
#endif
    {
#if (NOISE == GAUSSIAN)
      curandState localState;
      localState = state[pos(i, j, k)];
      REAL ran1 = curand_normal(&localState);
      Pnext[pos(i, j, k)] += sqrt(2. * Fnoise * dt / dxs) * ran1;
      state[pos(i, j, k)] = localState;
#endif
#if (NOISE == FLAT)
      curandState localState;
      localState = state[pos(i, j, k)];
      REAL ran1 = curand_uniform_double(&localState);
      Pnext[pos(i, j, k)] += Fnoise * sqrt(dt) * (ran1 - .5);
      state[pos(i, j, k)] = localState;

#endif
    }
#endif

    // BC at (x=0.5) or (x=Nx+0.5)
#if (WALLEFFECT == 0)
    if (i == 1) {
      Pnext[pos(IMIN, j, k)] = Pnext[pos(1, j, k)];
      Cucurr[pos(IMIN, j, k)] = Cucurr[pos(1, j, k)];
      if (j == 1) {
        Pnext[pos(IMIN, JMIN, HEL(k))] = Pnext[pos(1, 1, k)];
        Cucurr[pos(IMIN, JMIN, HEL(k))] = Cucurr[pos(1, 1, k)];
      } else if (j == Ny) {
        Pnext[pos(IMIN, JMAX, HEL(k))] = Pnext[pos(1, Ny, k)];
        Cucurr[pos(IMIN, JMAX, HEL(k))] = Cucurr[pos(1, Ny, k)];
      }
      if (k == 1) {
        Pnext[pos(IMIN, j, KMIN)] = Pnext[pos(1, j, 1)];
        Cucurr[pos(IMIN, j, KMIN)] = Cucurr[pos(1, j, 1)];
      } else if (k == Nz) {
        Pnext[pos(IMIN, SYM(j), KMAX)] = Pnext[pos(1, j, Nz)];
        Cucurr[pos(IMIN, SYM(j), KMAX)] = Cucurr[pos(1, j, Nz)];
      }
    } else if (i == Nx) {
      Pnext[pos(IMAX, j, k)] = Pnext[pos(Nx, j, k)] - dx;
      Cucurr[pos(IMAX, j, k)] = Cucurr[pos(Nx, j, k)];
      if (j == 1) {
        Pnext[pos(IMAX, JMIN, HEL(k))] = Pnext[pos(Nx, 1, k)] - dx;
        Cucurr[pos(IMAX, JMIN, HEL(k))] = Cucurr[pos(Nx, 1, k)];
      } else if (j == Ny) {
        Pnext[pos(IMAX, JMAX, HEL(k))] = Pnext[pos(Nx, Ny, k)] - dx;
        Cucurr[pos(IMAX, JMAX, HEL(k))] = Cucurr[pos(Nx, Ny, k)];
      }
      if (k == 1) {
        Pnext[pos(IMAX, j, KMIN)] = Pnext[pos(Nx, j, 1)] - dx;
        Cucurr[pos(IMAX, j, KMIN)] = Cucurr[pos(Nx, j, 1)];
      } else if (k == Nz) {
        Pnext[pos(IMAX, SYM(j), KMAX)] = Pnext[pos(Nx, j, Nz)] - dx;
        Cucurr[pos(IMAX, SYM(j), KMAX)] = Cucurr[pos(Nx, j, Nz)];
      }
    }

    // BC at (y=0.5) or (y=Ny+0.5)
    if (j == 1) {
      Pnext[pos(i, JMIN, HEL(k))] = Pnext[pos(i, 1, k)];
      Cucurr[pos(i, JMIN, HEL(k))] = Cucurr[pos(i, 1, k)];
      if (k == 1) {
        Pnext[pos(i, JMIN, HEL(KMIN))] = Pnext[pos(i, 1, 1)];
        Cucurr[pos(i, JMIN, HEL(KMIN))] = Cucurr[pos(i, 1, 1)];
      } else if (k == Nz) {
        Pnext[pos(i, SYM(JMIN), HEL(KMAX))] = Pnext[pos(i, 1, Nz)];
        Cucurr[pos(i, SYM(JMIN), HEL(KMAX))] = Cucurr[pos(i, 1, Nz)];
      }
    } else if (j == Ny) {
      Pnext[pos(i, JMAX, HEL(k))] = Pnext[pos(i, Ny, k)];
      Cucurr[pos(i, JMAX, HEL(k))] = Cucurr[pos(i, Ny, k)];
      if (k == 1) {
        Pnext[pos(i, JMAX, HEL(KMIN))] = Pnext[pos(i, Ny, 1)];
        Cucurr[pos(i, JMAX, HEL(KMIN))] = Cucurr[pos(i, Ny, 1)];
      } else if (k == Nz) {
        Pnext[pos(i, SYM(JMAX), HEL(KMAX))] = Pnext[pos(i, Ny, Nz)];
        Cucurr[pos(i, SYM(JMAX), HEL(KMAX))] = Cucurr[pos(i, Ny, Nz)];
      }
    }

    // BC at (z=0.5) or (z=Nz+0.5)
    if (k == 1) {
      Pnext[pos(i, j, KMIN)] = Pnext[pos(i, j, 1)];
      Cucurr[pos(i, j, KMIN)] = Cucurr[pos(i, j, 1)];
    } else if (k == Nz) {
      Pnext[pos(i, SYM(j), KMAX)] = Pnext[pos(i, j, Nz)];
      Cucurr[pos(i, SYM(j), KMAX)] = Cucurr[pos(i, j, Nz)];
    }
#endif
  }

#if (NOISE == CONSERVE)
#if (tmax_NOISE > 0)
  if (iter * dt * Tau0_sec < 60. * tmax_NOISE)
#endif
  {
    curandState localState;
    localState = state[pos(i, j, k)];
    Jn[pos(i, j, k)] = curand_normal(&localState);
    state[pos(i, j, k)] = localState;
  }
#endif
}

__global__ void Compute_U(REAL *Pcurr, REAL *Pnext, REAL *F, REAL *Cucurr, REAL *Cunext, REAL *Jn, Constants *Param) {
#define dpyl dpyr
#define dpzl dpzr
#define dpxu dpyr
#define dpzu dpzr
#define dpxd dpyr
#define dpzd dpzr
#define dpxt dpyr
#define dpyt dpzr
#define djxr dpxr
#define djxl dpxl
#define djyu dpyu
#define djyd dpyd
#define djzt dpzt
#define djzb dpzb

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  if (i * j * k * (i - Nx - 1) * (j - Ny - 1) * (k - Nz - 1)) //	i.e. (i!=0 && j!=0 && k!=0 && i!=Nx+1 && j!=Ny+1 && k!=Nz+1)
  {
#if (MIXd)
    REAL edgegrid = MIXd / dx_microns + Xtip; // [dx]
    int intedgegrid = (int)(edgegrid);        // []
    REAL r = (edgegrid - intedgegrid) * dx;   // change unit to [W]

    if (i > intedgegrid) {
      Cunext[pos(i, j, k)] = -1.;
    } else if (i < intedgegrid)
#endif
    {
      REAL u = Cucurr[pos(i, j, k)];
      REAL phi = F[pos(i, j, k)];
      REAL nphi = tanh(Pnext[pos(i, j, k)] / sqrt2);

      // ----
      // dUdt
      // ----
      Cunext[pos(i, j, k)] = (1. + omk * u) * (nphi - phi) + dt * D * 0.5 * ((2. - F[pos(i + 1, j, k)] - phi) * (Cucurr[pos(i + 1, j, k)] - u) - (2. - F[pos(i - 1, j, k)] - phi) * (u - Cucurr[pos(i - 1, j, k)]) + (2. - F[pos(i, j + 1, k)] - phi) * (Cucurr[pos(i, j + 1, k)] - u) - (2. - F[pos(i, j - 1, k)] - phi) * (u - Cucurr[pos(i, j - 1, k)]) + (2. - F[pos(i, j, k + 1)] - phi) * (Cucurr[pos(i, j, k + 1)] - u) - (2. - F[pos(i, j, k - 1)] - phi) * (u - Cucurr[pos(i, j, k - 1)])) / dx / dx;

#if (ANTITRAPPING)
      //		if(fabs(omp2)>1.e-10)
      if (omp2 > 0.) {
        // -----------------------------------------------------
        // Source term: div[(dphi/dt)gradphi/|gradphi|)]/sqrt(2)
        // -----------------------------------------------------
        REAL dpxr, dpyr, dnormr, dpxl, dnorml, dpyu, dnormu, dpyd, dnormd;
        REAL dpzr, dpzt, dpzb, dpxb, dpyb, dnormt, dnormb;

        // The unit gradient may be computed either by Grad(h_Psi) or Grad(h_Phi)
        dpxr = F[pos(i + 1, j, k)] - phi;
        dpyr = (F[pos(i + 1, j + 1, k)] + F[pos(i, j + 1, k)] - F[pos(i + 1, j - 1, k)] - F[pos(i, j - 1, k)]) / 4.;
        dpzr = (F[pos(i + 1, j, k + 1)] + F[pos(i, j, k + 1)] - F[pos(i + 1, j, k - 1)] - F[pos(i, j, k - 1)]) / 4.;
        dnormr = sqrt(dpxr * dpxr + dpyr * dpyr + dpzr * dpzr);

        dpxl = phi - F[pos(i - 1, j, k)];
        dpyl = (F[pos(i - 1, j + 1, k)] + F[pos(i, j + 1, k)] - F[pos(i - 1, j - 1, k)] - F[pos(i, j - 1, k)]) / 4.;
        dpzl = (F[pos(i - 1, j, k + 1)] + F[pos(i, j, k + 1)] - F[pos(i - 1, j, k - 1)] - F[pos(i, j, k - 1)]) / 4.;
        dnorml = sqrt(dpxl * dpxl + dpyl * dpyl + dpzl * dpzl);

        dpyu = F[pos(i, j + 1, k)] - phi;
        dpxu = (F[pos(i + 1, j + 1, k)] + F[pos(i + 1, j, k)] - F[pos(i - 1, j + 1, k)] - F[pos(i - 1, j, k)]) / 4.;
        dpzu = (F[pos(i, j + 1, k + 1)] + F[pos(i, j, k + 1)] - F[pos(i, j + 1, k - 1)] - F[pos(i, j, k - 1)]) / 4.;
        dnormu = sqrt(dpxu * dpxu + dpyu * dpyu + dpzu * dpzu);

        dpyd = phi - F[pos(i, j - 1, k)];
        dpxd = (F[pos(i + 1, j - 1, k)] + F[pos(i + 1, j, k)] - F[pos(i - 1, j - 1, k)] - F[pos(i - 1, j, k)]) / 4.;
        dpzd = (F[pos(i, j - 1, k + 1)] + F[pos(i, j, k + 1)] - F[pos(i, j - 1, k - 1)] - F[pos(i, j, k - 1)]) / 4.;
        dnormd = sqrt(dpxd * dpxd + dpyd * dpyd + dpzd * dpzd);

        dpzt = F[pos(i, j, k + 1)] - phi;
        dpxt = (F[pos(i + 1, j, k + 1)] + F[pos(i + 1, j, k)] - F[pos(i - 1, j, k + 1)] - F[pos(i - 1, j, k)]) / 4.;
        dpyt = (F[pos(i, j + 1, k + 1)] + F[pos(i, j + 1, k)] - F[pos(i, j - 1, k + 1)] - F[pos(i, j - 1, k)]) / 4.;
        dnormt = sqrt(dpxt * dpxt + dpyt * dpyt + dpzt * dpzt);

        dpzb = phi - F[pos(i, j, k - 1)];
        dpxb = (F[pos(i + 1, j, k - 1)] + F[pos(i + 1, j, k)] - F[pos(i - 1, j, k - 1)] - F[pos(i - 1, j, k)]) / 4.;
        dpyb = (F[pos(i, j + 1, k - 1)] + F[pos(i, j + 1, k)] - F[pos(i, j - 1, k - 1)] - F[pos(i, j - 1, k)]) / 4.;
        dnormb = sqrt(dpxb * dpxb + dpyb * dpyb + dpzb * dpzb);

        if ((dnormr * dnorml * dnormu * dnormd * dnormt * dnormb) > 0.) {
          REAL omp2dpsi = omp2 * (Pnext[pos(i, j, k)] - Pcurr[pos(i, j, k)]);

          djxr = 0.25 * ((1. - F[pos(i + 1, j, k)] * F[pos(i + 1, j, k)]) * (Pnext[pos(i + 1, j, k)] - Pcurr[pos(i + 1, j, k)]) * (1. + omk * Cucurr[pos(i + 1, j, k)]) + omp2dpsi * (1. + omk * u)) * dpxr / dnormr;
          djxl = 0.25 * ((1. - F[pos(i - 1, j, k)] * F[pos(i - 1, j, k)]) * (Pnext[pos(i - 1, j, k)] - Pcurr[pos(i - 1, j, k)]) * (1. + omk * Cucurr[pos(i - 1, j, k)]) + omp2dpsi * (1. + omk * u)) * dpxl / dnorml;
          djyu = 0.25 * ((1. - F[pos(i, j + 1, k)] * F[pos(i, j + 1, k)]) * (Pnext[pos(i, j + 1, k)] - Pcurr[pos(i, j + 1, k)]) * (1. + omk * Cucurr[pos(i, j + 1, k)]) + omp2dpsi * (1. + omk * u)) * dpyu / dnormu;
          djyd = 0.25 * ((1. - F[pos(i, j - 1, k)] * F[pos(i, j - 1, k)]) * (Pnext[pos(i, j - 1, k)] - Pcurr[pos(i, j - 1, k)]) * (1. + omk * Cucurr[pos(i, j - 1, k)]) + omp2dpsi * (1. + omk * u)) * dpyd / dnormd;
          djzt = 0.25 * ((1. - F[pos(i, j, k + 1)] * F[pos(i, j, k + 1)]) * (Pnext[pos(i, j, k + 1)] - Pcurr[pos(i, j, k + 1)]) * (1. + omk * Cucurr[pos(i, j, k + 1)]) + omp2dpsi * (1. + omk * u)) * dpzt / dnormt;
          djzb = 0.25 * ((1. - F[pos(i, j, k - 1)] * F[pos(i, j, k - 1)]) * (Pnext[pos(i, j, k - 1)] - Pcurr[pos(i, j, k - 1)]) * (1. + omk * Cucurr[pos(i, j, k - 1)]) + omp2dpsi * (1. + omk * u)) * dpzb / dnormb;

          // ----
          // dUdt
          // ----
          Cunext[pos(i, j, k)] += (djxr - djxl + djyu - djyd + djzt - djzb) / dx;
        }
      }
#endif

      // conserved noise
#if (NOISE == CONSERVE)
      REAL variance = D * Fnoise * dt / (dx_microns * dx_microns * dx_microns);
      variance = sqrt(variance);

      REAL jxnoise, jynoise, jznoise;
      REAL locvar = sqrt((1. - phi) * (1. + omk * u));

      // x
      if (i == 1) {
        jxnoise = sqrt((1. - F[pos(i + 1, j, k)]) * (1. + omk * Cucurr[pos(i + 1, j, k)])) * (Jn[pos(i + 1, j, k)] - Jn[pos(i, j, k)]);
      } else if (i == Nx) {
        jxnoise = -locvar * (Jn[pos(i, j, k)] - Jn[pos(i - 1, j, k)]);
      } else {
        jxnoise = sqrt((1. - F[pos(i + 1, j, k)]) * (1. + omk * Cucurr[pos(i + 1, j, k)])) * (Jn[pos(i + 1, j, k)] - Jn[pos(i, j, k)]) - locvar * (Jn[pos(i, j, k)] - Jn[pos(i - 1, j, k)]);
      }

      // y
      if (j == 1) {
#if (BOUND_COND_Y == NOFLUX)
        jynoise = sqrt((1. - F[pos(i, j + 1, k)]) * (1. + omk * Cucurr[pos(i, j + 1, k)])) * (Jn[pos(i, j + 1, k)] - Jn[pos(i, j, k)]);
#endif
#if (BOUND_COND_Y == PERIODIC)
        jynoise = sqrt((1. - F[pos(i, j + 1, k)]) * (1. + omk * Cucurr[pos(i, j + 1, k)])) * (Jn[pos(i, j + 1, k)] - Jn[pos(i, j, k)]) - locvar * (Jn[pos(i, j, k)] - Jn[pos(i, j - 1, k)]);
#endif
      } else if (j == Ny) {
#if (BOUND_COND_Y == NOFLUX)
        jynoise = -locvar * (Jn[pos(i, j, k)] - Jn[pos(i, j - 1, k)]);
#endif
#if (BOUND_COND_Y == PERIODIC)
        jynoise = sqrt((1. - F[pos(i, 1, k)]) * (1. + omk * Cucurr[pos(i, 1, k)])) * (Jn[pos(i, 1, k)] - Jn[pos(i, 0, k)]) - locvar * (Jn[pos(i, j, k)] - Jn[pos(i, j - 1, k)]);
#endif
      } else {
        jynoise = sqrt((1. - F[pos(i, j + 1, k)]) * (1. + omk * Cucurr[pos(i, j + 1, k)])) * (Jn[pos(i, j + 1, k)] - Jn[pos(i, j, k)]) - locvar * (Jn[pos(i, j, k)] - Jn[pos(i, j - 1, k)]);
      }

      // z
      if (k == 1) {
#if (BOUND_COND_Z == NOFLUX)
        jznoise = sqrt((1. - F[pos(i, j, k + 1)]) * (1. + omk * Cucurr[pos(i, j, k + 1)])) * (Jn[pos(i, j, k + 1)] - Jn[pos(i, j, k)]);
#endif
#if (BOUND_COND_Z == PERIODIC)
        jznoise = sqrt((1. - F[pos(i, j, k + 1)]) * (1. + omk * Cucurr[pos(i, j, k + 1)])) * (Jn[pos(i, j, k + 1)] - Jn[pos(i, j, k)]) - locvar * (Jn[pos(i, j, k)] - Jn[pos(i, j, k - 1)]);
#endif
      } else if (k == Nz) {
#if (BOUND_COND_Z == NOFLUX)
        jznoise = -locvar * (Jn[pos(i, j, k)] - Jn[pos(i, j, k - 1)]);
#endif
#if (BOUND_COND_Z == PERIODIC)
        jznoise = sqrt((1. - F[pos(i, j, 1)]) * (1. + omk * Cucurr[pos(i, j, 1)])) * (Jn[pos(i, j, 1)] - Jn[pos(i, j, 0)]) - locvar * (Jn[pos(i, j, k)] - Jn[pos(i, j, k - 1)]);
#endif
      } else {
        jznoise = sqrt((1. - F[pos(i, j, k + 1)]) * (1. + omk * Cucurr[pos(i, j, k + 1)])) * (Jn[pos(i, j, k + 1)] - Jn[pos(i, j, k)]) - locvar * (Jn[pos(i, j, k)] - Jn[pos(i, j, k - 1)]);
      }

      variance = 2. * variance * (jxnoise + jynoise + jznoise) / dx;

      Cunext[pos(i, j, k)] -= variance;
#endif

      // ----
      // dUdt
      // ----
      Cunext[pos(i, j, k)] /= (opk - omk * nphi);
      Cunext[pos(i, j, k)] += u;

#if (MIXd)
      if (i == (intedgegrid - 1)) {
        Cunext[pos(i + 1, j, k)] = Cunext[pos(i, j, k)] + (-1. - Cunext[pos(i, j, k)]) / (dx + r) * dx;
      }
#endif
    }

    if (fabs(Cunext[pos(i, j, k)]) > 1.1) {
      printf("(iter=%d)Ucurr(%d,%d,%d)=%g, next=%g,Warning!!!!!!!!\n", iter, i, j, k, Cucurr[pos(i, j, k)], Cunext[pos(i, j, k)]);
    }
  }

  if ((!i) * (!j) * (!k)) // i.e. (i==0 && j==0 && k==0)
  {
    iter++;

    REAL vptemp = Vp;

    // update pulling velocity for the next time step

#if (OSC_Velocity != WITHOUT)
    if (iter * dt * Tau0_sec > OSC_t0) {
      OSCVamp = 0.;

      // printf("iter = %i, OSC_t0 = %g, time = %g\n",iter, OSC_t0,iter*dt*Tau0_sec);

#if (OSC_Velocity == CONST_V)
      {
        OSCVamp = OSCVamp0;
      }
#endif

#if (OSC_Velocity == LINEAR)
      {

        OSCVamp = OSCVamp0 / (TOTALTIME - OSC_t0) * Tau0_sec * (iter * dt - OSC_t0 / Tau0_sec);
      }
#endif

#if (OSC_Velocity == SINOSC)
      {
        OSCVamp = OSCVamp0 * sin(2. * PI * (iter * dt * Tau0_sec - OSC_t0) / OSC_Period);
      }
#endif

#if (OSC_Velocity == STEPLIKE)
      {
        REAL OSCTime = iter * dt * Tau0_sec - OSC_t0; // time after turning on the perturbation Va

        if ((OSCTime - (OSCNstep - 1) * OSC_Period) <= 0.5 * OSC_Period) {
#if (if_Vamp_Up)
          OSCVamp = OSCVamp0 * OSC_Vamp_Up;
#else
          OSCVamp = OSCVamp0;
#endif
        } else {
          OSCVamp = -OSCVamp0;
        }

        if (OSCTime > OSCNstep * OSC_Period) {
          OSCNstep = OSCNstep + 1;
        }
      }
#endif

#if (OSC_tk != 0)
      if (iter * dt * Tau0_sec > OSC_tk) {
        OSCVamp = 0.;
      }
#endif

      vptemp += OSCVamp;
    }

#endif

    Lenpull += dt * vptemp;
  }
}

__global__ void PullBack(REAL *P, REAL *U, REAL *Pnext, REAL *Cunext, Constants *Param) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

#if (PBFRAME == LAB)
  int off = 1;
#endif
#if (PBFRAME == TIP)
  int off = (int)(Xtip - xint / dx);
#endif

  if (i < Nx + 1 - off) {
    Pnext[pos(i, j, k)] = P[pos(i + 1, j, k)];
    Cunext[pos(i, j, k)] = U[pos(i + 1, j, k)];
  } else {
    Pnext[pos(i, j, k)] = P[pos(Nx - off, j, k)] - (i - Nx + off) * dx;
    Cunext[pos(i, j, k)] = -1.;
  }
  if (i == 0 && j == 0 && k == 0) {
    xoffs += off * dx;
  }
}

__global__ void BC(REAL *Pcurr, REAL *Pnext, REAL *Cucurr, REAL *Cunext) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  // BC at (x=0.5) or (x=Nx+0.5)
  if (i == 1) {
    Pnext[pos(IMIN, j, k)] = Pnext[pos(1, j, k)];
    Cucurr[pos(IMIN, j, k)] = Cucurr[pos(1, j, k)];
    if (j == 1) {
      Pnext[pos(IMIN, JMIN, HEL(k))] = Pnext[pos(1, 1, k)];
      Cucurr[pos(IMIN, JMIN, HEL(k))] = Cucurr[pos(1, 1, k)];
    } else if (j == Ny) {
      Pnext[pos(IMIN, JMAX, HEL(k))] = Pnext[pos(1, Ny, k)];
      Cucurr[pos(IMIN, JMAX, HEL(k))] = Cucurr[pos(1, Ny, k)];
    }
    if (k == 1) {
      Pnext[pos(IMIN, j, KMIN)] = Pnext[pos(1, j, 1)];
      Cucurr[pos(IMIN, j, KMIN)] = Cucurr[pos(1, j, 1)];
    } else if (k == Nz) {
      Pnext[pos(IMIN, SYM(j), KMAX)] = Pnext[pos(1, j, Nz)];
      Cucurr[pos(IMIN, SYM(j), KMAX)] = Cucurr[pos(1, j, Nz)];
    }
  } else if (i == Nx) {
    Pnext[pos(IMAX, j, k)] = Pnext[pos(Nx, j, k)] - dx;
    Cucurr[pos(IMAX, j, k)] = Cucurr[pos(Nx, j, k)];
    if (j == 1) {
      Pnext[pos(IMAX, JMIN, HEL(k))] = Pnext[pos(Nx, 1, k)] - dx;
      Cucurr[pos(IMAX, JMIN, HEL(k))] = Cucurr[pos(Nx, 1, k)];
    } else if (j == Ny) {
      Pnext[pos(IMAX, JMAX, HEL(k))] = Pnext[pos(Nx, Ny, k)] - dx;
      Cucurr[pos(IMAX, JMAX, HEL(k))] = Cucurr[pos(Nx, Ny, k)];
    }
    if (k == 1) {
      Pnext[pos(IMAX, j, KMIN)] = Pnext[pos(Nx, j, 1)] - dx;
      Cucurr[pos(IMAX, j, KMIN)] = Cucurr[pos(Nx, j, 1)];
    } else if (k == Nz) {
      Pnext[pos(IMAX, SYM(j), KMAX)] = Pnext[pos(Nx, j, Nz)] - dx;
      Cucurr[pos(IMAX, SYM(j), KMAX)] = Cucurr[pos(Nx, j, Nz)];
    }
  }

  // BC at (y=0.5) or (y=Ny+0.5)
  if (j == 1) {
    Pnext[pos(i, JMIN, HEL(k))] = Pnext[pos(i, 1, k)];
    Cucurr[pos(i, JMIN, HEL(k))] = Cucurr[pos(i, 1, k)];
    if (k == 1) {
      Pnext[pos(i, JMIN, HEL(KMIN))] = Pnext[pos(i, 1, 1)];
      Cucurr[pos(i, JMIN, HEL(KMIN))] = Cucurr[pos(i, 1, 1)];
    } else if (k == Nz) {
      Pnext[pos(i, SYM(JMIN), HEL(KMAX))] = Pnext[pos(i, 1, Nz)];
      Cucurr[pos(i, SYM(JMIN), HEL(KMAX))] = Cucurr[pos(i, 1, Nz)];
    }
  } else if (j == Ny) {
    Pnext[pos(i, JMAX, HEL(k))] = Pnext[pos(i, Ny, k)];
    Cucurr[pos(i, JMAX, HEL(k))] = Cucurr[pos(i, Ny, k)];
    if (k == 1) {
      Pnext[pos(i, JMAX, HEL(KMIN))] = Pnext[pos(i, Ny, 1)];
      Cucurr[pos(i, JMAX, HEL(KMIN))] = Cucurr[pos(i, Ny, 1)];
    } else if (k == Nz) {
      Pnext[pos(i, SYM(JMAX), HEL(KMAX))] = Pnext[pos(i, Ny, Nz)];
      Cucurr[pos(i, SYM(JMAX), HEL(KMAX))] = Cucurr[pos(i, Ny, Nz)];
    }
  }

  // BC at (z=0.5) or (z=Nz+0.5)
  if (k == 1) {
    Pnext[pos(i, j, KMIN)] = Pnext[pos(i, j, 1)];
    Cucurr[pos(i, j, KMIN)] = Cucurr[pos(i, j, 1)];
  } else if (k == Nz) {
    Pnext[pos(i, SYM(j), KMAX)] = Pnext[pos(i, j, Nz)];
    Cucurr[pos(i, SYM(j), KMAX)] = Cucurr[pos(i, j, Nz)];
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Processing //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
__global__ void GetXsl_YZ(REAL *P, REAL *Xyz) {
#define pos2D(y, z) (WIDTH * (y) + (z))
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;

  REAL x, maxX = 0.;
  for (int i = 1; i <= Nx; i++) {
    if (P[pos(i, j, k)] * P[pos(i + 1, j, k)] < 0.) {
      x = 1. * i + P[pos(i, j, k)] / (P[pos(i, j, k)] - P[pos(i + 1, j, k)]);
      if (x > maxX) {
        maxX = x;
      }
    }
  }
  Xyz[pos2D(j, k)] = maxX;
}

__global__ void GetXtip(REAL *Xyz, Constants *Param) {
#define pos2D(y, z) (WIDTH * (y) + (z))
  // -------------------
  // Find Xtip,Ytip,Ztip
  // -------------------
  REAL x, maxX = 0., maxY = 0., maxZ = 0.;
  for (int j = 1; j <= Ny; j++) {
    for (int k = 1; k <= Nz; k++) {
      x = Xyz[pos2D(j, k)];
      if (x > maxX) {
        maxX = x;
        maxY = j;
        maxZ = k;
      }
    }
  }
  Xtip = maxX;
  Ytip = maxY;
  Ztip = maxZ;
}

__global__ void GetRtip(REAL *Xyz, REAL *P, Constants *Param) {
#define pos2D(y, z) (WIDTH * (y) + (z))
  // -----------------
  // Radii calculation
  // -----------------
  int hNpol = Npol / 2;

  // -------------------------------------------------------------
  int itip;
  REAL Xtip_y, Xtip_z;
  REAL pzr = 0., ppl = 0., pmi = 0.;
  REAL dpdx = 0., d2pdr2 = 0., curv = 0.;
#define d2pdy2 d2pdr2
#define d2pdz2 d2pdr2
  REAL ARRxyz[Npol] = {0.};
  REAL PXYZ[Npol] = {0.};
  REAL Psi_X[Npol] = {0.};
  REAL d2Psidr2_X[Npol] = {0.};
#define x_ ARRxyz
#define y_ ARRxyz
#define z_ ARRxyz
  // Along (z=Ztip), Curvature = (d2p/dy2)/(dp/dx) at tip location
#define Xsl_Y          PXYZ
#define Psi_Y          PXYZ
#define PsiYtip_X      Psi_X
#define d2Psidy2Ytip_X d2Psidr2_X
  // Along (y=Ytip), Curvature = (d2p/dz2)/(dp/dx) at tip location
#define Xsl_Z          PXYZ
#define Psi_Z          PXYZ
#define PsiZtip_X      Psi_X
#define d2Psidz2Ztip_X d2Psidr2_X
  // -------------------------------------------------------------

  // --------------
  // Along (z=Ztip)
  // --------------
  // (y-0.5) ==> y_[]
  // X(s/l)(y) ==> Xsl_Y[]
  for (int p = 0; p < Npol; p++) {
    y_[p] = (REAL)(p - hNpol + 0.5);
    Xsl_Y[p] = Xyz[pos2D((int)(fabs(y_[p]) + 0.5 + Ytip), (int)(Ztip))];
  }
  // Xsl_Y[](y=Ytip) ==> Xtip_z (i.e Xtip at z=Ztip)
  Xtip_z = PolInt(y_, Xsl_Y, 0.);
  itip = (int)(Xtip_z);

  // Psi(y=Ytip)(x) ==> PsiYtip_X[]
  // d2Psi/dy2(y=Ytip)(x) ==> d2Psidy2Ytip_X[]
  for (int i = itip - hNpol + 1; i <= itip + hNpol; i++) {
    // Psi(y)(x~i) ==> Psi_Y[]
    for (int j = 0; j < hNpol; j++) {
      Psi_Y[hNpol + j] = P[pos(i, j + 1, (int)Ztip)];
      Psi_Y[hNpol - 1 - j] = Psi_Y[hNpol + j];
    }

    pzr = PolInt(y_, Psi_Y, 0.);
    ppl = PolInt(y_, Psi_Y, 0.1);
    pmi = PolInt(y_, Psi_Y, -0.1);

    PsiYtip_X[i - itip + hNpol - 1] = pzr;
    d2Psidy2Ytip_X[i - itip + hNpol - 1] = (ppl - 2. * pzr + pmi) / (0.1 * 0.1);
  }
  // x ==> x_[]
  for (int p = 0; p < Npol; p++) {
    x_[p] = (REAL)(itip - hNpol + p);
  }
  // d(PsiYtip_X[])/dx(x=Xtip_z) ==> dpdx
  ppl = PolInt(x_, PsiYtip_X, Xtip_z + 0.1);
  pmi = PolInt(x_, PsiYtip_X, Xtip_z - 0.1);
  dpdx = (ppl - pmi) / 0.2;
  // d2Psidy2Ytip_X[](x=Xtip_z) ==> d2pdy2
  d2pdy2 = PolInt(x_, d2Psidy2Ytip_X, Xtip_z);
  // Curvature = [(d2p/dy2)/(dp/dx)](x=Xtip_z,y=Ytip)
  curv = 1. * d2pdy2 / dpdx;
  RadY = dx / curv;

  // --------------
  // Along (y=Ytip)
  // --------------
  pzr = 0.;
  ppl = 0.;
  pmi = 0.;
  dpdx = 0.;
  d2pdz2 = 0.;
  curv = 0.;
  // (z-0.5) ==> z_[]
  // X(s/l)(z) ==> Xsl_Z[]
  for (int p = 0; p < Npol; p++) {
    z_[p] = (REAL)(p - hNpol + 0.5);
    Xsl_Z[p] = Xyz[pos2D((int)(Ytip), (int)(fabs(z_[p]) + 0.5 + Ztip))];
  }
  // Xsl_Z[](z=Ztip) ==> Xtip_y
  Xtip_y = PolInt(z_, Xsl_Z, 0.);
  itip = (int)(Xtip_y);

  // Psi(z=Ztip)(x) ==> PsiZtip_X[]
  // d2Psi/dz2(z=Ztip)(x) ==> d2Psidz2Ztip_X[]
  for (int i = itip - hNpol + 1; i <= itip + hNpol; i++) {
    // Psi(z)(x~i) ==> Psi_Z[]
    for (int j = 0; j < hNpol; j++) {
      Psi_Z[hNpol + j] = P[pos(i, (int)Ytip, j + 1)];
      Psi_Z[hNpol - 1 - j] = Psi_Z[hNpol + j];
    }

    pzr = PolInt(z_, Psi_Z, 0.);
    ppl = PolInt(z_, Psi_Z, 0.1);
    pmi = PolInt(z_, Psi_Z, -0.1);

    PsiZtip_X[i - itip + hNpol - 1] = pzr;
    d2Psidz2Ztip_X[i - itip + hNpol - 1] = (ppl - 2. * pzr + pmi) / (0.1 * 0.1);
  }
  // x ==> x_[]
  for (int p = 0; p < Npol; p++) {
    x_[p] = (REAL)(itip - hNpol + p);
  }

  // d(PsiZtip_X[])/dx(x=Xtip_y) ==> dpdx
  ppl = PolInt(x_, PsiZtip_X, Xtip_y + 0.1);
  pmi = PolInt(x_, PsiZtip_X, Xtip_y - 0.1);
  dpdx = (ppl - pmi) / 0.2;
  // d2Psidz2Ztip_X[](x=Xtip_y) ==> d2pdz2
  d2pdz2 = PolInt(x_, d2Psidz2Ztip_X, Xtip_y);
  // Curvature = [(d2p/dy2)/(dp/dx)](x=Xtip_y,y=Ytip)
  curv = 1. * d2pdz2 / dpdx;
  RadZ = dx / curv;
}

__device__ REAL PolInt(REAL *XA, REAL *YA, REAL X) {
#define VERYSMALL (1.e-12)

  REAL DY = 0.;
  REAL Y = -1.;

  REAL CC[Npol], DD[Npol];

  int NS = 0;
  REAL DIF = fabs(X - XA[0]);

  for (int I = 0; I < Npol; I++) {
    REAL DIFT = fabs(X - XA[I]);
    if (DIFT < DIF) {
      NS = I;
      DIF = DIFT;
    }
    CC[I] = YA[I];
    DD[I] = YA[I];
  }

  Y = YA[NS];
  NS = NS - 1;

  for (int M = 1; M <= Npol - 1; M++) {
    for (int I = 0; I < Npol - M; I++) {
      REAL HO = XA[I] - X;
      REAL HP = XA[I + M] - X;
      REAL W = CC[I + 1] - DD[I];
      REAL DEN = HO - HP;
      if (fabs(DEN) < VERYSMALL) {
        printf("POLINT - Fatal error, DEN=0.0\n");
        return Y;
      }
      DEN = W / DEN;
      DD[I] = HP * DEN;
      CC[I] = HO * DEN;
    }
    if (2 * NS < Npol - M) {
      DY = CC[NS + 1];
    } else {
      DY = DD[NS];
      NS = NS - 1;
    }
    Y = Y + DY;
  }
  return Y;
}

__device__ REAL RootBis(REAL *xp, REAL *yp) {
#define VERYSMALL (1.e-12)
  REAL xacc = VERYSMALL;
  const int IterMax = 40;
  int it;
  REAL f, fmid, h, xmid, xval;

  int im = 1;
  int iM = Npol - 2;
  REAL x1 = xp[im] + xacc;
  REAL x2 = xp[iM] - xacc;
  f = PolInt(xp, yp, x1);
  fmid = PolInt(xp, yp, x2);

  if (f * fmid > 0.) {
    int two = 1;
    do {

      two++;
      im += two % 2;
      iM -= (two + 1) % 2;
      x1 = xp[im];
      x2 = xp[iM];
      f = PolInt(xp, yp, x1);
      fmid = PolInt(xp, yp, x2);

    } while (f * fmid > 0. && im < iM);
    if (f * fmid > 0.) {
      printf("RootBis: Root must be bracketed for bisection...\n");
      return -1.;
    }
  }

  if (f < 0.) {
    xval = x1;
    h = x2 - x1;
  } else {
    xval = x2;
    h = x1 - x2;
  }
  for (it = 1; it <= IterMax; it++) {
    h = h * .5;
    xmid = xval + h;
    fmid = PolInt(xp, yp, xmid);

    if (fmid <= 0.) {
      xval = xmid;
    }
    if (fabs(h) < xacc || fmid == 0.) {
      return xval;
    }
  }
  return xval;
}

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Output /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
void WriteFields(char Prefix[LENMAX], int index, REAL *P, REAL *U, Constants *Param, int SVG) {
  //================================================
  // Name Output Files
  //================================================
  char FileNameF[LENMAX];
  char FileNameC[LENMAX];
  char FileNameCompoX[LENMAX];
  char Indx[LENMAX];
  char PField[LENMAX] = "PF";
  char CField[LENMAX] = "C";
  if (SVG > 0) {
    sprintf(PField, "Psi");
    sprintf(CField, "Compo");
  }
  if (index == IndexFINAL) {
    sprintf(Indx, "Final");
  } else if (index >= 0) {
    sprintf(Indx, "%d", index);
  } else {
    sprintf(Indx, "Error");
  }
  printf("Writing file *%s.%s.* ... ", Prefix, Indx);
  sprintf(FileNameF, "%s_%s.%s.vtk", PField, Prefix, Indx);
  sprintf(FileNameC, "%s_%s.%s.vtk", CField, Prefix, Indx);
  sprintf(FileNameCompoX, "CompoX_%s.%s.dat", Prefix, Indx);

  //================================================
  // Output Fields
  //================================================
  REAL Delta = 1. - (Xtip * dx - x0 + xoffs - Vp * dt * iter) / lT;
  REAL psi, phi, c;
  int XmaxC = XoutMAX;
  int XmaxF = XoutMAX;
  if (SVG > 0) {
    XmaxC = Nx + 2;
    XmaxF = Nx + 2;
  }

  // .vtk header
  FILE *OutFileF;
  OutFileF = fopen(FileNameF, "w");
  fprintf(OutFileF, "# vtk DataFile Version 3.0\n");
  fprintf(OutFileF, "Delta %g\n", Delta);
  fprintf(OutFileF, "ASCII\n");
  fprintf(OutFileF, "DATASET STRUCTURED_POINTS\n");
  fprintf(OutFileF, "DIMENSIONS %d %d %d\n", (XmaxF - XoutMIN), Ny + 2, Nz + 2);
  fprintf(OutFileF, "ASPECT_RATIO %f %f %f\n", 1., 1., 1.);
  fprintf(OutFileF, "ORIGIN 0 0 0\n");
  fprintf(OutFileF, "POINT_DATA %d\n", (XmaxF - XoutMIN) * (Ny + 2) * (Nz + 2));
  fprintf(OutFileF, "SCALARS PF double 1\n");
  fprintf(OutFileF, "LOOKUP_TABLE default\n");

  // .vtk header
  FILE *OutFileC;
  OutFileC = fopen(FileNameC, "w");
  fprintf(OutFileC, "# vtk DataFile Version 3.0\n");
  fprintf(OutFileC, "Delta %g\n", Delta);
  fprintf(OutFileC, "ASCII\n");
  fprintf(OutFileC, "DATASET STRUCTURED_POINTS\n");
  fprintf(OutFileC, "DIMENSIONS %d %d %d\n", (XmaxC - XoutMIN), Ny + 2, Nz + 2);
  fprintf(OutFileC, "ASPECT_RATIO %f %f %f\n", 1., 1., 1.);
  fprintf(OutFileC, "ORIGIN 0 0 0\n");
  fprintf(OutFileC, "POINT_DATA %d\n", (XmaxC - XoutMIN) * (Ny + 2) * (Nz + 2));
  fprintf(OutFileC, "SCALARS C double 1\n");
  fprintf(OutFileC, "LOOKUP_TABLE default\n");

  //=======================
  // Write Fields to files
  //=======================
  if (SVG > 0) // SVG: Psi & Compo
  {
    for (int k = 0; k < Nz + 2; k++) {
      for (int j = 0; j < Ny + 2; j++) {
        for (int i = 0; i < Nx + 2; i++) {
          psi = P[pos(i, j, k)];
          c = 0.5 * (opk - omk * tanh(psi / sqrt2)) * (1. + omk * U[pos(i, j, k)]);
          fprintf(OutFileF, DECIMALS_SVG, psi);
          fprintf(OutFileC, DECIMALS_SVG, c);
        }
      }
    }
  } else // Movies: PF & C
  {
    for (int k = 0; k < Nz + 2; k++) {
      for (int j = 0; j < Ny + 2; j++) {
        for (int i = XoutMIN; i < XmaxF; i++) {
          phi = tanh(P[pos(i, j, k)] / sqrt2);
          fprintf(OutFileF, DECIMALS_P, phi);
        }
      }
    }
    for (int k = 0; k < Nz + 2; k++) {
      for (int j = 0; j < Ny + 2; j++) {
        for (int i = XoutMIN; i < XmaxC; i++) {
          c = 0.5 * (opk - omk * tanh(P[pos(i, j, k)] / sqrt2)) * (1. + omk * U[pos(i, j, k)]);
          fprintf(OutFileC, DECIMALS_C, c);
        }
      }
    }
  }
  fclose(OutFileF);
  fclose(OutFileC);

  //================================================
  // Output Composition profile at the tip location
  //================================================
  int jtip = (int)(Ytip);
  int ktip = (int)(Ztip);
  FILE *OutFileCompoX = fopen(FileNameCompoX, "w");
  fprintf(OutFileCompoX, "#t=%g \n", iter * dt * Tau0_sec);
  fprintf(OutFileCompoX, "#itip=%g \t", Xtip);
  fprintf(OutFileCompoX, "jtip=%d \t", jtip);
  fprintf(OutFileCompoX, "ktip=%d \n", ktip);
  fprintf(OutFileCompoX, "x \t");
  fprintf(OutFileCompoX, "c/cl0 \n");
  for (int i = 0; i < Nx + 2; i++) {
    phi = tanh(P[pos(i, jtip, ktip)] / sqrt2);
    c = 0.5 * (opk - omk * phi) * (1. + omk * U[pos(i, jtip, ktip)]);
    // c=U[pos(i,jtip,ktip)];
    fprintf(OutFileCompoX, "%g \t", i * dx_microns);
    fprintf(OutFileCompoX, "%g \n", c);
  }
  fclose(OutFileCompoX);

  //================================================
  printf("written");
  if (SVG > 0) {
    printf(" (SVG)");
  }
  printf(".\n");
}

#if (AMPSEARCH > 0)
void WriteAmplitude(char Prefix[LENMAX], int searchloc, REAL timing, REAL *P, int outnum) {

  char FileNameAmp[LENMAX];
  sprintf(FileNameAmp, "Amp_%s.%d.dat", Prefix, searchloc);
  FILE *OutFileAmp = fopen(FileNameAmp, "a");

  REAL IntLoc[outnum];
  int point = 0;

  // initialization
  for (int i = 0; i < outnum; i++) {
    IntLoc[i] = 0.;
  }

  fprintf(OutFileAmp, "%g \t", timing); // #1 time

  for (int j = 1; j < Ny; j++) {

    if (P[pos(searchloc, j, 1)] * P[pos(searchloc, j + 1, 1)] < 0 && point < outnum) {
      IntLoc[point] = j + P[pos(searchloc, j, 1)] / (P[pos(searchloc, j, 1)] - P[pos(searchloc, j + 1, 1)]);
      point++;
    }
  }

  for (int j = 0; j < outnum; j++) {
    fprintf(OutFileAmp, "%g \t", IntLoc[j]); // on Y direction
    IntLoc[j] = 0.;
  }

  point = 0;

  for (int k = 1; k < Nz; k++) {

    if (P[pos(searchloc, 1, k)] * P[pos(searchloc, 1, k + 1)] < 0 && point < outnum) {
      IntLoc[point] = k + P[pos(searchloc, 1, k)] / (P[pos(searchloc, 1, k)] - P[pos(searchloc, 1, k + 1)]);
      point++;
    }
  }

  for (int k = 0; k < outnum; k++) {
    fprintf(OutFileAmp, "%g \t", IntLoc[k]); // on Z direction
    IntLoc[k] = 0.;
  }
  fprintf(OutFileAmp, "\n");

  fclose(OutFileAmp);
}
#endif

/////////////////////////////////////////////////////////////////////////////////
///////////////////////////  Cuda Device Management  ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
void DisplayDeviceProperties(int Ndev) {
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
    printf("\nCUDA ver                         \t %d.%d", deviceProp.major, deviceProp.minor);
    printf("\nClock rate                       \t %d KHz", deviceProp.clockRate);
    printf("\nTexture Alignment                \t %ld bytes", (long int)(deviceProp.textureAlignment));
    printf("\nDevice Overlap                   \t %s", deviceProp.deviceOverlap ? "Allowed" : "Not Allowed");
    printf("\nNumber of Multi processors       \t %d", deviceProp.multiProcessorCount);
    printf("\n==============================================================\n");
  } else {
    printf("\nCould not get properties for device %d.....\n", Ndev);
  }
}
void GetMemUsage(int *Array, int Num) {
  char buffer[LENMAX];
  std::string StrUse = "";
  FILE *pipe = popen("nvidia-smi -q --display=MEMORY | grep Used ", "r");
  while (!feof(pipe)) {
    if (fgets(buffer, LENMAX, pipe) != NULL) {
      StrUse += buffer;
    }
  }
  pclose(pipe);
  for (int dev = 0; dev < Num; dev++) {
    std::istringstream iss(StrUse.substr(StrUse.find(":") + 1, StrUse.find("MB") - StrUse.find(":") - 1));
    iss >> Array[dev];
    StrUse = StrUse.substr(StrUse.find("\n") + 1, StrUse.length() - StrUse.find("\n") - 1);
  }
}
int GetFreeDevice(int Num) {
  int FreeDev = -1;
  int MemFree = 15;
  int *Memory_Use = new int[Num];

  // Check utilization of Devices
  GetMemUsage(Memory_Use, Num);
  // See if one is free
  int dev = 0;
  do {
    if (Memory_Use[dev] < MemFree) {
      // Found one...
      FreeDev = dev;
      // Check if it is really free...
      system("sleep 1s");
      GetMemUsage(Memory_Use, Num);
      if (Memory_Use[dev] > MemFree) {
        FreeDev = -1;
      }
      // twice...
      system("sleep 1s");
      GetMemUsage(Memory_Use, Num);
      if (Memory_Use[dev] > MemFree) {
        FreeDev = -1;
      }
    }
    dev++;
  } while (FreeDev == -1 && dev < Num);

  delete[] Memory_Use;

  if (FreeDev == -1) {
    printf("=======================================\n");
    system("nvidia-smi -q --display=MEMORY |grep U");
    printf("=======================================\n");
    printf("NO AVAILABLE GPU: SIMULATION ABORTED...\n");
    printf("=======================================\n\n");
  }
  return FreeDev;
}
void AutoBlockSize(int *Bloc) {
  for (int bx = 1; bx <= BSIZEMAX; bx++) {
    if (((Nx + 2.) / bx) == ((Nx + 2) / bx)) {
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
}
