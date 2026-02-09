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



#ifndef MACRO_H_
#define MACRO_H_

#define NGPU 4
#define NewP 0 // Set 1 to use two shells form of Compute_P <100>+<110>; set 2 to use three shells
#define NewU 0 // Set 1 to use two shells form of Compute_U <100>+<110>; set 2 to use three shells; set 3 to use 1 lattice shell <100>, with the approximated anti-trapping, average flux
// -----------------------
//		SIMULATION ID
// -----------------------
#define PREFIX "DSIR_v0.75" // Prefix of Output File Name

// -----------------------
//		EXPERIMENT
// -----------------------
// Process
#define VELOCITY  (6.)        // micron/s
#define GRAD0     (12.5 * 100) // K/m, the starting gradient
#define GRAD1     (12.5 * 100) // K/m, the ending gradient
#define TIME0     0           // s, the starting time to vary G
#define TIME1     0           // s, the time when G=GRAD1;
#define LateralG  (0)         // Set integer |LateralG|>0 for a lateral temperature gradient
#define Lateral3D 0           // Set Lateral3D = 1 to use a lateral temperature gradient in y and z directions
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
#define Nx        718    // Dimension X
#define Ny        256     // Dimension Y
#define Nz        256    // Dimension Z

// Boundary conditions
#define BOUND_COND_Y NOFLUX
#define BOUND_COND_Z NOFLUX

// effect of wall
#define WALLEFFECT WITHOUT
#define WALLSLOPE  1.
// wall slope

// Grain angles
// Grain 1 (pha = -1)
#define AngleA1 (0.) // angle 1 (alpha) [degree]
#define AngleB1 (0.) // angel 2 (beta) [degree]
#define AngleC1 (0.) // angel 3 (gamma) [degree]

#define GByloc   (0) // Grain boundary location y = GByloc[dx]. Set GByloc=0 if single grain (still do the grain index calculation), the default is grain 2 (grain index = 1).
#define InnerBCL (0) // Impose inner boundary condition until xoff < (1+InnerBCL/10)*POSITION_0 (unit: [W]). Set InnerBCL=0 if no inner BC.

// Grain 2 (pha = 1)
#define AngleA2 (0.) // angle 1 (alpha) [degree]
#define AngleB2 (0.) // angel 2 (beta) [degree]
#define AngleC2 (0.) // angel 3 (gamma) [degree]
//

// Discretization
#define E   154.1666667 // E=W/d0
#define dx  1.2  // Grid size [W]
#define dt0 1.   // Time step (input, may be reajusted)
#define Kdt .9   // Time step adjustment criterion (dt<=Kdt*dx*dx*dx/6/D)

// Oscillation
#define OSC_Velocity WITHOUT // It can be WITHOUT, SINOSC, CONST_V, LINEAR or STEPLIKE
#define OSC_Vamp     (2.)    // Same as OSC_Amp [micron/s]
#define if_Vamp_Up   0       // set to 1 if the upper osc amplitude is different from the lower (only for steplike)
#define OSC_Vamp_Up  (2.)    // the upper osc amplitude [micron/s}
#define OSC_t0       (100.)  // Same as OSC_Onset [s]
#define OSC_tk       (0)     // Time to kill OSCVamp [s]; Set 0 if no OSC_tk
#define OSC_Period   (24.)   // The Vp oscillation period [s]

// Noise
#define RSEED      0 // if 0, seed=time(NULL); RSEED otherwise
#define NOISE      WITHOUT
#define Fnoise     (0.01)
#define tmax_NOISE 0 // (int) in minutes

// New impose: Thermal drift
#define Thermaltau 0    // [s], + switch: if >0, switch on, if =0 switch off
#define Thermaldzt (0.) // [microns]

// Sample parameters
#define Thot    (340.8) // Hot temperature [K]
#define Tcold   (304.6) // Cold temperature [K]
#define Lsample (3.0e4) // Sample length [microns]

// Computational
#define REAL         float // use double for TFC
#define PBFRAME      TIP
#define ANTITRAPPING 1

// -----------------------
//	 INITIAL CONDITIONS
// -----------------------
#define UNDERCOOL_0 (0.)
#define POSITION_0  (800.) // [microns]
#define MIXd        0      // [microns]

#define IQY -1 // Wavenumber of the initial perturbation /Y
#define IQZ -1 // Wavenumber of the initial perturbation /Z
#define AMP 1. // [dx]

// Input File
#define INITfromFILE  0                                  // Init fromVTK or fromDAT
#define INIT_FILE     "Ny152_D270_G12_k010_V60_dx8_W100" // Defaul  file name Psi.Init.dat
#define INIT_FILE_DIR "../"                              // the directory of init files, only for fromVTK
#define FROM_X        0
#define MIRROR_Y      0
#define MIRROR_Z      0
#define MULTIPLY_Y    1
#define MULTIPLY_Z    1
#define CUTONEFOURTH  0

// -----------------------
//		  OUTPUT
// -----------------------
#define NOUTFIELDS  (-2) // Output movie files (compressed). If <0, output every POSITION_0/(-1.*NOUTFIELDS))
#define NOUTTIP     7200
#define NOUTSVG     (1) // Output savedata (uncompressed).
#define COMPRESS    1   // Only for the final data
#define OUTINDEXSEC 1   // The index in the unit of s
#define SUMMARIZE   0

// ***
// #define XoutMAX			0       // (x_tip + XoutMAX) for (PF and C) fields, if 0 output full fields (Nx+2) [for 3D]
#define XoutMIN 0
#define XoutMAX Nx
#define EVERY   2 //

// for 2D-record
#define CutZplane (0)      // number of output for 2D plane at Zloc (if CutZplane>0, recording) (if CutZplane<0, every POSITION_0/(-1*CutZplane))
#define Zloc      (Nz / 2) // for 2D plane at z=Zloc plane
#define Xout2DMAX 0        // (x_tip + XoutMAX) for fields, if 0 output full fields (Nx+2) [for 2D]

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
#define FLAT     1
#define GAUSSIAN 2
#define CONSERVE 3
#define LAB      0
#define TIP      1
#define WSLOPE   1
#define NzSLOPE  2
#define fromVTK  1
#define fromDAT  2

#define SINOSC   1 // Flags for Vp oscillations
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
////////////////////////////////

#endif
