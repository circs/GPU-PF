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



#include "gpu-kernels.h"

// C++ & CUDA headers
#include "math.h"
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
// Project headers
#include "gpu-manager.h"
#include "macro.h"
#include "param-manager.h"

__global__ void setup_kernel(unsigned long long seed, curandState *States, int AcmNx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  curand_init(seed, pos(AcmNx + i, j, k), 0, &States[pos(i, j, k)]);
}

///////////////
// Constants //
///////////////
#define sqrt2  (*Ct).sqrt2
#define PI     (*Ct).PI
#define dt     (*Ct).dt
#define kcoeff (*Ct).kcoeff
#define omk    (*Ct).omk
#define opk    (*Ct).opk
#define Eps4   (*Ct).Eps4
#define D      (*Ct).D
#define Lambda (*Ct).Lambda

/*
#define Alpha			(*Ct).Alpha
#define sAlpha			(*Ct).sAlpha
#define cAlpha			(*Ct).cAlpha
#define sAlpha2			(*Ct).sAlpha2
#define cAlpha2			(*Ct).cAlpha2
#define s2Alpha			(*Ct).s2Alpha
#define c2Alpha			(*Ct).c2Alpha
#define cAlphasAlpha	(*Ct).cAlphasAlpha

#define Beta			(*Ct).Beta
#define sBeta			(*Ct).sBeta
#define cBeta			(*Ct).cBeta
#define sBeta2          (*Ct).sBeta2
#define cBeta2			(*Ct).cBeta2
#define s2Beta			(*Ct).s2Beta
#define c2Beta			(*Ct).c2Beta
#define cBetasBeta      (*Ct).cBetasBeta
*/

#define r11 (*Ct).r11
#define r12 (*Ct).r12
#define r13 (*Ct).r13

#define r21 (*Ct).r21
#define r22 (*Ct).r22
#define r23 (*Ct).r23

#define r31 (*Ct).r31
#define r32 (*Ct).r32
#define r33 (*Ct).r33

#define Vp (*Ct).Vp
// #define lT				(*Ct).lT
#define lT0 (*Ct).lT0
#define lT1 (*Ct).lT1

#define W_microns  (*Ct).W_microns
#define dx_microns (*Ct).dx_microns
#define Tau0_sec   (*Ct).Tau0_sec

// local heat
#define Hamp (*Ct).Hamp
#define slht (*Ct).slht
#define flht (*Ct).flht

// for thermal drift
#define Tdtau (*Ct).Tdtau
#define Tddzt (*Ct).Tddzt

#if (OSC_Velocity)
#define OSCVamp0 (*Ct).OSCVamp0
#endif

///////////////
// Variables //
///////////////
#define iter  (*Vb).iter
#define niter (*Vb).niter

#define lT (*Vb).lT

#define xint  (*Vb).xint
#define xoffs (*Vb).xoffs
#define x0    (*Vb).x0

#define xtip  (*Vb).xtip
#define ytip  (*Vb).ytip
#define ztip  (*Vb).ztip
#define xtip1 (*Vb).xtip1
#define ytip1 (*Vb).ytip1
#define ztip1 (*Vb).ztip1
#define xtip2 (*Vb).xtip2
#define ytip2 (*Vb).ytip2
#define ztip2 (*Vb).ztip2

#define RadY (*Vb).RadY
#define RadZ (*Vb).RadZ

#define Vel   (*Vb).Vel
#define Delta (*Vb).Delta
#define Omega (*Vb).Omega

// for grain boundaries
#define jGBy (*Vb).jGBy

// for oscillating Vp
#define Lenpull (*Vb).Lenpull
#if (OSC_Velocity)
#define OSCVamp  (*Vb).OSCVamp
#define OSCNstep (*Vb).OSCNstep // for step-like forcing
#endif

///////////////
//   Macros  //
///////////////

#define IMIN     0
#define IMAX     (Nx + 1)
#define DEV_IMAX (DevNx + 1)

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

__global__ void Init(REAL *P1, REAL *P2,
                     REAL *U1, REAL *U2,
                     signed char *Phase,
                     curandState *States,
                     Constants *Ct,
                     Variables *Vb,
                     int AcmNx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  //----------------------------
  // Initial interface position
  //----------------------------
  REAL LateralValue = 0.;
  REAL LateralCorrect = (LateralG > 0) ? 0.25 : 0.; // for a concave interface, make xtip at POSITION_0
#if (Lateral3D == 1)                                // lateral gradient in y and z directions: default is a convex shape
  LateralValue = LateralG * (cos(sqrt(pow(j - (Ny + 1.) / 2., 2) + pow(k - (Nz + 1.) / 2., 2)) / sqrt(pow(Ny + 1., 2) + pow(Nz + 1., 2)) * PI) - 1.);
#else // lateral gradient only in the y direction
                                                    // LateralValue=LateralG*(sin(1.*j/(Ny+2.)*PI)-1.);
  LateralValue = -LateralG * ((1. * j / (Ny + 2.) - 0.5) * (1. * j / (Ny + 2.) - 0.5) - LateralCorrect);
#endif

  REAL xintyz = xint / dx - LateralValue;
#if (IQY != 0 && IQZ != 0)
#if (IQY < 0 || IQZ < 0)
  //----------------------------------------------
  // Random initial perturbation of amplitude AMP
  //----------------------------------------------
  curandState localState;
  localState = States[pos(i, j, k)];
  REAL ran = curand_uniform_double(&localState);
  States[pos(i, j, k)] = localState; // The localState changes each time after using curand
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

  P1[pos(i, j, k)] = P2[pos(i, j, k)] = -(AcmNx + i - xintyz) * dx;

  //-----------------------
  // Initial concentration
  //-----------------------
  REAL phi = tanh(P1[pos(i, j, k)] / sqrt2);
  REAL c = ((AcmNx + i) < xintyz) ? (0.5 * (opk - omk * phi)) * (1. - (1. - UNDERCOOL_0) * omk) : (0.5 * (opk - omk * phi)) * (kcoeff + omk * (1. - (1. - UNDERCOOL_0)) * exp(-(AcmNx + i - xintyz) * dx * Vp / D));

  U1[pos(i, j, k)] = U2[pos(i, j, k)] = (2. * c - opk + omk * phi) / omk / (opk - omk * phi);

  Phase[pos(i, j, k)] = 0;

  if (P1[pos(i, j, k)] > -50.) {
    // Phase[pos(i,j,k)] = 1; // if solid, pha=1. need to change for 2 grains

    if (j < jGBy) {
      Phase[pos(i, j, k)] = -1; // for grain 1
    } else {
      Phase[pos(i, j, k)] = 1; // for grain 2
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Computation /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
#define omp2 (1. - phi * phi)

__global__ void Compute_P(REAL *Pcurr, REAL *Pnext,
                          REAL *Ucurr,
                          REAL *F,
                          signed char *Phase,
                          curandState *States,
                          Constants *Ct,
                          Variables *Vb,
                          int DevNx, int AcmNx,
                          REAL *Tsbox) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  REAL psi = Pcurr[pos(i, j, k)];
  REAL phi = F[pos(i, j, k)] = tanh(psi / sqrt2);
  signed char pha = Phase[pos(i, j, k)];

  if (i * j * k * (i - DevNx - 1) * (j - Ny - 1) * (k - Nz - 1)) //  i.e. (i!=0 && j!=0 && k!=0 && i!=Nx+1 && j!=Ny+1 && k!=Nz+1)
  {
    // position variables
    int im1 = i - 1;
    int ip1 = i + 1;

    int jm1 = j - 1;
    int jp1 = j + 1;

    int km1 = k - 1;
    int kp1 = k + 1;

#if (WALLEFFECT == WSLOPE) //***What's the purpose of it?
    if ((k - 1) * (k - Nz) == 0) {

      Pnext[pos(i, j, k)] = Pnext[pos(i, j, k)];

    } else
#elif (WALLEFFECT == NzSLOPE)
    if ((k - Nz) == 0) {

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

      // first derivatives (each coordinate)
      REAL phx = ((Pcurr[pos(ip1, j, k)] - Pcurr[pos(im1, j, k)]) / dx2);
      REAL phy = ((Pcurr[pos(i, jp1, k)] - Pcurr[pos(i, jm1, k)]) / dx2);
      REAL phz = ((Pcurr[pos(i, j, kp1)] - Pcurr[pos(i, j, km1)]) / dx2);

      // second deriviatives - sqrt2 * phi * (first derivatives)^2
      REAL phxx = (Pcurr[pos(ip1, j, k)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(im1, j, k)]) / dxs - sqrt2 * phi * phx * phx;
      REAL phyy = (Pcurr[pos(i, jp1, k)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(i, jm1, k)]) / dxs - sqrt2 * phi * phy * phy;
      REAL phzz = (Pcurr[pos(i, j, kp1)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(i, j, km1)]) / dxs - sqrt2 * phi * phz * phz;

      REAL sumnn = phxx + phyy + phzz;

      if (fabs(omp2) > 0.) {
        // phase evaluation in the liquid
        // phase: 1 or -1 for solid, 0 for liquid
        // thershold omp2 >= 0.01
        // if (!pha)
        if (!pha && fabs(omp2) >= 0.001) {
          // sum all neighbors
          int sum = Phase[pos(ip1, j, k)] + Phase[pos(im1, j, k)] + Phase[pos(i, jp1, k)] + Phase[pos(i, jm1, k)] + Phase[pos(i, j, kp1)] + Phase[pos(i, j, km1)] + Phase[pos(ip1, jp1, k)] + Phase[pos(ip1, jm1, k)] + Phase[pos(im1, jp1, k)] + Phase[pos(im1, jm1, k)] + Phase[pos(i, jp1, kp1)] + Phase[pos(i, jp1, km1)] + Phase[pos(i, jm1, kp1)] + +Phase[pos(i, jm1, km1)] + Phase[pos(ip1, j, kp1)] + Phase[pos(im1, j, kp1)] + Phase[pos(ip1, j, km1)] + Phase[pos(im1, j, km1)] + Phase[pos(ip1, jp1, kp1)] + Phase[pos(ip1, jp1, km1)] + Phase[pos(ip1, jm1, kp1)] + Phase[pos(ip1, jm1, km1)] + Phase[pos(im1, jp1, kp1)] + Phase[pos(im1, jp1, km1)] + Phase[pos(im1, jm1, kp1)] + Phase[pos(im1, jm1, km1)];

          if (sum) {
            Phase[pos(i, j, k)] = pha = (sum > 0) - (sum < 0);
          }
        }

#if (GByloc == 0)
        pha = 1; // the default is grain 2 when we use a single grain (GByloc==0)
#endif

        // if it has Phase value (not in the liquid),
        if (pha) {
          // printf("Solid, %d\n",pha);

          pha = (1 + pha) / 2;

          // rotation matrix
          REAL ar11 = r11[pha];
          REAL ar12 = r12[pha];
          REAL ar13 = r13[pha];

          REAL ar21 = r21[pha];
          REAL ar22 = r22[pha];
          REAL ar23 = r23[pha];

          REAL ar31 = r31[pha];
          REAL ar32 = r32[pha];
          REAL ar33 = r33[pha];

          // mixed derivatives - sqrt2 * phi * (mulitiplication of the first derivatives )
          REAL phxy = (Pcurr[pos(ip1, jp1, k)] - Pcurr[pos(im1, jp1, k)] - Pcurr[pos(ip1, jm1, k)] + Pcurr[pos(im1, jm1, k)]) / dxs4 - sqrt2 * phi * phx * phy;
          REAL phyz = (Pcurr[pos(i, jp1, kp1)] - Pcurr[pos(i, jm1, kp1)] - Pcurr[pos(i, jp1, km1)] + Pcurr[pos(i, jm1, km1)]) / dxs4 - sqrt2 * phi * phy * phz;
          REAL phxz = (Pcurr[pos(ip1, j, kp1)] - Pcurr[pos(im1, j, kp1)] - Pcurr[pos(ip1, j, km1)] + Pcurr[pos(im1, j, km1)]) / dxs4 - sqrt2 * phi * phz * phx;

          // Rotate
          // Rotation of the first derivatives

          REAL dphix = phx * ar11 + phy * ar12 + phz * ar13;
          REAL dphiy = phx * ar21 + phy * ar22 + phz * ar23;
          REAL dphiz = phx * ar31 + phy * ar32 + phz * ar33;

          REAL pxy = dphix * dphiy;
          REAL pxz = dphix * dphiz;
          REAL pyz = dphiy * dphiz;

          REAL px2 = dphix * dphix;
          REAL py2 = dphiy * dphiy;
          REAL pz2 = dphiz * dphiz;
          REAL xnorm = px2 + py2 + pz2;

          REAL dphixx = ar11 * ar11 * phxx + ar12 * ar12 * phyy + ar13 * ar13 * phzz + 2. * (ar11 * ar12 * phxy + ar12 * ar13 * phyz + ar11 * ar13 * phxz);
          REAL dphiyy = ar21 * ar21 * phxx + ar22 * ar22 * phyy + ar23 * ar23 * phzz + 2. * (ar21 * ar22 * phxy + ar22 * ar23 * phyz + ar21 * ar23 * phxz);
          REAL dphizz = ar31 * ar31 * phxx + ar32 * ar32 * phyy + ar33 * ar33 * phzz + 2. * (ar31 * ar32 * phxy + ar32 * ar33 * phyz + ar31 * ar33 * phxz);

          sumnn = dphixx + dphiyy + dphizz;

          REAL xnorm2 = xnorm * xnorm;
          REAL xnorm3 = xnorm2 * xnorm;

          REAL dphixy = ar11 * (ar21 * phxx + ar22 * phxy + ar23 * phxz) + ar12 * (ar21 * phxy + ar22 * phyy + ar23 * phyz) + ar13 * (ar21 * phxz + ar22 * phyz + ar23 * phzz);

          REAL dphiyz = ar21 * (ar31 * phxx + ar32 * phxy + ar33 * phxz) + ar22 * (ar31 * phxy + ar32 * phyy + ar33 * phyz) + ar23 * (ar31 * phxz + ar32 * phyz + ar33 * phzz);

          REAL dphixz = ar31 * (ar11 * phxx + ar12 * phxy + ar13 * phxz) + ar32 * (ar11 * phxy + ar12 * phyy + ar13 * phyz) + ar33 * (ar11 * phxz + ar12 * phyz + ar13 * phzz);

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

        } else {
          // in the liquid
          anis = 0;
          xi = 1;
        }
      }

      // ----
      // dpdt
      // ----

#if (TIME0 > 0)
      lT = (iter * dt * Tau0_sec < TIME0) ? lT0 : (iter * dt * Tau0_sec > TIME1) ? lT1
                                                                                 : lT0 + (lT1 - lT0) / (TIME1 - TIME0) * (iter * dt * Tau0_sec - TIME0);
#endif

      // REAL temp=(i*dx+xoffs-x0-Vp*dt*iter)/lT;

      ////////
#if (Thermaltau == 0)
      // no thermal effect
#if (OSC_Velocity == WITHOUT)
      REAL LateralValue = 0.;
#if (LateralG != 0)
#if (Lateral3D == 1) // lateral gradient in y and z directions: default is a convex shape
      LateralValue = dx * LateralG * cos(sqrt(pow(j - (Ny + 1.) / 2., 2) + pow(k - (Nz + 1.) / 2., 2)) / sqrt(pow(Ny + 1., 2) + pow(Nz + 1., 2)) * PI);
#else // lateral gradient only in the y direction
      // LateralValue=dx*LateralG*sin(1.*j/(Ny+2.)*PI);
      LateralValue = dx * LateralG * (0. - (1. * j / (Ny + 2.) - 0.5) * (1. * j / (Ny + 2.) - 0.5));
#endif
#endif
      REAL temp = ((AcmNx + i) * dx + LateralValue + xoffs - x0 - Vp * dt * iter) / lT;
#else
      REAL temp = ((AcmNx + i) * dx + xoffs - x0 - Lenpull) / lT;
#endif 
#else
      // thermal drift effect: temp = temp + Tddzt * (1- exp( -1.*iter*dt/Tdtau ) )
      REAL temp = ((AcmNx + i) * dx + xoffs - x0 - Vp * dt * iter + Tddzt * (1. - exp(-1. * iter * dt / Tdtau))) / lT;
#endif

      ///////

      REAL Tau = (temp > 1.) ? kcoeff : (1. - omk * temp);

#if (NewP == 0) // Use 1 lattice shell <100>: variant
      Pnext[pos(i, j, k)] = psi + dt * (xi * xi * prefacs * sumnn + sqrt2 * phi - sqrt2 * omp2 * Lambda * (Ucurr[pos(i, j, k)] + temp) + anis * prefacs) / (xi * xi * prefacs) / (Tau);
#elif (NewP == 1) // Use 2 lattice shells <100>+<110>: invariant
      REAL Lap100110 = 2. * (Pcurr[pos(ip1, j, k)] + Pcurr[pos(i, jp1, k)] + Pcurr[pos(i, j, kp1)] + Pcurr[pos(im1, j, k)] + Pcurr[pos(i, jm1, k)] + Pcurr[pos(i, j, km1)]); // <100> neighbors
      Lap100110 = Lap100110 + Pcurr[pos(ip1, jp1, k)] + Pcurr[pos(ip1, j, kp1)] + Pcurr[pos(i, jp1, kp1)] + Pcurr[pos(im1, jp1, k)] + Pcurr[pos(im1, j, kp1)] + Pcurr[pos(i, jm1, kp1)] +
                  Pcurr[pos(ip1, jm1, k)] + Pcurr[pos(ip1, j, km1)] + Pcurr[pos(i, jp1, km1)] + Pcurr[pos(im1, jm1, k)] + Pcurr[pos(im1, j, km1)] + Pcurr[pos(i, jm1, km1)] - 24. * Pcurr[pos(i, j, k)]; // <110> neighbors
      Lap100110 = Lap100110 / 6. / dxs;
      // the invariant gradient of psi (<100> and <110>)
      REAL Grad100 = phx * phx + phy * phy + phz * phz;
      REAL Grad110 = (Pcurr[pos(ip1, jp1, k)] - Pcurr[pos(im1, jm1, k)]) * (Pcurr[pos(ip1, jp1, k)] - Pcurr[pos(im1, jm1, k)]) + (Pcurr[pos(ip1, jm1, k)] - Pcurr[pos(im1, jp1, k)]) * (Pcurr[pos(ip1, jm1, k)] - Pcurr[pos(im1, jp1, k)]);      // phxy110a and phxy110b
      Grad110 = Grad110 + (Pcurr[pos(ip1, j, kp1)] - Pcurr[pos(im1, j, km1)]) * (Pcurr[pos(ip1, j, kp1)] - Pcurr[pos(im1, j, km1)]) + (Pcurr[pos(im1, j, kp1)] - Pcurr[pos(ip1, j, km1)]) * (Pcurr[pos(im1, j, kp1)] - Pcurr[pos(ip1, j, km1)]); // phxz110a and phxz110b
      Grad110 = Grad110 + (Pcurr[pos(i, jp1, kp1)] - Pcurr[pos(i, jm1, km1)]) * (Pcurr[pos(i, jp1, kp1)] - Pcurr[pos(i, jm1, km1)]) + (Pcurr[pos(i, jm1, kp1)] - Pcurr[pos(i, jp1, km1)]) * (Pcurr[pos(i, jm1, kp1)] - Pcurr[pos(i, jp1, km1)]); // phyz110a and phyz110b
      Grad110 = Grad110 / 16. / dxs;

      sumnn = Lap100110 - phi * sqrt2 * (Grad100 / 3. + 2. * Grad110 / 3.);

      Pnext[pos(i, j, k)] = psi + dt * (sqrt2 * phi - sqrt2 * omp2 * Lambda * (Ucurr[pos(i, j, k)] + temp) + anis * prefacs) / (xi * xi * prefacs) / (Tau) + dt * sumnn / Tau;
#elif (NewP == 2) // Use 3 lattice shells: invariant
      REAL Lap3 = 16. * (Pcurr[pos(ip1, j, k)] + Pcurr[pos(i, jp1, k)] + Pcurr[pos(i, j, kp1)] + Pcurr[pos(im1, j, k)] + Pcurr[pos(i, jm1, k)] + Pcurr[pos(i, j, km1)]); // <100> neighbors
      Lap3 = Lap3 + 4. * (Pcurr[pos(ip1, jp1, k)] + Pcurr[pos(ip1, j, kp1)] + Pcurr[pos(i, jp1, kp1)] + Pcurr[pos(im1, jp1, k)] + Pcurr[pos(im1, j, kp1)] + Pcurr[pos(i, jm1, kp1)] +
                          Pcurr[pos(ip1, jm1, k)] + Pcurr[pos(ip1, j, km1)] + Pcurr[pos(i, jp1, km1)] + Pcurr[pos(im1, jm1, k)] + Pcurr[pos(im1, j, km1)] + Pcurr[pos(i, jm1, km1)]); // <110> neighbors
      Lap3 = Lap3 + Pcurr[pos(ip1, jp1, kp1)] + Pcurr[pos(im1, jp1, kp1)] + Pcurr[pos(ip1, jm1, kp1)] + Pcurr[pos(ip1, jp1, km1)]                                                     // <111> neighbors
             + Pcurr[pos(ip1, jm1, km1)] + Pcurr[pos(im1, jp1, km1)] + Pcurr[pos(im1, jm1, kp1)] + Pcurr[pos(im1, jm1, km1)] - 152. * Pcurr[pos(i, j, k)];
      Lap3 = Lap3 / 36. / dxs; // Laplacian: 4/9 <100> + 4/9 <110> + 1/9 <111>
      // the invariant gradient of psi (<100>, <110> and <111>)
      REAL Grad100 = phx * phx + phy * phy + phz * phz;
      REAL Grad110 = (Pcurr[pos(ip1, jp1, k)] - Pcurr[pos(im1, jm1, k)]) * (Pcurr[pos(ip1, jp1, k)] - Pcurr[pos(im1, jm1, k)]) + (Pcurr[pos(ip1, jm1, k)] - Pcurr[pos(im1, jp1, k)]) * (Pcurr[pos(ip1, jm1, k)] - Pcurr[pos(im1, jp1, k)]);      // phxy110a and phxy110b
      Grad110 = Grad110 + (Pcurr[pos(ip1, j, kp1)] - Pcurr[pos(im1, j, km1)]) * (Pcurr[pos(ip1, j, kp1)] - Pcurr[pos(im1, j, km1)]) + (Pcurr[pos(im1, j, kp1)] - Pcurr[pos(ip1, j, km1)]) * (Pcurr[pos(im1, j, kp1)] - Pcurr[pos(ip1, j, km1)]); // phxz110a and phxz110b
      Grad110 = Grad110 + (Pcurr[pos(i, jp1, kp1)] - Pcurr[pos(i, jm1, km1)]) * (Pcurr[pos(i, jp1, kp1)] - Pcurr[pos(i, jm1, km1)]) + (Pcurr[pos(i, jm1, kp1)] - Pcurr[pos(i, jp1, km1)]) * (Pcurr[pos(i, jm1, kp1)] - Pcurr[pos(i, jp1, km1)]); // phyz110a and phyz110b
      Grad110 = Grad110 / 16. / dxs;
      REAL Grad111 = (Pcurr[pos(ip1, jp1, kp1)] - Pcurr[pos(im1, jm1, km1)]) * (Pcurr[pos(ip1, jp1, kp1)] - Pcurr[pos(im1, jm1, km1)]);
      Grad111 = Grad111 + (Pcurr[pos(ip1, jp1, km1)] - Pcurr[pos(im1, jm1, kp1)]) * (Pcurr[pos(ip1, jp1, km1)] - Pcurr[pos(im1, jm1, kp1)]);
      Grad111 = Grad111 + (Pcurr[pos(ip1, jm1, kp1)] - Pcurr[pos(im1, jp1, km1)]) * (Pcurr[pos(ip1, jm1, kp1)] - Pcurr[pos(im1, jp1, km1)]);
      Grad111 = Grad111 + (Pcurr[pos(im1, jp1, kp1)] - Pcurr[pos(ip1, jm1, km1)]) * (Pcurr[pos(im1, jp1, kp1)] - Pcurr[pos(ip1, jm1, km1)]);
      Grad111 = Grad111 / 144. / dxs;

      sumnn = Lap3 - phi * sqrt2 * (4. * Grad100 / 9. + 4. * Grad110 / 9. + Grad111 / 9.);

      Pnext[pos(i, j, k)] = psi + dt * (sqrt2 * phi - sqrt2 * omp2 * Lambda * (Ucurr[pos(i, j, k)] + temp) + anis * prefacs) / (xi * xi * prefacs) / (Tau) + dt * sumnn / Tau;
#endif

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
    }

#if (NOISE != WITHOUT)
#if (tmax_NOISE > 0)
    if (iter * dt * Tau0_sec < 60. * tmax_NOISE)
#endif
    {
#if (NOISE == GAUSSIAN)
      curandState localState;
      localState = States[pos(i, j, k)];
      REAL ran1 = curand_normal(&localState);
      Pnext[pos(i, j, k)] += sqrt(2. * Fnoise * dt / dxs) * ran1;
      States[pos(i, j, k)] = localState;
#endif
#if (NOISE == FLAT)
      curandState localState;
      localState = States[pos(i, j, k)];
      REAL ran1 = curand_uniform_double(&localState);
      Pnext[pos(i, j, k)] += Fnoise * sqrt(dt) * (ran1 - .5);
      States[pos(i, j, k)] = localState;

#endif
    }
#endif
  }
}

#if (NewU == 0) // use 1 lattice shell <100>
__global__ void Compute_U(REAL *Pcurr, REAL *Pnext,
                          REAL *Ucurr, REAL *Unext,
                          REAL *F,
                          Constants *Ct,
                          Variables *Vb,
                          int DevNx, int AcmNx) {
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

  if (i * j * k * (i - DevNx - 1) * (j - Ny - 1) * (k - Nz - 1)) //  i.e. (i!=0 && j!=0 && k!=0 && i!=Nx+1 && j!=Ny+1 && k!=Nz+1)
  {

    // position variables
    int im1 = i - 1;
    int ip1 = i + 1;

    int jm1 = j - 1;
    int jp1 = j + 1;

    int km1 = k - 1;
    int kp1 = k + 1;

#if (MIXd)
    REAL edgegrid = MIXd / dx_microns + Xtip; // [dx]
    int intedgegrid = (int)(edgegrid);        // []
    REAL r = (edgegrid - intedgegrid) * dx;   // change unit to [W]

    if ((AcmNx + i) > intedgegrid) {
      Unext[pos(i, j, k)] = -1.;
    } else if ((AcmNx + i) < intedgegrid)
#endif
    {
      REAL u = Ucurr[pos(i, j, k)];
      REAL phi = F[pos(i, j, k)];
      REAL nphi = tanh(Pnext[pos(i, j, k)] / sqrt2);

      // ----
      // dUdt
      // ----

      Unext[pos(i, j, k)] = (1. + omk * u) * (nphi - phi) + dt * D * 0.5 * ((2. - F[pos(ip1, j, k)] - phi) * (Ucurr[pos(ip1, j, k)] - u) - (2. - F[pos(im1, j, k)] - phi) * (u - Ucurr[pos(im1, j, k)]) + (2. - F[pos(i, jp1, k)] - phi) * (Ucurr[pos(i, jp1, k)] - u) - (2. - F[pos(i, jm1, k)] - phi) * (u - Ucurr[pos(i, jm1, k)]) + (2. - F[pos(i, j, kp1)] - phi) * (Ucurr[pos(i, j, kp1)] - u) - (2. - F[pos(i, j, km1)] - phi) * (u - Ucurr[pos(i, j, km1)])) / dx / dx;

#if (ANTITRAPPING)
      //      if(fabs(omp2)>1.e-10)
      if (omp2 > 0.) {
        // -----------------------------------------------------
        // Source term: div[(dphi/dt)gradphi/|gradphi|)]/sqrt(2)
        // -----------------------------------------------------
        REAL dpxr, dpyr, dnormr, dpxl, dnorml, dpyu, dnormu, dpyd, dnormd;
        REAL dpzr, dpzt, dpzb, dpxb, dpyb, dnormt, dnormb;

        // The unit gradient may be computed either by Grad(h_Psi) or Grad(h_Phi)
        dpxr = F[pos(ip1, j, k)] - phi;
        dpyr = (F[pos(ip1, jp1, k)] + F[pos(i, jp1, k)] - F[pos(ip1, jm1, k)] - F[pos(i, jm1, k)]) / 4.;
        dpzr = (F[pos(ip1, j, kp1)] + F[pos(i, j, kp1)] - F[pos(ip1, j, km1)] - F[pos(i, j, km1)]) / 4.;
        dnormr = sqrt(dpxr * dpxr + dpyr * dpyr + dpzr * dpzr);

        dpxl = phi - F[pos(im1, j, k)];
        dpyl = (F[pos(im1, jp1, k)] + F[pos(i, jp1, k)] - F[pos(im1, jm1, k)] - F[pos(i, jm1, k)]) / 4.;
        dpzl = (F[pos(im1, j, kp1)] + F[pos(i, j, kp1)] - F[pos(im1, j, km1)] - F[pos(i, j, km1)]) / 4.;
        dnorml = sqrt(dpxl * dpxl + dpyl * dpyl + dpzl * dpzl);

        dpyu = F[pos(i, jp1, k)] - phi;
        dpxu = (F[pos(ip1, jp1, k)] + F[pos(ip1, j, k)] - F[pos(im1, jp1, k)] - F[pos(im1, j, k)]) / 4.;
        dpzu = (F[pos(i, jp1, kp1)] + F[pos(i, j, kp1)] - F[pos(i, jp1, km1)] - F[pos(i, j, km1)]) / 4.;
        dnormu = sqrt(dpxu * dpxu + dpyu * dpyu + dpzu * dpzu);

        dpyd = phi - F[pos(i, jm1, k)];
        dpxd = (F[pos(ip1, jm1, k)] + F[pos(ip1, j, k)] - F[pos(im1, jm1, k)] - F[pos(im1, j, k)]) / 4.;
        dpzd = (F[pos(i, jm1, kp1)] + F[pos(i, j, kp1)] - F[pos(i, jm1, km1)] - F[pos(i, j, km1)]) / 4.;
        dnormd = sqrt(dpxd * dpxd + dpyd * dpyd + dpzd * dpzd);

        dpzt = F[pos(i, j, kp1)] - phi;
        dpxt = (F[pos(ip1, j, kp1)] + F[pos(ip1, j, k)] - F[pos(im1, j, kp1)] - F[pos(im1, j, k)]) / 4.;
        dpyt = (F[pos(i, jp1, kp1)] + F[pos(i, jp1, k)] - F[pos(i, jm1, kp1)] - F[pos(i, jm1, k)]) / 4.;
        dnormt = sqrt(dpxt * dpxt + dpyt * dpyt + dpzt * dpzt);

        dpzb = phi - F[pos(i, j, km1)];
        dpxb = (F[pos(ip1, j, km1)] + F[pos(ip1, j, k)] - F[pos(im1, j, km1)] - F[pos(im1, j, k)]) / 4.;
        dpyb = (F[pos(i, jp1, km1)] + F[pos(i, jp1, k)] - F[pos(i, jm1, km1)] - F[pos(i, jm1, k)]) / 4.;
        dnormb = sqrt(dpxb * dpxb + dpyb * dpyb + dpzb * dpzb);

        if ((dnormr * dnorml * dnormu * dnormd * dnormt * dnormb) > 0.) {
          REAL omp2dpsi = omp2 * (Pnext[pos(i, j, k)] - Pcurr[pos(i, j, k)]);

          djxr = 0.25 * ((1. - F[pos(ip1, j, k)] * F[pos(ip1, j, k)]) * (Pnext[pos(ip1, j, k)] - Pcurr[pos(ip1, j, k)]) * (1. + omk * Ucurr[pos(ip1, j, k)]) + omp2dpsi * (1. + omk * u)) * dpxr / dnormr;
          djxl = 0.25 * ((1. - F[pos(im1, j, k)] * F[pos(im1, j, k)]) * (Pnext[pos(im1, j, k)] - Pcurr[pos(im1, j, k)]) * (1. + omk * Ucurr[pos(im1, j, k)]) + omp2dpsi * (1. + omk * u)) * dpxl / dnorml;
          djyu = 0.25 * ((1. - F[pos(i, jp1, k)] * F[pos(i, jp1, k)]) * (Pnext[pos(i, jp1, k)] - Pcurr[pos(i, jp1, k)]) * (1. + omk * Ucurr[pos(i, jp1, k)]) + omp2dpsi * (1. + omk * u)) * dpyu / dnormu;
          djyd = 0.25 * ((1. - F[pos(i, jm1, k)] * F[pos(i, jm1, k)]) * (Pnext[pos(i, jm1, k)] - Pcurr[pos(i, jm1, k)]) * (1. + omk * Ucurr[pos(i, jm1, k)]) + omp2dpsi * (1. + omk * u)) * dpyd / dnormd;
          djzt = 0.25 * ((1. - F[pos(i, j, kp1)] * F[pos(i, j, kp1)]) * (Pnext[pos(i, j, kp1)] - Pcurr[pos(i, j, kp1)]) * (1. + omk * Ucurr[pos(i, j, kp1)]) + omp2dpsi * (1. + omk * u)) * dpzt / dnormt;
          djzb = 0.25 * ((1. - F[pos(i, j, km1)] * F[pos(i, j, km1)]) * (Pnext[pos(i, j, km1)] - Pcurr[pos(i, j, km1)]) * (1. + omk * Ucurr[pos(i, j, km1)]) + omp2dpsi * (1. + omk * u)) * dpzb / dnormb;

          // ----
          // dUdt
          // ----
          Unext[pos(i, j, k)] += (djxr - djxl + djyu - djyd + djzt - djzb) / dx;
        }
      }
#endif

      // ----
      // dUdt
      // ----
      Unext[pos(i, j, k)] /= (opk - omk * nphi);
      Unext[pos(i, j, k)] += u;

#if (MIXd)
      if ((AcmNx + i) == (intedgegrid - 1)) {
        Unext[pos(i + 1, j, k)] = Unext[pos(i, j, k)] + (-1. - Unext[pos(i, j, k)]) / (dx + r) * dx;
      }
#endif
    }

    if (fabs(Unext[pos(i, j, k)]) > 1.1) {
      printf("(iter=%d)Ucurr(%d,%d,%d)=%g, next=%g,Warning!!!!!!!!\n", iter, i, j, k, Ucurr[pos(i, j, k)], Unext[pos(i, j, k)]);
    }
  }

  if ((!i) * (!j) * (!k)) // i.e. (i==0 && j==0 && k==0)
  {
    // iter++;
  }
}
#elif (NewU == 3) // use 1 lattice shell <100>, with the approximated anti-trapping, average flux
__global__ void Compute_U(REAL *Pcurr, REAL *Pnext,
                          REAL *Ucurr, REAL *Unext,
                          REAL *F,
                          Constants *Ct,
                          Variables *Vb,
                          int DevNx, int AcmNx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  if (i * j * k * (i - DevNx - 1) * (j - Ny - 1) * (k - Nz - 1)) //  i.e. (i!=0 && j!=0 && k!=0 && i!=Nx+1 && j!=Ny+1 && k!=Nz+1)
  {
    // position variables
    int im1 = i - 1;
    int ip1 = i + 1;

    int jm1 = j - 1;
    int jp1 = j + 1;

    int km1 = k - 1;
    int kp1 = k + 1;

#if (MIXd)
    REAL edgegrid = MIXd / dx_microns + Xtip; // [dx]
    int intedgegrid = (int)(edgegrid);        // []
    REAL r = (edgegrid - intedgegrid) * dx;   // change unit to [W]

    if ((AcmNx + i) > intedgegrid) {
      Unext[pos(i, j, k)] = -1.;
    } else if ((AcmNx + i) < intedgegrid)
#endif
    {
      REAL u = Ucurr[pos(i, j, k)];
      REAL phi = F[pos(i, j, k)];
      REAL nphi = tanh(Pnext[pos(i, j, k)] / sqrt2);

      if ((1. - phi) >= 1.e-6) // one-sided model, no diffusion and no anti-trapping in solid
      {
        REAL S000, S100, Sm100, S010, S0m10, S001, S00m1;                                        // Scalar field at the <100> lattice family
        REAL S110, S1m10, Sm110, Sm1m10, S101, S10m1, Sm101, Sm10m1, S011, S01m1, S0m11, S0m1m1; // Scalar field at the <110> lattice family
        REAL F100, F010, F001, Fm100, F0m10, F00m1;                                              // fluxes in the <100> directions
        REAL src100;

        // ------------------------
        // The diffusion term (*dt)
        // ------------------------
        S000 = 1. - F[pos(i, j, k)];

        S100 = 1. - F[pos(ip1, j, k)]; // Scalar field at the <100> lattice family
        Sm100 = 1. - F[pos(im1, j, k)];
        S010 = 1. - F[pos(i, jp1, k)];
        S0m10 = 1. - F[pos(i, jm1, k)];
        S001 = 1. - F[pos(i, j, kp1)];
        S00m1 = 1. - F[pos(i, j, km1)];

        S110 = 1. - F[pos(ip1, jp1, k)]; // Scalar field at the <110> lattice family
        S1m10 = 1. - F[pos(ip1, jm1, k)];
        Sm110 = 1. - F[pos(im1, jp1, k)];
        Sm1m10 = 1. - F[pos(im1, jm1, k)];
        S101 = 1. - F[pos(ip1, j, kp1)];
        S10m1 = 1. - F[pos(ip1, j, km1)];
        Sm101 = 1. - F[pos(im1, j, kp1)];
        Sm10m1 = 1. - F[pos(im1, j, km1)];
        S011 = 1. - F[pos(i, jp1, kp1)];
        S01m1 = 1. - F[pos(i, jp1, km1)];
        S0m11 = 1. - F[pos(i, jm1, kp1)];
        S0m1m1 = 1. - F[pos(i, jm1, km1)];

        F100 = (0.25 * (S100 + S000) + 0.0625 * (S110 + S010 + S001 + S101 + S1m10 + S0m10 + S00m1 + S10m1)) * (Ucurr[pos(ip1, j, k)] - u); // *dx
        F010 = (0.25 * (S010 + S000) + 0.0625 * (S110 + S100 + S001 + S011 + Sm110 + Sm100 + S00m1 + S01m1)) * (Ucurr[pos(i, jp1, k)] - u);
        F001 = (0.25 * (S001 + S000) + 0.0625 * (S101 + S100 + S010 + S011 + Sm101 + Sm100 + S0m10 + S0m11)) * (Ucurr[pos(i, j, kp1)] - u);
        Fm100 = (0.25 * (Sm100 + S000) + 0.0625 * (Sm110 + S010 + S001 + Sm101 + Sm1m10 + S0m10 + S00m1 + Sm10m1)) * (Ucurr[pos(im1, j, k)] - u);
        F0m10 = (0.25 * (S0m10 + S000) + 0.0625 * (S1m10 + S100 + S001 + S0m11 + Sm1m10 + Sm100 + S00m1 + S0m1m1)) * (Ucurr[pos(i, jm1, k)] - u);
        F00m1 = (0.25 * (S00m1 + S000) + 0.0625 * (S10m1 + S100 + S010 + S01m1 + Sm10m1 + Sm100 + S0m10 + S0m1m1)) * (Ucurr[pos(i, j, km1)] - u);

        src100 = (F100 + F010 + F001 + Fm100 + F0m10 + F00m1) / dx / dx; // the divergence evaluated by <100>

        // ----
        // dUdt
        // ----
        Unext[pos(i, j, k)] = (1. + omk * u) * (nphi - phi) + dt * D * src100;

#if (ANTITRAPPING)
        REAL psi = Pcurr[pos(i, j, k)];

        // the anti-trapping (*2*dt) with denominator by approximation
        S000 = (1. - F[pos(i, j, k)] * F[pos(i, j, k)]) * (Pnext[pos(i, j, k)] - Pcurr[pos(i, j, k)]) * (1. + omk * Ucurr[pos(i, j, k)]);
        S100 = (1. - F[pos(ip1, j, k)] * F[pos(ip1, j, k)]) * (Pnext[pos(ip1, j, k)] - Pcurr[pos(ip1, j, k)]) * (1. + omk * Ucurr[pos(ip1, j, k)]);
        Sm100 = (1. - F[pos(im1, j, k)] * F[pos(im1, j, k)]) * (Pnext[pos(im1, j, k)] - Pcurr[pos(im1, j, k)]) * (1. + omk * Ucurr[pos(im1, j, k)]);
        S010 = (1. - F[pos(i, jp1, k)] * F[pos(i, jp1, k)]) * (Pnext[pos(i, jp1, k)] - Pcurr[pos(i, jp1, k)]) * (1. + omk * Ucurr[pos(i, jp1, k)]);
        S0m10 = (1. - F[pos(i, jm1, k)] * F[pos(i, jm1, k)]) * (Pnext[pos(i, jm1, k)] - Pcurr[pos(i, jm1, k)]) * (1. + omk * Ucurr[pos(i, jm1, k)]);
        S001 = (1. - F[pos(i, j, kp1)] * F[pos(i, j, kp1)]) * (Pnext[pos(i, j, kp1)] - Pcurr[pos(i, j, kp1)]) * (1. + omk * Ucurr[pos(i, j, kp1)]);
        S00m1 = (1. - F[pos(i, j, km1)] * F[pos(i, j, km1)]) * (Pnext[pos(i, j, km1)] - Pcurr[pos(i, j, km1)]) * (1. + omk * Ucurr[pos(i, j, km1)]);

        S110 = (1. - F[pos(ip1, jp1, k)] * F[pos(ip1, jp1, k)]) * (Pnext[pos(ip1, jp1, k)] - Pcurr[pos(ip1, jp1, k)]) * (1. + omk * Ucurr[pos(ip1, jp1, k)]);
        S1m10 = (1. - F[pos(ip1, jm1, k)] * F[pos(ip1, jm1, k)]) * (Pnext[pos(ip1, jm1, k)] - Pcurr[pos(ip1, jm1, k)]) * (1. + omk * Ucurr[pos(ip1, jm1, k)]);
        Sm110 = (1. - F[pos(im1, jp1, k)] * F[pos(im1, jp1, k)]) * (Pnext[pos(im1, jp1, k)] - Pcurr[pos(im1, jp1, k)]) * (1. + omk * Ucurr[pos(im1, jp1, k)]);
        Sm1m10 = (1. - F[pos(im1, jm1, k)] * F[pos(im1, jm1, k)]) * (Pnext[pos(im1, jm1, k)] - Pcurr[pos(im1, jm1, k)]) * (1. + omk * Ucurr[pos(im1, jm1, k)]);
        S101 = (1. - F[pos(ip1, j, kp1)] * F[pos(ip1, j, kp1)]) * (Pnext[pos(ip1, j, kp1)] - Pcurr[pos(ip1, j, kp1)]) * (1. + omk * Ucurr[pos(ip1, j, kp1)]);
        S10m1 = (1. - F[pos(ip1, j, km1)] * F[pos(ip1, j, km1)]) * (Pnext[pos(ip1, j, km1)] - Pcurr[pos(ip1, j, km1)]) * (1. + omk * Ucurr[pos(ip1, j, km1)]);
        Sm101 = (1. - F[pos(im1, j, kp1)] * F[pos(im1, j, kp1)]) * (Pnext[pos(im1, j, kp1)] - Pcurr[pos(im1, j, kp1)]) * (1. + omk * Ucurr[pos(im1, j, kp1)]);
        Sm10m1 = (1. - F[pos(im1, j, km1)] * F[pos(im1, j, km1)]) * (Pnext[pos(im1, j, km1)] - Pcurr[pos(im1, j, km1)]) * (1. + omk * Ucurr[pos(im1, j, km1)]);
        S011 = (1. - F[pos(i, jp1, kp1)] * F[pos(i, jp1, kp1)]) * (Pnext[pos(i, jp1, kp1)] - Pcurr[pos(i, jp1, kp1)]) * (1. + omk * Ucurr[pos(i, jp1, kp1)]);
        S01m1 = (1. - F[pos(i, jp1, km1)] * F[pos(i, jp1, km1)]) * (Pnext[pos(i, jp1, km1)] - Pcurr[pos(i, jp1, km1)]) * (1. + omk * Ucurr[pos(i, jp1, km1)]);
        S0m11 = (1. - F[pos(i, jm1, kp1)] * F[pos(i, jm1, kp1)]) * (Pnext[pos(i, jm1, kp1)] - Pcurr[pos(i, jm1, kp1)]) * (1. + omk * Ucurr[pos(i, jm1, kp1)]);
        S0m1m1 = (1. - F[pos(i, jm1, km1)] * F[pos(i, jm1, km1)]) * (Pnext[pos(i, jm1, km1)] - Pcurr[pos(i, jm1, km1)]) * (1. + omk * Ucurr[pos(i, jm1, km1)]);

        F100 = (0.25 * (S100 + S000) + 0.0625 * (S110 + S010 + S001 + S101 + S1m10 + S0m10 + S00m1 + S10m1)) * (Pcurr[pos(ip1, j, k)] - psi); // *dx
        F010 = (0.25 * (S010 + S000) + 0.0625 * (S110 + S100 + S001 + S011 + Sm110 + Sm100 + S00m1 + S01m1)) * (Pcurr[pos(i, jp1, k)] - psi);
        F001 = (0.25 * (S001 + S000) + 0.0625 * (S101 + S100 + S010 + S011 + Sm101 + Sm100 + S0m10 + S0m11)) * (Pcurr[pos(i, j, kp1)] - psi);
        Fm100 = (0.25 * (Sm100 + S000) + 0.0625 * (Sm110 + S010 + S001 + Sm101 + Sm1m10 + S0m10 + S00m1 + Sm10m1)) * (Pcurr[pos(im1, j, k)] - psi);
        F0m10 = (0.25 * (S0m10 + S000) + 0.0625 * (S1m10 + S100 + S001 + S0m11 + Sm1m10 + Sm100 + S00m1 + S0m1m1)) * (Pcurr[pos(i, jm1, k)] - psi);
        F00m1 = (0.25 * (S00m1 + S000) + 0.0625 * (S10m1 + S100 + S010 + S01m1 + Sm10m1 + Sm100 + S0m10 + S0m1m1)) * (Pcurr[pos(i, j, km1)] - psi);

        src100 = (F100 + F010 + F001 + Fm100 + F0m10 + F00m1) / dx / dx; // the divergence evaluated by <100>

        Unext[pos(i, j, k)] += 0.5 * src100; // dt has been multiplied
#endif
      } else {
        Unext[pos(i, j, k)] = (1. + omk * u) * (nphi - phi);
      }

      // ----
      // dUdt
      // ----
      Unext[pos(i, j, k)] /= (opk - omk * nphi);
      Unext[pos(i, j, k)] += u;

#if (MIXd)
      if ((AcmNx + i) == (intedgegrid - 1)) {
        Unext[pos(i + 1, j, k)] = Unext[pos(i, j, k)] + (-1. - Unext[pos(i, j, k)]) / (dx + r) * dx;
      }
#endif
    }

    if (fabs(Unext[pos(i, j, k)]) > 1.1) {
      printf("(iter=%d)Ucurr(%d,%d,%d)=%g, next=%g,Warning!!!!!!!!\n", iter, i, j, k, Ucurr[pos(i, j, k)], Unext[pos(i, j, k)]);
    }
  }

  if ((!i) * (!j) * (!k)) // i.e. (i==0 && j==0 && k==0)
  {
    // iter++;
  }
}
#else // use 2 or 3 lattice shells
__global__ void Compute_U(REAL *Pcurr, REAL *Pnext,
                          REAL *Ucurr, REAL *Unext,
                          REAL *F,
                          Constants *Ct,
                          Variables *Vb,
                          int DevNx, int AcmNx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  if (i * j * k * (i - DevNx - 1) * (j - Ny - 1) * (k - Nz - 1)) //	i.e. (i!=0 && j!=0 && k!=0 && i!=Nx+1 && j!=Ny+1 && k!=Nz+1)
  {
    // position variables
    int im1 = i - 1;
    int ip1 = i + 1;

    int jm1 = j - 1;
    int jp1 = j + 1;

    int km1 = k - 1;
    int kp1 = k + 1;

#if (MIXd)
    REAL edgegrid = MIXd / dx_microns + Xtip; // [dx]
    int intedgegrid = (int)(edgegrid);        // []
    REAL r = (edgegrid - intedgegrid) * dx;   // change unit to [W]

    if ((AcmNx + i) > intedgegrid) {
      Unext[pos(i, j, k)] = -1.;
    } else if ((AcmNx + i) < intedgegrid)
#endif
    {
      REAL u = Ucurr[pos(i, j, k)];
      REAL phi = F[pos(i, j, k)];
      REAL nphi = tanh(Pnext[pos(i, j, k)] / sqrt2);

      if ((1. - phi) >= 1.e-6) // one-sided model, no diffusion and no anti-trapping in solid
      {
        REAL S000, S100, Sm100, S010, S0m10, S001, S00m1;                                        // Scalar field at the <100> lattice family
        REAL S110, S1m10, Sm110, Sm1m10, S101, S10m1, Sm101, Sm10m1, S011, S01m1, S0m11, S0m1m1; // Scalar field at the <110> lattice family
        REAL S111, Sm111, S1m11, S11m1, Sm1m11, Sm11m1, S1m1m1, Sm1m1m1;                         // Scalar field at the <111> lattice family
        REAL F100, F010, F001, Fm100, F0m10, F00m1;                                              // fluxes in the <100> directions
        REAL F110, Fm110, F1m10, Fm1m10, F101, Fm101, F10m1, Fm10m1, F011, F0m11, F01m1, F0m1m1; // fluxes in the <111> directions
        REAL src100, src110;

        // ------------------------
        // The diffusion term (*dt)
        // ------------------------
        S000 = 1. - F[pos(i, j, k)];

        S100 = 1. - F[pos(ip1, j, k)]; // Scalar field at the <100> lattice family
        Sm100 = 1. - F[pos(im1, j, k)];
        S010 = 1. - F[pos(i, jp1, k)];
        S0m10 = 1. - F[pos(i, jm1, k)];
        S001 = 1. - F[pos(i, j, kp1)];
        S00m1 = 1. - F[pos(i, j, km1)];

        S110 = 1. - F[pos(ip1, jp1, k)]; // Scalar field at the <110> lattice family
        S1m10 = 1. - F[pos(ip1, jm1, k)];
        Sm110 = 1. - F[pos(im1, jp1, k)];
        Sm1m10 = 1. - F[pos(im1, jm1, k)];
        S101 = 1. - F[pos(ip1, j, kp1)];
        S10m1 = 1. - F[pos(ip1, j, km1)];
        Sm101 = 1. - F[pos(im1, j, kp1)];
        Sm10m1 = 1. - F[pos(im1, j, km1)];
        S011 = 1. - F[pos(i, jp1, kp1)];
        S01m1 = 1. - F[pos(i, jp1, km1)];
        S0m11 = 1. - F[pos(i, jm1, kp1)];
        S0m1m1 = 1. - F[pos(i, jm1, km1)];

        S111 = 1. - F[pos(ip1, jp1, kp1)]; // Scalar field at the <111> lattice family
        Sm111 = 1. - F[pos(im1, jp1, kp1)];
        S1m11 = 1. - F[pos(ip1, jm1, kp1)];
        S11m1 = 1. - F[pos(ip1, jp1, km1)];
        Sm1m11 = 1. - F[pos(im1, jm1, kp1)];
        Sm11m1 = 1. - F[pos(im1, jp1, km1)];
        S1m1m1 = 1. - F[pos(ip1, jm1, km1)];
        Sm1m1m1 = 1. - F[pos(im1, jm1, km1)];

        F100 = (0.25 * (S100 + S000) + 0.0625 * (S110 + S010 + S001 + S101 + S1m10 + S0m10 + S00m1 + S10m1)) * (Ucurr[pos(ip1, j, k)] - u); // *dx
        F010 = (0.25 * (S010 + S000) + 0.0625 * (S110 + S100 + S001 + S011 + Sm110 + Sm100 + S00m1 + S01m1)) * (Ucurr[pos(i, jp1, k)] - u);
        F001 = (0.25 * (S001 + S000) + 0.0625 * (S101 + S100 + S010 + S011 + Sm101 + Sm100 + S0m10 + S0m11)) * (Ucurr[pos(i, j, kp1)] - u);
        Fm100 = (0.25 * (Sm100 + S000) + 0.0625 * (Sm110 + S010 + S001 + Sm101 + Sm1m10 + S0m10 + S00m1 + Sm10m1)) * (Ucurr[pos(im1, j, k)] - u);
        F0m10 = (0.25 * (S0m10 + S000) + 0.0625 * (S1m10 + S100 + S001 + S0m11 + Sm1m10 + Sm100 + S00m1 + S0m1m1)) * (Ucurr[pos(i, jm1, k)] - u);
        F00m1 = (0.25 * (S00m1 + S000) + 0.0625 * (S10m1 + S100 + S010 + S01m1 + Sm10m1 + Sm100 + S0m10 + S0m1m1)) * (Ucurr[pos(i, j, km1)] - u);

        src100 = (F100 + F010 + F001 + Fm100 + F0m10 + F00m1) / dx / dx; // the divergence evaluated by <100>

        F110 = (0.1875 * (S100 + S110 + S010 + S000) + 0.03125 * (S111 + S101 + S011 + S001 + S11m1 + S10m1 + S01m1 + S00m1)) * (Ucurr[pos(ip1, jp1, k)] - u); // Sqrt(2)*dx
        Fm110 = (0.1875 * (Sm100 + Sm110 + S010 + S000) + 0.03125 * (Sm111 + Sm101 + S011 + S001 + Sm11m1 + Sm10m1 + S01m1 + S00m1)) * (Ucurr[pos(im1, jp1, k)] - u);
        F1m10 = (0.1875 * (S100 + S1m10 + S0m10 + S000) + 0.03125 * (S1m11 + S101 + S0m11 + S001 + S1m1m1 + S10m1 + S0m1m1 + S00m1)) * (Ucurr[pos(ip1, jm1, k)] - u);
        Fm1m10 = (0.1875 * (Sm100 + Sm1m10 + S0m10 + S000) + 0.03125 * (Sm1m11 + Sm101 + S0m11 + S001 + Sm1m1m1 + Sm10m1 + S0m1m1 + S00m1)) * (Ucurr[pos(im1, jm1, k)] - u);
        F101 = (0.1875 * (S100 + S101 + S001 + S000) + 0.03125 * (S111 + S110 + S011 + S010 + S1m11 + S1m10 + S0m11 + S0m10)) * (Ucurr[pos(ip1, j, kp1)] - u);
        Fm101 = (0.1875 * (Sm100 + Sm101 + S001 + S000) + 0.03125 * (Sm111 + Sm110 + S011 + S010 + Sm1m11 + Sm1m10 + S0m11 + S0m10)) * (Ucurr[pos(im1, j, kp1)] - u);
        F10m1 = (0.1875 * (S100 + S10m1 + S00m1 + S000) + 0.03125 * (S11m1 + S110 + S01m1 + S010 + S1m1m1 + S1m10 + S0m1m1 + S0m10)) * (Ucurr[pos(ip1, j, km1)] - u);
        Fm10m1 = (0.1875 * (Sm100 + Sm10m1 + S00m1 + S000) + 0.03125 * (Sm11m1 + Sm110 + S01m1 + S010 + Sm1m1m1 + Sm1m10 + S0m1m1 + S0m10)) * (Ucurr[pos(im1, j, km1)] - u);
        F011 = (0.1875 * (S001 + S011 + S010 + S000) + 0.03125 * (S111 + S110 + S101 + S100 + Sm111 + Sm110 + Sm101 + Sm100)) * (Ucurr[pos(i, jp1, kp1)] - u);
        F0m11 = (0.1875 * (S001 + S0m11 + S0m10 + S000) + 0.03125 * (S1m11 + S1m10 + S101 + S100 + Sm1m11 + Sm1m10 + Sm101 + Sm100)) * (Ucurr[pos(i, jm1, kp1)] - u);
        F01m1 = (0.1875 * (S00m1 + S01m1 + S010 + S000) + 0.03125 * (S11m1 + S110 + S10m1 + S100 + Sm11m1 + Sm110 + Sm10m1 + Sm100)) * (Ucurr[pos(i, jp1, km1)] - u);
        F0m1m1 = (0.1875 * (S00m1 + S0m1m1 + S0m10 + S000) + 0.03125 * (S1m1m1 + S1m10 + S10m1 + S100 + Sm1m1m1 + Sm1m10 + Sm10m1 + Sm100)) * (Ucurr[pos(i, jm1, km1)] - u);

        src110 = 0.25 * (F110 + Fm110 + F1m10 + Fm1m10 + F101 + Fm101 + F10m1 + Fm10m1 + F011 + F0m11 + F01m1 + F0m1m1) / dx / dx; // the divergence evaluated by <110>

#if (NewU == 2) // use 3 lattice shells
        REAL src111, F111, Fm111, F1m11, F11m1, Fm1m11, Fm11m1, F1m1m1, Fm1m1m1;

        F111 = 0.125 * (S000 + S100 + S010 + S001 + S110 + S101 + S011 + S111) * (Ucurr[pos(ip1, jp1, kp1)] - u); // Sqrt(3)*dx
        Fm111 = 0.125 * (S000 + Sm100 + S010 + S001 + Sm110 + Sm101 + S011 + Sm111) * (Ucurr[pos(im1, jp1, kp1)] - u);
        F1m11 = 0.125 * (S000 + S100 + S0m10 + S001 + S1m10 + S101 + S0m11 + S1m11) * (Ucurr[pos(ip1, jm1, kp1)] - u);
        F11m1 = 0.125 * (S000 + S100 + S010 + S00m1 + S110 + S10m1 + S01m1 + S11m1) * (Ucurr[pos(ip1, jp1, km1)] - u);
        Fm1m11 = 0.125 * (S000 + Sm100 + S0m10 + S001 + Sm1m10 + Sm101 + S0m11 + Sm1m11) * (Ucurr[pos(im1, jm1, kp1)] - u);
        Fm11m1 = 0.125 * (S000 + Sm100 + S010 + S00m1 + Sm110 + Sm10m1 + S01m1 + Sm11m1) * (Ucurr[pos(im1, jp1, km1)] - u);
        F1m1m1 = 0.125 * (S000 + S100 + S0m10 + S00m1 + S1m10 + S10m1 + S0m1m1 + S1m1m1) * (Ucurr[pos(ip1, jm1, km1)] - u);
        Fm1m1m1 = 0.125 * (S000 + Sm100 + S0m10 + S00m1 + Sm1m10 + Sm10m1 + S0m1m1 + Sm1m1m1) * (Ucurr[pos(im1, jm1, km1)] - u);

        src111 = 0.25 * (F111 + Fm111 + F1m11 + F11m1 + Fm1m11 + Fm11m1 + F1m1m1 + Fm1m1m1) / dx / dx; // the divergence evaluated by <111>
#endif

        // ----
        // dUdt
        // ----
#if (NewU == 1)
        Unext[pos(i, j, k)] = (1. + omk * u) * (nphi - phi) + dt * D * (src100 / 3. + 2. * src110 / 3.);
#elif (NewU == 2)
        Unext[pos(i, j, k)] = (1. + omk * u) * (nphi - phi) + dt * D * (5. * src100 / 9. + 2. * src110 / 9. + 2. * src111 / 9.);
#endif

#if (ANTITRAPPING)
        REAL psi = Pcurr[pos(i, j, k)];

        // the anti-trapping (*2*dt) with denominator by approximation
        S000 = (1. - F[pos(i, j, k)] * F[pos(i, j, k)]) * (Pnext[pos(i, j, k)] - Pcurr[pos(i, j, k)]) * (1. + omk * Ucurr[pos(i, j, k)]);
        S100 = (1. - F[pos(ip1, j, k)] * F[pos(ip1, j, k)]) * (Pnext[pos(ip1, j, k)] - Pcurr[pos(ip1, j, k)]) * (1. + omk * Ucurr[pos(ip1, j, k)]);
        Sm100 = (1. - F[pos(im1, j, k)] * F[pos(im1, j, k)]) * (Pnext[pos(im1, j, k)] - Pcurr[pos(im1, j, k)]) * (1. + omk * Ucurr[pos(im1, j, k)]);
        S010 = (1. - F[pos(i, jp1, k)] * F[pos(i, jp1, k)]) * (Pnext[pos(i, jp1, k)] - Pcurr[pos(i, jp1, k)]) * (1. + omk * Ucurr[pos(i, jp1, k)]);
        S0m10 = (1. - F[pos(i, jm1, k)] * F[pos(i, jm1, k)]) * (Pnext[pos(i, jm1, k)] - Pcurr[pos(i, jm1, k)]) * (1. + omk * Ucurr[pos(i, jm1, k)]);
        S001 = (1. - F[pos(i, j, kp1)] * F[pos(i, j, kp1)]) * (Pnext[pos(i, j, kp1)] - Pcurr[pos(i, j, kp1)]) * (1. + omk * Ucurr[pos(i, j, kp1)]);
        S00m1 = (1. - F[pos(i, j, km1)] * F[pos(i, j, km1)]) * (Pnext[pos(i, j, km1)] - Pcurr[pos(i, j, km1)]) * (1. + omk * Ucurr[pos(i, j, km1)]);

        S110 = (1. - F[pos(ip1, jp1, k)] * F[pos(ip1, jp1, k)]) * (Pnext[pos(ip1, jp1, k)] - Pcurr[pos(ip1, jp1, k)]) * (1. + omk * Ucurr[pos(ip1, jp1, k)]);
        S1m10 = (1. - F[pos(ip1, jm1, k)] * F[pos(ip1, jm1, k)]) * (Pnext[pos(ip1, jm1, k)] - Pcurr[pos(ip1, jm1, k)]) * (1. + omk * Ucurr[pos(ip1, jm1, k)]);
        Sm110 = (1. - F[pos(im1, jp1, k)] * F[pos(im1, jp1, k)]) * (Pnext[pos(im1, jp1, k)] - Pcurr[pos(im1, jp1, k)]) * (1. + omk * Ucurr[pos(im1, jp1, k)]);
        Sm1m10 = (1. - F[pos(im1, jm1, k)] * F[pos(im1, jm1, k)]) * (Pnext[pos(im1, jm1, k)] - Pcurr[pos(im1, jm1, k)]) * (1. + omk * Ucurr[pos(im1, jm1, k)]);
        S101 = (1. - F[pos(ip1, j, kp1)] * F[pos(ip1, j, kp1)]) * (Pnext[pos(ip1, j, kp1)] - Pcurr[pos(ip1, j, kp1)]) * (1. + omk * Ucurr[pos(ip1, j, kp1)]);
        S10m1 = (1. - F[pos(ip1, j, km1)] * F[pos(ip1, j, km1)]) * (Pnext[pos(ip1, j, km1)] - Pcurr[pos(ip1, j, km1)]) * (1. + omk * Ucurr[pos(ip1, j, km1)]);
        Sm101 = (1. - F[pos(im1, j, kp1)] * F[pos(im1, j, kp1)]) * (Pnext[pos(im1, j, kp1)] - Pcurr[pos(im1, j, kp1)]) * (1. + omk * Ucurr[pos(im1, j, kp1)]);
        Sm10m1 = (1. - F[pos(im1, j, km1)] * F[pos(im1, j, km1)]) * (Pnext[pos(im1, j, km1)] - Pcurr[pos(im1, j, km1)]) * (1. + omk * Ucurr[pos(im1, j, km1)]);
        S011 = (1. - F[pos(i, jp1, kp1)] * F[pos(i, jp1, kp1)]) * (Pnext[pos(i, jp1, kp1)] - Pcurr[pos(i, jp1, kp1)]) * (1. + omk * Ucurr[pos(i, jp1, kp1)]);
        S01m1 = (1. - F[pos(i, jp1, km1)] * F[pos(i, jp1, km1)]) * (Pnext[pos(i, jp1, km1)] - Pcurr[pos(i, jp1, km1)]) * (1. + omk * Ucurr[pos(i, jp1, km1)]);
        S0m11 = (1. - F[pos(i, jm1, kp1)] * F[pos(i, jm1, kp1)]) * (Pnext[pos(i, jm1, kp1)] - Pcurr[pos(i, jm1, kp1)]) * (1. + omk * Ucurr[pos(i, jm1, kp1)]);
        S0m1m1 = (1. - F[pos(i, jm1, km1)] * F[pos(i, jm1, km1)]) * (Pnext[pos(i, jm1, km1)] - Pcurr[pos(i, jm1, km1)]) * (1. + omk * Ucurr[pos(i, jm1, km1)]);

        S111 = (1. - F[pos(ip1, jp1, kp1)] * F[pos(ip1, jp1, kp1)]) * (Pnext[pos(ip1, jp1, kp1)] - Pcurr[pos(ip1, jp1, kp1)]) * (1. + omk * Ucurr[pos(ip1, jp1, kp1)]);
        Sm111 = (1. - F[pos(im1, jp1, kp1)] * F[pos(im1, jp1, kp1)]) * (Pnext[pos(im1, jp1, kp1)] - Pcurr[pos(im1, jp1, kp1)]) * (1. + omk * Ucurr[pos(im1, jp1, kp1)]);
        S1m11 = (1. - F[pos(ip1, jm1, kp1)] * F[pos(ip1, jm1, kp1)]) * (Pnext[pos(ip1, jm1, kp1)] - Pcurr[pos(ip1, jm1, kp1)]) * (1. + omk * Ucurr[pos(ip1, jm1, kp1)]);
        S11m1 = (1. - F[pos(ip1, jp1, km1)] * F[pos(ip1, jp1, km1)]) * (Pnext[pos(ip1, jp1, km1)] - Pcurr[pos(ip1, jp1, km1)]) * (1. + omk * Ucurr[pos(ip1, jp1, km1)]);
        Sm1m11 = (1. - F[pos(im1, jm1, kp1)] * F[pos(im1, jm1, kp1)]) * (Pnext[pos(im1, jm1, kp1)] - Pcurr[pos(im1, jm1, kp1)]) * (1. + omk * Ucurr[pos(im1, jm1, kp1)]);
        Sm11m1 = (1. - F[pos(im1, jp1, km1)] * F[pos(im1, jp1, km1)]) * (Pnext[pos(im1, jp1, km1)] - Pcurr[pos(im1, jp1, km1)]) * (1. + omk * Ucurr[pos(im1, jp1, km1)]);
        S1m1m1 = (1. - F[pos(ip1, jm1, km1)] * F[pos(ip1, jm1, km1)]) * (Pnext[pos(ip1, jm1, km1)] - Pcurr[pos(ip1, jm1, km1)]) * (1. + omk * Ucurr[pos(ip1, jm1, km1)]);
        Sm1m1m1 = (1. - F[pos(im1, jm1, km1)] * F[pos(im1, jm1, km1)]) * (Pnext[pos(im1, jm1, km1)] - Pcurr[pos(im1, jm1, km1)]) * (1. + omk * Ucurr[pos(im1, jm1, km1)]);

        F100 = (0.25 * (S100 + S000) + 0.0625 * (S110 + S010 + S001 + S101 + S1m10 + S0m10 + S00m1 + S10m1)) * (Pcurr[pos(ip1, j, k)] - psi); // *dx
        F010 = (0.25 * (S010 + S000) + 0.0625 * (S110 + S100 + S001 + S011 + Sm110 + Sm100 + S00m1 + S01m1)) * (Pcurr[pos(i, jp1, k)] - psi);
        F001 = (0.25 * (S001 + S000) + 0.0625 * (S101 + S100 + S010 + S011 + Sm101 + Sm100 + S0m10 + S0m11)) * (Pcurr[pos(i, j, kp1)] - psi);
        Fm100 = (0.25 * (Sm100 + S000) + 0.0625 * (Sm110 + S010 + S001 + Sm101 + Sm1m10 + S0m10 + S00m1 + Sm10m1)) * (Pcurr[pos(im1, j, k)] - psi);
        F0m10 = (0.25 * (S0m10 + S000) + 0.0625 * (S1m10 + S100 + S001 + S0m11 + Sm1m10 + Sm100 + S00m1 + S0m1m1)) * (Pcurr[pos(i, jm1, k)] - psi);
        F00m1 = (0.25 * (S00m1 + S000) + 0.0625 * (S10m1 + S100 + S010 + S01m1 + Sm10m1 + Sm100 + S0m10 + S0m1m1)) * (Pcurr[pos(i, j, km1)] - psi);

        src100 = (F100 + F010 + F001 + Fm100 + F0m10 + F00m1) / dx / dx; // the divergence evaluated by <100>

        F110 = (0.1875 * (S100 + S110 + S010 + S000) + 0.03125 * (S111 + S101 + S011 + S001 + S11m1 + S10m1 + S01m1 + S00m1)) * (Pcurr[pos(ip1, jp1, k)] - psi); // Sqrt(2)*dx
        Fm110 = (0.1875 * (Sm100 + Sm110 + S010 + S000) + 0.03125 * (Sm111 + Sm101 + S011 + S001 + Sm11m1 + Sm10m1 + S01m1 + S00m1)) * (Pcurr[pos(im1, jp1, k)] - psi);
        F1m10 = (0.1875 * (S100 + S1m10 + S0m10 + S000) + 0.03125 * (S1m11 + S101 + S0m11 + S001 + S1m1m1 + S10m1 + S0m1m1 + S00m1)) * (Pcurr[pos(ip1, jm1, k)] - psi);
        Fm1m10 = (0.1875 * (Sm100 + Sm1m10 + S0m10 + S000) + 0.03125 * (Sm1m11 + Sm101 + S0m11 + S001 + Sm1m1m1 + Sm10m1 + S0m1m1 + S00m1)) * (Pcurr[pos(im1, jm1, k)] - psi);
        F101 = (0.1875 * (S100 + S101 + S001 + S000) + 0.03125 * (S111 + S110 + S011 + S010 + S1m11 + S1m10 + S0m11 + S0m10)) * (Pcurr[pos(ip1, j, kp1)] - psi);
        Fm101 = (0.1875 * (Sm100 + Sm101 + S001 + S000) + 0.03125 * (Sm111 + Sm110 + S011 + S010 + Sm1m11 + Sm1m10 + S0m11 + S0m10)) * (Pcurr[pos(im1, j, kp1)] - psi);
        F10m1 = (0.1875 * (S100 + S10m1 + S00m1 + S000) + 0.03125 * (S11m1 + S110 + S01m1 + S010 + S1m1m1 + S1m10 + S0m1m1 + S0m10)) * (Pcurr[pos(ip1, j, km1)] - psi);
        Fm10m1 = (0.1875 * (Sm100 + Sm10m1 + S00m1 + S000) + 0.03125 * (Sm11m1 + Sm110 + S01m1 + S010 + Sm1m1m1 + Sm1m10 + S0m1m1 + S0m10)) * (Pcurr[pos(im1, j, km1)] - psi);
        F011 = (0.1875 * (S001 + S011 + S010 + S000) + 0.03125 * (S111 + S110 + S101 + S100 + Sm111 + Sm110 + Sm101 + Sm100)) * (Pcurr[pos(i, jp1, kp1)] - psi);
        F0m11 = (0.1875 * (S001 + S0m11 + S0m10 + S000) + 0.03125 * (S1m11 + S1m10 + S101 + S100 + Sm1m11 + Sm1m10 + Sm101 + Sm100)) * (Pcurr[pos(i, jm1, kp1)] - psi);
        F01m1 = (0.1875 * (S00m1 + S01m1 + S010 + S000) + 0.03125 * (S11m1 + S110 + S10m1 + S100 + Sm11m1 + Sm110 + Sm10m1 + Sm100)) * (Pcurr[pos(i, jp1, km1)] - psi);
        F0m1m1 = (0.1875 * (S00m1 + S0m1m1 + S0m10 + S000) + 0.03125 * (S1m1m1 + S1m10 + S10m1 + S100 + Sm1m1m1 + Sm1m10 + Sm10m1 + Sm100)) * (Pcurr[pos(i, jm1, km1)] - psi);

        src110 = 0.25 * (F110 + Fm110 + F1m10 + Fm1m10 + F101 + Fm101 + F10m1 + Fm10m1 + F011 + F0m11 + F01m1 + F0m1m1) / dx / dx; // the divergence evaluated by <110>

#if (NewU == 2) // use 3 lattice shells
        F111 = 0.125 * (S000 + S100 + S010 + S001 + S110 + S101 + S011 + S111) * (Pcurr[pos(ip1, jp1, kp1)] - psi); // Sqrt(3)*dx
        Fm111 = 0.125 * (S000 + Sm100 + S010 + S001 + Sm110 + Sm101 + S011 + Sm111) * (Pcurr[pos(im1, jp1, kp1)] - psi);
        F1m11 = 0.125 * (S000 + S100 + S0m10 + S001 + S1m10 + S101 + S0m11 + S1m11) * (Pcurr[pos(ip1, jm1, kp1)] - psi);
        F11m1 = 0.125 * (S000 + S100 + S010 + S00m1 + S110 + S10m1 + S01m1 + S11m1) * (Pcurr[pos(ip1, jp1, km1)] - psi);
        Fm1m11 = 0.125 * (S000 + Sm100 + S0m10 + S001 + Sm1m10 + Sm101 + S0m11 + Sm1m11) * (Pcurr[pos(im1, jm1, kp1)] - psi);
        Fm11m1 = 0.125 * (S000 + Sm100 + S010 + S00m1 + Sm110 + Sm10m1 + S01m1 + Sm11m1) * (Pcurr[pos(im1, jp1, km1)] - psi);
        F1m1m1 = 0.125 * (S000 + S100 + S0m10 + S00m1 + S1m10 + S10m1 + S0m1m1 + S1m1m1) * (Pcurr[pos(ip1, jm1, km1)] - psi);
        Fm1m1m1 = 0.125 * (S000 + Sm100 + S0m10 + S00m1 + Sm1m10 + Sm10m1 + S0m1m1 + Sm1m1m1) * (Pcurr[pos(im1, jm1, km1)] - psi);

        src111 = 0.25 * (F111 + Fm111 + F1m11 + F11m1 + Fm1m11 + Fm11m1 + F1m1m1 + Fm1m1m1) / dx / dx; // the divergence evaluated by <111>
#endif

#if (NewU == 1)
        Unext[pos(i, j, k)] += 0.5 * (src100 / 3. + 2. * src110 / 3.); // dt has been multiplied
#elif (NewU == 2)
        Unext[pos(i, j, k)] += 0.5 * (5. * src100 / 9. + 2. * src110 / 9. + 2. * src111 / 9.); // dt has been multiplied
#endif
#endif
      } else {
        Unext[pos(i, j, k)] = (1. + omk * u) * (nphi - phi);
      }

      // ----
      // dUdt
      // ----
      Unext[pos(i, j, k)] /= (opk - omk * nphi);
      Unext[pos(i, j, k)] += u;

#if (MIXd)
      if ((AcmNx + i) == (intedgegrid - 1)) {
        Unext[pos(i + 1, j, k)] = Unext[pos(i, j, k)] + (-1. - Unext[pos(i, j, k)]) / (dx + r) * dx;
      }
#endif
    }

    if (fabs(Unext[pos(i, j, k)]) > 1.1) {
      printf("(iter=%d)Ucurr(%d,%d,%d)=%g, next=%g,Warning!!!!!!!!\n", iter, i, j, k, Ucurr[pos(i, j, k)], Unext[pos(i, j, k)]);
    }
  }

  if ((!i) * (!j) * (!k)) // i.e. (i==0 && j==0 && k==0)
  {
    // iter++;
  }
}
#endif

#if (InnerBCL > 0)
__global__ void Compute_P_BCin(REAL *Pcurr, REAL *Pnext,
                               REAL *Ucurr,
                               REAL *F,
                               signed char *Phase,
                               curandState *state,
                               Constants *Ct,
                               Variables *Vb,
                               int DevNx, int AcmNx,
                               REAL *Tsbox) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  // update current psi, phi, pha field including boundaries
  REAL psi = Pcurr[pos(i, j, k)];
  REAL phi = F[pos(i, j, k)] = tanh(psi / sqrt2);
  signed char pha = Phase[pos(i, j, k)];

  Pnext[pos(i, j, k)] = psi;

  if (i * j * k * (i - DevNx - 1) * (j - Ny - 1) * (k - Nz - 1)) //  i.e. (i!=0 && j!=0 && k!=0 && i!=Nx+1 && j!=Ny+1 && k!=Nz+1)
  {
    // position variables
    int i0 = i;
    int im1 = i - 1;
    int ip1 = i + 1;

    int j0 = j;
    int jm1 = j - 1;
    int jp1 = j + 1;

    int k0 = k;
    int km1 = k - 1;
    int kp1 = k + 1;

    // set maximum length limit for inner noflux boundaries
    int iGB = (int)(((1 + InnerBCL / 10.) * xint - xoffs) / dx);
    iGB = (iGB > 0.9 * Nx) ? 0.9 * Nx : iGB;

    if ((AcmNx + i) < iGB) {
      if (j == jGBy) {
        // Field(i,jGB,k) = Field(i,jGB-1,k) noflux boundary locates in between (jGB-1) and jGB
        j--;
        jm1--;
        jp1--;

        // re-calculate current fields for the inner boundaries
        psi = Pcurr[pos(i, j, k)];
        phi = tanh(psi / sqrt2);
        pha = Phase[pos(i, j, k)];

      } else if (j == jGBy + 1) {
        // Field(i,jGB+1,k) = Field(i,jGB+2,k) noflux boundary locates in between (jGB+1) and (jGB+2)
        j++;
        jm1++;
        jp1++;

        // re-calculate current fields for the inner boundaries
        psi = Pcurr[pos(i, j, k)];
        phi = tanh(psi / sqrt2);
        pha = Phase[pos(i, j, k)];
      }
    }
    // for inneer BC

#if (WALLEFFECT > 0)
    if ((k0 - 1) * (k0 - Nz) == 0) {

      Pnext[pos(i0, j0, k0)] = Pnext[pos(i0, j0, k0)];

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

      // first derivatives (each coordinate)
      REAL phx = ((Pcurr[pos(ip1, j, k)] - Pcurr[pos(im1, j, k)]) / dx2);
      REAL phy = ((Pcurr[pos(i, jp1, k)] - Pcurr[pos(i, jm1, k)]) / dx2);
      REAL phz = ((Pcurr[pos(i, j, kp1)] - Pcurr[pos(i, j, km1)]) / dx2);

      // second deriviatives - sqrt2 * phi * (first derivatives)^2
      REAL phxx = (Pcurr[pos(ip1, j, k)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(im1, j, k)]) / dxs - sqrt2 * phi * phx * phx;
      REAL phyy = (Pcurr[pos(i, jp1, k)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(i, jm1, k)]) / dxs - sqrt2 * phi * phy * phy;
      REAL phzz = (Pcurr[pos(i, j, kp1)] - 2. * Pcurr[pos(i, j, k)] + Pcurr[pos(i, j, km1)]) / dxs - sqrt2 * phi * phz * phz;

      REAL sumnn = phxx + phyy + phzz;

      // Anisotropy
      // if(omp2>0.)

      if (fabs(omp2) > 0.) {
        // phase evaluation in the liquid
        // phase: 1 or -1 for solid, 0 for liquid
        // thershold omp2 >= 0.01
        // if (!pha)
        if (!pha && fabs(omp2) >= 0.001) {
          // sum all neighbors
          int sum = Phase[pos(ip1, j, k)] + Phase[pos(im1, j, k)] + Phase[pos(i, jp1, k)] + Phase[pos(i, jm1, k)] + Phase[pos(i, j, kp1)] + Phase[pos(i, j, km1)] + Phase[pos(ip1, jp1, k)] + Phase[pos(ip1, jm1, k)] + Phase[pos(im1, jp1, k)] + Phase[pos(im1, jm1, k)] + Phase[pos(i, jp1, kp1)] + Phase[pos(i, jp1, km1)] + Phase[pos(i, jm1, kp1)] + +Phase[pos(i, jm1, km1)] + Phase[pos(ip1, j, kp1)] + Phase[pos(im1, j, kp1)] + Phase[pos(ip1, j, km1)] + Phase[pos(im1, j, km1)] + Phase[pos(ip1, jp1, kp1)] + Phase[pos(ip1, jp1, km1)] + Phase[pos(ip1, jm1, kp1)] + Phase[pos(ip1, jm1, km1)] + Phase[pos(im1, jp1, kp1)] + Phase[pos(im1, jp1, km1)] + Phase[pos(im1, jm1, kp1)] + Phase[pos(im1, jm1, km1)];

          if (sum) {
            // update phase field
            Phase[pos(i0, j0, k0)] = pha = (sum > 0) - (sum < 0);
          }
        }

        // if it has Phase value (not in the liquid),
        if (pha) {
          // printf("Solid, %d\n",pha);

          pha = (1 + pha) / 2;

          // rotation matrix
          REAL ar11 = r11[pha];
          REAL ar12 = r12[pha];
          REAL ar13 = r13[pha];

          REAL ar21 = r21[pha];
          REAL ar22 = r22[pha];
          REAL ar23 = r23[pha];

          REAL ar31 = r31[pha];
          REAL ar32 = r32[pha];
          REAL ar33 = r33[pha];

          // mixed derivatives - sqrt2 * phi * (mulitiplication of the first derivatives )
          REAL phxy = (Pcurr[pos(ip1, jp1, k)] - Pcurr[pos(im1, jp1, k)] - Pcurr[pos(ip1, jm1, k)] + Pcurr[pos(im1, jm1, k)]) / dxs4 - sqrt2 * phi * phx * phy;
          REAL phyz = (Pcurr[pos(i, jp1, kp1)] - Pcurr[pos(i, jm1, kp1)] - Pcurr[pos(i, jp1, km1)] + Pcurr[pos(i, jm1, km1)]) / dxs4 - sqrt2 * phi * phy * phz;
          REAL phxz = (Pcurr[pos(ip1, j, kp1)] - Pcurr[pos(im1, j, kp1)] - Pcurr[pos(ip1, j, km1)] + Pcurr[pos(im1, j, km1)]) / dxs4 - sqrt2 * phi * phz * phx;

          // Rotate
          // Rotation of the first derivatives

          REAL dphix = phx * ar11 + phy * ar12 + phz * ar13;
          REAL dphiy = phx * ar21 + phy * ar22 + phz * ar23;
          REAL dphiz = phx * ar31 + phy * ar32 + phz * ar33;

          REAL pxy = dphix * dphiy;
          REAL pxz = dphix * dphiz;
          REAL pyz = dphiy * dphiz;

          REAL px2 = dphix * dphix;
          REAL py2 = dphiy * dphiy;
          REAL pz2 = dphiz * dphiz;
          REAL xnorm = px2 + py2 + pz2;

          REAL dphixx = ar11 * ar11 * phxx + ar12 * ar12 * phyy + ar13 * ar13 * phzz + 2. * (ar11 * ar12 * phxy + ar12 * ar13 * phyz + ar11 * ar13 * phxz);
          REAL dphiyy = ar21 * ar21 * phxx + ar22 * ar22 * phyy + ar23 * ar23 * phzz + 2. * (ar21 * ar22 * phxy + ar22 * ar23 * phyz + ar21 * ar23 * phxz);
          REAL dphizz = ar31 * ar31 * phxx + ar32 * ar32 * phyy + ar33 * ar33 * phzz + 2. * (ar31 * ar32 * phxy + ar32 * ar33 * phyz + ar31 * ar33 * phxz);

          sumnn = dphixx + dphiyy + dphizz;

          REAL xnorm2 = xnorm * xnorm;
          REAL xnorm3 = xnorm2 * xnorm;

          REAL dphixy = ar11 * (ar21 * phxx + ar22 * phxy + ar23 * phxz) + ar12 * (ar21 * phxy + ar22 * phyy + ar23 * phyz) + ar13 * (ar21 * phxz + ar22 * phyz + ar23 * phzz);

          REAL dphiyz = ar21 * (ar31 * phxx + ar32 * phxy + ar33 * phxz) + ar22 * (ar31 * phxy + ar32 * phyy + ar33 * phyz) + ar23 * (ar31 * phxz + ar32 * phyz + ar33 * phzz);

          REAL dphixz = ar31 * (ar11 * phxx + ar12 * phxy + ar13 * phxz) + ar32 * (ar11 * phxy + ar12 * phyy + ar13 * phyz) + ar33 * (ar11 * phxz + ar12 * phyz + ar13 * phzz);

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

        } else {
          // in the liquid
          anis = 0;
          xi = 1;
        }
      }

      // ----
      // dpdt
      // ----

#if (TIME0 > 0)
      lT = (iter * dt * Tau0_sec < TIME0) ? lT0 : (iter * dt * Tau0_sec > TIME1) ? lT1
                                                                                 : lT0 + (lT1 - lT0) / (TIME1 - TIME0) * (iter * dt * Tau0_sec - TIME0);
#endif

#if (Thermaltau == 0)
// no thermal effect
#if (OSC_Velocity == WITHOUT)
      REAL temp = ((AcmNx + i) * dx + xoffs - x0 - Vp * dt * iter) / lT;
#else
      REAL temp = ((AcmNx + i) * dx + xoffs - x0 - Lenpull) / lT;
#endif
#else
      // thermal drift effect: temp = temp + Tddzt * (1- exp( -1.*iter*dt/Tdtau ) )
      REAL temp = ((AcmNx + i) * dx + xoffs - x0 - Vp * dt * iter + Tddzt * (1. - exp(-1. * iter * dt / Tdtau))) / lT;
#endif
      REAL Tau = (temp > 1.) ? kcoeff : (1. - omk * temp);

      Pnext[pos(i0, j0, k0)] = psi + dt * (xi * xi * prefacs * sumnn + sqrt2 * phi - sqrt2 * omp2 * Lambda * (Ucurr[pos(i, j, k)] + temp) + anis * prefacs) / (xi * xi * prefacs) / (Tau);

      // Pnext[pos(i0,j0,k0)] =  psi ;

#if (WALLEFFECT == WSLOPE)
      if (k0 == 2) {
        Pnext[pos(i0, j0, 1)] = Pnext[pos(i0, j0, k0)] - WALLSLOPE * dx;
      } else if (k0 == (Nz - 1)) {
        Pnext[pos(i0, j0, Nz)] = Pnext[pos(i0, j0, k0)] - WALLSLOPE * dx;
      }
#endif

#if (WALLEFFECT == NzSLOPE)
      if (k0 == (Nz - 1)) {
        Pnext[pos(i0, j0, Nz)] = Pnext[pos(i0, j0, k0)] - WALLSLOPE * dx;
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
      Pnext[pos(i0, j0, k0)] += sqrt(2. * Fnoise * dt / dxs) * ran1;
      state[pos(i, j, k)] = localState;
#endif
#if (NOISE == FLAT)
      curandState localState;
      localState = state[pos(i, j, k)];
      REAL ran1 = curand_uniform_double(&localState);
      Pnext[pos(i0, j0, k0)] += Fnoise * sqrt(dt) * (ran1 - .5);
      state[pos(i, j, k)] = localState;

#endif
    }
#endif
  }
}

__global__ void Compute_U_BCin(REAL *Pcurr, REAL *Pnext,
                               REAL *Ucurr, REAL *Unext,
                               REAL *F,
                               Constants *Ct,
                               Variables *Vb,
                               int DevNx, int AcmNx) {
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

  if (i * j * k * (i - DevNx - 1) * (j - Ny - 1) * (k - Nz - 1)) //  i.e. (i!=0 && j!=0 && k!=0 && i!=Nx+1 && j!=Ny+1 && k!=Nz+1)
  {

    // position variables
    int i0 = i;
    int im1 = i - 1;
    int ip1 = i + 1;

    int j0 = j;
    int jm1 = j - 1;
    int jp1 = j + 1;

    int k0 = k;
    int km1 = k - 1;
    int kp1 = k + 1;

    // set maximum length limit for inner noflux boundaries
    int iGB = (int)(((1 + InnerBCL / 10.) * xint - xoffs) / dx);
    iGB = (iGB > 0.9 * Nx) ? 0.9 * Nx : iGB;

    if ((AcmNx + i) < iGB) {
      if (j == jGBy) {
        // Field(i,jGB,k) = Field(i,jGB-1,k) noflux boundary locates in between (jGB-1) and jGB
        j--;
        jm1--;
        jp1--;
      } else if (j == jGBy + 1) {
        // Field(i,jGB+1,k) = Field(i,jGB+2,k) noflux boundary locates in between (jGB+1) and (jGB+2)
        j++;
        jm1++;
        jp1++;
      }
    }
    // for inneer BC

#if (MIXd)
    REAL edgegrid = MIXd / dx_microns + Xtip; // [dx]
    int intedgegrid = (int)(edgegrid);        // []
    REAL r = (edgegrid - intedgegrid) * dx;   // change unit to [W]

    if ((AcmNx + i) > intedgegrid) {
      Unext[pos(i0, j0, k0)] = -1.;
    } else if ((AcmNx + i) < intedgegrid)
#endif
    {
      REAL u = Ucurr[pos(i, j, k)];
      REAL phi = F[pos(i, j, k)];
      REAL nphi = tanh(Pnext[pos(i, j, k)] / sqrt2);

      // ----
      // dUdt
      // ----
      Unext[pos(i0, j0, k0)] = (1. + omk * u) * (nphi - phi) + dt * D * 0.5 * ((2. - F[pos(ip1, j, k)] - phi) * (Ucurr[pos(ip1, j, k)] - u) - (2. - F[pos(im1, j, k)] - phi) * (u - Ucurr[pos(im1, j, k)]) + (2. - F[pos(i, jp1, k)] - phi) * (Ucurr[pos(i, jp1, k)] - u) - (2. - F[pos(i, jm1, k)] - phi) * (u - Ucurr[pos(i, jm1, k)]) + (2. - F[pos(i, j, kp1)] - phi) * (Ucurr[pos(i, j, kp1)] - u) - (2. - F[pos(i, j, km1)] - phi) * (u - Ucurr[pos(i, j, km1)])) / dx / dx;

#if (ANTITRAPPING)
      //      if(fabs(omp2)>1.e-10)
      if (omp2 > 0.) {
        // -----------------------------------------------------
        // Source term: div[(dphi/dt)gradphi/|gradphi|)]/sqrt(2)
        // -----------------------------------------------------
        REAL dpxr, dpyr, dnormr, dpxl, dnorml, dpyu, dnormu, dpyd, dnormd;
        REAL dpzr, dpzt, dpzb, dpxb, dpyb, dnormt, dnormb;

        // The unit gradient may be computed either by Grad(h_Psi) or Grad(h_Phi)
        dpxr = F[pos(ip1, j, k)] - phi;
        dpyr = (F[pos(ip1, jp1, k)] + F[pos(i, jp1, k)] - F[pos(ip1, jm1, k)] - F[pos(i, jm1, k)]) / 4.;
        dpzr = (F[pos(ip1, j, kp1)] + F[pos(i, j, kp1)] - F[pos(ip1, j, km1)] - F[pos(i, j, km1)]) / 4.;
        dnormr = sqrt(dpxr * dpxr + dpyr * dpyr + dpzr * dpzr);

        dpxl = phi - F[pos(im1, j, k)];
        dpyl = (F[pos(im1, jp1, k)] + F[pos(i, jp1, k)] - F[pos(im1, jm1, k)] - F[pos(i, jm1, k)]) / 4.;
        dpzl = (F[pos(im1, j, kp1)] + F[pos(i, j, kp1)] - F[pos(im1, j, km1)] - F[pos(i, j, km1)]) / 4.;
        dnorml = sqrt(dpxl * dpxl + dpyl * dpyl + dpzl * dpzl);

        dpyu = F[pos(i, jp1, k)] - phi;
        dpxu = (F[pos(ip1, jp1, k)] + F[pos(ip1, j, k)] - F[pos(im1, jp1, k)] - F[pos(im1, j, k)]) / 4.;
        dpzu = (F[pos(i, jp1, kp1)] + F[pos(i, j, kp1)] - F[pos(i, jp1, km1)] - F[pos(i, j, km1)]) / 4.;
        dnormu = sqrt(dpxu * dpxu + dpyu * dpyu + dpzu * dpzu);

        dpyd = phi - F[pos(i, jm1, k)];
        dpxd = (F[pos(ip1, jm1, k)] + F[pos(ip1, j, k)] - F[pos(im1, jm1, k)] - F[pos(im1, j, k)]) / 4.;
        dpzd = (F[pos(i, jm1, kp1)] + F[pos(i, j, kp1)] - F[pos(i, jm1, km1)] - F[pos(i, j, km1)]) / 4.;
        dnormd = sqrt(dpxd * dpxd + dpyd * dpyd + dpzd * dpzd);

        dpzt = F[pos(i, j, kp1)] - phi;
        dpxt = (F[pos(ip1, j, kp1)] + F[pos(ip1, j, k)] - F[pos(im1, j, kp1)] - F[pos(im1, j, k)]) / 4.;
        dpyt = (F[pos(i, jp1, kp1)] + F[pos(i, jp1, k)] - F[pos(i, jm1, kp1)] - F[pos(i, jm1, k)]) / 4.;
        dnormt = sqrt(dpxt * dpxt + dpyt * dpyt + dpzt * dpzt);

        dpzb = phi - F[pos(i, j, km1)];
        dpxb = (F[pos(ip1, j, km1)] + F[pos(ip1, j, k)] - F[pos(im1, j, km1)] - F[pos(im1, j, k)]) / 4.;
        dpyb = (F[pos(i, jp1, km1)] + F[pos(i, jp1, k)] - F[pos(i, jm1, km1)] - F[pos(i, jm1, k)]) / 4.;
        dnormb = sqrt(dpxb * dpxb + dpyb * dpyb + dpzb * dpzb);

        if ((dnormr * dnorml * dnormu * dnormd * dnormt * dnormb) > 0.) {
          REAL omp2dpsi = omp2 * (Pnext[pos(i, j, k)] - Pcurr[pos(i, j, k)]);

          djxr = 0.25 * ((1. - F[pos(ip1, j, k)] * F[pos(ip1, j, k)]) * (Pnext[pos(ip1, j, k)] - Pcurr[pos(ip1, j, k)]) * (1. + omk * Ucurr[pos(ip1, j, k)]) + omp2dpsi * (1. + omk * u)) * dpxr / dnormr;
          djxl = 0.25 * ((1. - F[pos(im1, j, k)] * F[pos(im1, j, k)]) * (Pnext[pos(im1, j, k)] - Pcurr[pos(im1, j, k)]) * (1. + omk * Ucurr[pos(im1, j, k)]) + omp2dpsi * (1. + omk * u)) * dpxl / dnorml;
          djyu = 0.25 * ((1. - F[pos(i, jp1, k)] * F[pos(i, jp1, k)]) * (Pnext[pos(i, jp1, k)] - Pcurr[pos(i, jp1, k)]) * (1. + omk * Ucurr[pos(i, jp1, k)]) + omp2dpsi * (1. + omk * u)) * dpyu / dnormu;
          djyd = 0.25 * ((1. - F[pos(i, jm1, k)] * F[pos(i, jm1, k)]) * (Pnext[pos(i, jm1, k)] - Pcurr[pos(i, jm1, k)]) * (1. + omk * Ucurr[pos(i, jm1, k)]) + omp2dpsi * (1. + omk * u)) * dpyd / dnormd;
          djzt = 0.25 * ((1. - F[pos(i, j, kp1)] * F[pos(i, j, kp1)]) * (Pnext[pos(i, j, kp1)] - Pcurr[pos(i, j, kp1)]) * (1. + omk * Ucurr[pos(i, j, kp1)]) + omp2dpsi * (1. + omk * u)) * dpzt / dnormt;
          djzb = 0.25 * ((1. - F[pos(i, j, km1)] * F[pos(i, j, km1)]) * (Pnext[pos(i, j, km1)] - Pcurr[pos(i, j, km1)]) * (1. + omk * Ucurr[pos(i, j, km1)]) + omp2dpsi * (1. + omk * u)) * dpzb / dnormb;

          // ----
          // dUdt
          // ----
          Unext[pos(i0, j0, k0)] += (djxr - djxl + djyu - djyd + djzt - djzb) / dx;
        }
      }
#endif

      // ----
      // dUdt
      // ----
      Unext[pos(i0, j0, k0)] /= (opk - omk * nphi);
      Unext[pos(i0, j0, k0)] += u;

#if (MIXd)
      if (i0 == (intedgegrid - 1)) {
        Unext[pos(i0 + 1, j0, k0)] = Unext[pos(i0, j0, k0)] + (-1. - Unext[pos(i0, j0, k0)]) / (dx + r) * dx;
      }
#endif
    }

    if (fabs(Unext[pos(i0, j0, k0)]) > 1.1) {
      printf("(iter=%d)Ucurr(%d,%d,%d)=%g, next=%g,Warning!!!!!!!!\n", iter, i0, j0, k0, Ucurr[pos(i0, j0, k0)], Unext[pos(i0, j0, k0)]);
    }
  }

  if ((!i) * (!j) * (!k)) // i.e. (i==0 && j==0 && k==0)
  {
    // iter++;
  }
}

#endif // end of #if(InnerBCL>0)

__global__ void PullBack(REAL *Pcurr, REAL *Pnext,
                         REAL *Ucurr, REAL *Unext,
                         signed char *Phase, signed char *TempPha,
                         Constants *Ct,
                         Variables *Vb,
                         int DevNx, int AcmNx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

#if (PBFRAME == LAB)
  int off = 1;
#endif
#if (PBFRAME == TIP)
  int off = (int)(xtip - xint / dx);
#endif

  if (AcmNx + i < Nx + 1 - off) {
    if (i < DevNx + 1) {
      Pnext[pos(i, j, k)] = Pcurr[pos(i + 1, j, k)];
      Unext[pos(i, j, k)] = Ucurr[pos(i + 1, j, k)];

      TempPha[pos(i, j, k)] = Phase[pos(i + 1, j, k)];
    }
  } else {
    Pnext[pos(i, j, k)] = Pcurr[pos(DevNx - off, j, k)] - (AcmNx + i - Nx + off) * dx;
    Unext[pos(i, j, k)] = -1.;
    TempPha[pos(i, j, k)] = 0;
  }

  if (i == 0 && j == 0 && k == 0) {
    xoffs += off * dx;
  }
}

__global__ void Boundary(REAL *Field, int DevNx, bool if_last) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  REAL ExtraDx;
  if (if_last) // If it is the last GPU for Psi boundary then ExtraDx = dx
  {
    ExtraDx = dx;
  } else {
    ExtraDx = 0.;
  }

  if (i * j * k * (i - DevNx - 1) * (j - Ny - 1) * (k - Nz - 1)) {

    // #if(WALLEFFECT==0)
    //  BC at (x=0.5) or (x=Nx+0.5)
    if (i == 1) {
      Field[pos(IMIN, j, k)] = Field[pos(1, j, k)];
      if (j == 1) {
        Field[pos(IMIN, JMIN, HEL(k))] = Field[pos(1, 1, k)];
        if (k == 1) {
          Field[pos(IMIN, JMIN, KMIN)] = Field[pos(1, 1, 1)];
        } // Vertex
        else if (k == Nz) {
          Field[pos(IMIN, JMIN, KMAX)] = Field[pos(1, 1, Nz)];
        } // Vertex
      } else if (j == Ny) {
        Field[pos(IMIN, JMAX, HEL(k))] = Field[pos(1, Ny, k)];
        if (k == 1) {
          Field[pos(IMIN, JMAX, KMIN)] = Field[pos(1, Ny, 1)];
        } // Vertex
        else if (k == Nz) {
          Field[pos(IMIN, JMAX, KMAX)] = Field[pos(1, Ny, Nz)];
        } // Vertex
      }
      if (k == 1) {
        Field[pos(IMIN, j, KMIN)] = Field[pos(1, j, 1)];
      } else if (k == Nz) {
        Field[pos(IMIN, SYM(j), KMAX)] = Field[pos(1, j, Nz)];
      }
    } else if (i == DevNx) {
      Field[pos(DEV_IMAX, j, k)] = Field[pos(DevNx, j, k)] - ExtraDx;
      if (j == 1) {
        Field[pos(DEV_IMAX, JMIN, HEL(k))] = Field[pos(DevNx, 1, k)] - ExtraDx;
        if (k == 1) {
          Field[pos(DEV_IMAX, JMIN, KMIN)] = Field[pos(DevNx, 1, 1)] - ExtraDx;
        } // Vertex
        else if (k == Nz) {
          Field[pos(DEV_IMAX, JMIN, KMAX)] = Field[pos(DevNx, 1, Nz)] - ExtraDx;
        } // Vertex
      } else if (j == Ny) {
        Field[pos(DEV_IMAX, JMAX, HEL(k))] = Field[pos(DevNx, Ny, k)] - ExtraDx;
        if (k == 1) {
          Field[pos(DEV_IMAX, JMAX, KMIN)] = Field[pos(DevNx, Ny, 1)] - ExtraDx;
        } // Vertex
        else if (k == Nz) {
          Field[pos(DEV_IMAX, JMAX, KMAX)] = Field[pos(DevNx, Ny, Nz)] - ExtraDx;
        } // Vertex
      }
      if (k == 1) {
        Field[pos(DEV_IMAX, j, KMIN)] = Field[pos(DevNx, j, 1)] - ExtraDx;
      } else if (k == Nz) {
        Field[pos(DEV_IMAX, SYM(j), KMAX)] = Field[pos(DevNx, j, Nz)] - ExtraDx;
      }
    }

    // BC at (y=0.5) or (y=Ny+0.5)
    if (j == 1) {
      Field[pos(i, JMIN, HEL(k))] = Field[pos(i, 1, k)];
      if (k == 1) {
        Field[pos(i, JMIN, HEL(KMIN))] = Field[pos(i, 1, 1)];
      } else if (k == Nz) {
        Field[pos(i, SYM(JMIN), HEL(KMAX))] = Field[pos(i, 1, Nz)];
      }
    } else if (j == Ny) {
      Field[pos(i, JMAX, HEL(k))] = Field[pos(i, Ny, k)];
      if (k == 1) {
        Field[pos(i, JMAX, HEL(KMIN))] = Field[pos(i, Ny, 1)];
      } else if (k == Nz) {
        Field[pos(i, SYM(JMAX), HEL(KMAX))] = Field[pos(i, Ny, Nz)];
      }
    }

    // BC at (z=0.5) or (z=Nz+0.5)
    if (k == 1) {
      Field[pos(i, j, KMIN)] = Field[pos(i, j, 1)];
    } else if (k == Nz) {
      Field[pos(i, SYM(j), KMAX)] = Field[pos(i, j, Nz)];
    }

    // #endif

    /*
    #if(WALLEFFECT>0)
        // BC at (x=0.5) or (x=Nx+0.5)
        if(i==1)
        {
            Field[pos(IMIN,j,k)]=Field[pos(1,j,k)];
            if(j==1)
            {
                Field[pos(IMIN,JMIN,HEL(k))]=Field[pos(1,1,k)];
            }
            else if(j==Ny)
            {
                Field[pos(IMIN,JMAX,HEL(k))]=Field[pos(1,Ny,k)];
            }
            if(k==1)
            {
                Field[pos(IMIN,j,KMIN)]=Field[pos(1,j,1)];
            }
            else if(k==Nz)
            {
                Field[pos(IMIN,SYM(j),KMAX)]=Field[pos(1,j,Nz)];
            }
        }
        else if(i==DevNx)
        {
            Field[pos(DEV_IMAX,j,k)]=Field[pos(DevNx,j,k)]-ExtraDx;  // keep the (-dx) term shown in single GPU version
            if(j==1)
            {
                Field[pos(DEV_IMAX,JMIN,HEL(k))]=Field[pos(DevNx,1,k)]-ExtraDx;
            }
            else if(j==Ny)
            {
                Field[pos(DEV_IMAX,JMAX,HEL(k))]=Field[pos(DevNx,Ny,k)]-ExtraDx;
            }
            if(k==1)
            {
                Field[pos(DEV_IMAX,j,KMIN)]=Field[pos(DevNx,j,1)]-ExtraDx;
            }
            else if(k==Nz)
            {
                Field[pos(DEV_IMAX,SYM(j),KMAX)]=Field[pos(DevNx,j,Nz)]-ExtraDx;
            }
        }

        // BC at (y=0.5) or (y=Ny+0.5)
        if(j==1)
        {
            Field[pos(i,JMIN,HEL(k))]=Field[pos(i,1,k)];
            if(k==1)
            {
                Field[pos(i,JMIN,HEL(KMIN))]=Field[pos(i,1,1)];
            }
            else if(k==Nz)
            {
                Field[pos(i,SYM(JMIN),HEL(KMAX))]=Field[pos(i,1,Nz)];
            }
        }
        else if(j==Ny)
        {
            Field[pos(i,JMAX,HEL(k))]=Field[pos(i,Ny,k)];
            if(k==1)
            {
                Field[pos(i,JMAX,HEL(KMIN))]=Field[pos(i,Ny,1)];
            }
            else if(k==Nz)
            {
                Field[pos(i,SYM(JMAX),HEL(KMAX))]=Field[pos(i,Ny,Nz)];
            }
        }

        // BC at (z=0.5) or (z=Nz+0.5)
        if(k==1)
        {
            Field[pos(i,j,KMIN)]=Field[pos(i,j,1)];
        }
        else if(k==Nz)
        {
            Field[pos(i,SYM(j),KMAX)]=Field[pos(i,j,Nz)];
        }
    #endif
    */
  }
}

__global__ void Boundary_Pha(signed char *Field, int DevNx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  if (i * j * k * (i - DevNx - 1) * (j - Ny - 1) * (k - Nz - 1)) {

    // #if(WALLEFFECT==0)
    //  BC at (x=0.5) or (x=Nx+0.5)
    if (i == 1) {
      Field[pos(IMIN, j, k)] = Field[pos(1, j, k)];
      if (j == 1) {
        Field[pos(IMIN, JMIN, HEL(k))] = Field[pos(1, 1, k)];
        if (k == 1) {
          Field[pos(IMIN, JMIN, KMIN)] = Field[pos(1, 1, 1)];
        } // Vertex
        else if (k == Nz) {
          Field[pos(IMIN, JMIN, KMAX)] = Field[pos(1, 1, Nz)];
        } // Vertex
      } else if (j == Ny) {
        Field[pos(IMIN, JMAX, HEL(k))] = Field[pos(1, Ny, k)];
        if (k == 1) {
          Field[pos(IMIN, JMAX, KMIN)] = Field[pos(1, Ny, 1)];
        } // Vertex
        else if (k == Nz) {
          Field[pos(IMIN, JMAX, KMAX)] = Field[pos(1, Ny, Nz)];
        } // Vertex
      }
      if (k == 1) {
        Field[pos(IMIN, j, KMIN)] = Field[pos(1, j, 1)];
      } else if (k == Nz) {
        Field[pos(IMIN, SYM(j), KMAX)] = Field[pos(1, j, Nz)];
      }
    } else if (i == DevNx) {
      Field[pos(DEV_IMAX, j, k)] = Field[pos(DevNx, j, k)];
      if (j == 1) {
        Field[pos(DEV_IMAX, JMIN, HEL(k))] = Field[pos(DevNx, 1, k)];
        if (k == 1) {
          Field[pos(DEV_IMAX, JMIN, KMIN)] = Field[pos(DevNx, 1, 1)];
        } // Vertex
        else if (k == Nz) {
          Field[pos(DEV_IMAX, JMIN, KMAX)] = Field[pos(DevNx, 1, Nz)];
        } // Vertex
      } else if (j == Ny) {
        Field[pos(DEV_IMAX, JMAX, HEL(k))] = Field[pos(DevNx, Ny, k)];
        if (k == 1) {
          Field[pos(DEV_IMAX, JMAX, KMIN)] = Field[pos(DevNx, Ny, 1)];
        } // Vertex
        else if (k == Nz) {
          Field[pos(DEV_IMAX, JMAX, KMAX)] = Field[pos(DevNx, Ny, Nz)];
        } // Vertex
      }
      if (k == 1) {
        Field[pos(DEV_IMAX, j, KMIN)] = Field[pos(DevNx, j, 1)];
      } else if (k == Nz) {
        Field[pos(DEV_IMAX, SYM(j), KMAX)] = Field[pos(DevNx, j, Nz)];
      }
    }

    // BC at (y=0.5) or (y=Ny+0.5)
    if (j == 1) {
      Field[pos(i, JMIN, HEL(k))] = Field[pos(i, 1, k)];
      if (k == 1) {
        Field[pos(i, JMIN, HEL(KMIN))] = Field[pos(i, 1, 1)];
      } else if (k == Nz) {
        Field[pos(i, SYM(JMIN), HEL(KMAX))] = Field[pos(i, 1, Nz)];
      }
    } else if (j == Ny) {
      Field[pos(i, JMAX, HEL(k))] = Field[pos(i, Ny, k)];
      if (k == 1) {
        Field[pos(i, JMAX, HEL(KMIN))] = Field[pos(i, Ny, 1)];
      } else if (k == Nz) {
        Field[pos(i, SYM(JMAX), HEL(KMAX))] = Field[pos(i, Ny, Nz)];
      }
    }

    // BC at (z=0.5) or (z=Nz+0.5)
    if (k == 1) {
      Field[pos(i, j, KMIN)] = Field[pos(i, j, 1)];
    } else if (k == Nz) {
      Field[pos(i, SYM(j), KMAX)] = Field[pos(i, j, Nz)];
    }

    // #endif

    /*
    #if(WALLEFFECT>0)
        // BC at (x=0.5) or (x=Nx+0.5)
        if(i==1)
        {
            Field[pos(IMIN,j,k)]=Field[pos(1,j,k)];
            if(j==1)
            {
                Field[pos(IMIN,JMIN,HEL(k))]=Field[pos(1,1,k)];
            }
            else if(j==Ny)
            {
                Field[pos(IMIN,JMAX,HEL(k))]=Field[pos(1,Ny,k)];
            }
            if(k==1)
            {
                Field[pos(IMIN,j,KMIN)]=Field[pos(1,j,1)];
            }
            else if(k==Nz)
            {
                Field[pos(IMIN,SYM(j),KMAX)]=Field[pos(1,j,Nz)];
            }
        }
        else if(i==DevNx)
        {
            Field[pos(DEV_IMAX,j,k)]=Field[pos(DevNx,j,k)];  // keep the (-dx) term shown in single GPU version
            if(j==1)
            {
                Field[pos(DEV_IMAX,JMIN,HEL(k))]=Field[pos(DevNx,1,k)];
            }
            else if(j==Ny)
            {
                Field[pos(DEV_IMAX,JMAX,HEL(k))]=Field[pos(DevNx,Ny,k)];
            }
            if(k==1)
            {
                Field[pos(DEV_IMAX,j,KMIN)]=Field[pos(DevNx,j,1)];
            }
            else if(k==Nz)
            {
                Field[pos(DEV_IMAX,SYM(j),KMAX)]=Field[pos(DevNx,j,Nz)];
            }
        }

        // BC at (y=0.5) or (y=Ny+0.5)
        if(j==1)
        {
            Field[pos(i,JMIN,HEL(k))]=Field[pos(i,1,k)];
            if(k==1)
            {
                Field[pos(i,JMIN,HEL(KMIN))]=Field[pos(i,1,1)];
            }
            else if(k==Nz)
            {
                Field[pos(i,SYM(JMIN),HEL(KMAX))]=Field[pos(i,1,Nz)];
            }
        }
        else if(j==Ny)
        {
            Field[pos(i,JMAX,HEL(k))]=Field[pos(i,Ny,k)];
            if(k==1)
            {
                Field[pos(i,JMAX,HEL(KMIN))]=Field[pos(i,Ny,1)];
            }
            else if(k==Nz)
            {
                Field[pos(i,SYM(JMAX),HEL(KMAX))]=Field[pos(i,Ny,Nz)];
            }
        }

        // BC at (z=0.5) or (z=Nz+0.5)
        if(k==1)
        {
            Field[pos(i,j,KMIN)]=Field[pos(i,j,1)];
        }
        else if(k==Nz)
        {
            Field[pos(i,SYM(j),KMAX)]=Field[pos(i,j,Nz)];
        }
    #endif
    */
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Processing //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

__global__ void GetXsl_YZ(REAL *P, signed char *Phase,
                          REAL *Xmax1, REAL *Xmax2,
                          int DevNx, int AcmNx) {
#define pos2D(y, z) (WIDTH * (y) + (z))
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;

  REAL x, maxX = 0.;
  REAL maxX1 = 0.;
  REAL maxX2 = 0.;
  signed char pha = 0;

  for (int i = 1; i <= DevNx; i++) {
    pha = Phase[pos(i, j, k)];

    if (P[pos(i, j, k)] * P[pos(i + 1, j, k)] < 0.) {
      x = 1. * (AcmNx + i) + P[pos(i, j, k)] / (P[pos(i, j, k)] - P[pos(i + 1, j, k)]);

      if ((x > maxX) && (pha == -1)) {
        // for grain 1
        maxX1 = x;
      } else if ((x > maxX) && (pha == 1)) {
        // for grain 2
        maxX2 = x;
      }
    }
  }
  Xmax1[pos2D(j, k)] = maxX1;
  Xmax2[pos2D(j, k)] = maxX2;
}

__global__ void GetXtip(REAL *Xmax1, REAL *Xmax2, Variables *Vb) {
#define pos2D(y, z) (WIDTH * (y) + (z))
  // -------------------
  // Find xtip,ytip,ztip
  // -------------------
  REAL x1, maxX1 = 0., maxY1 = 0., maxZ1 = 0.;
  REAL x2, maxX2 = 0., maxY2 = 0., maxZ2 = 0.;

  for (int j = 1; j <= Ny; j++) {
    for (int k = 1; k <= Nz; k++) {
      x1 = Xmax1[pos2D(j, k)];
      x2 = Xmax2[pos2D(j, k)];

      if (x1 > maxX1) {
        maxX1 = x1;
        maxY1 = j;
        maxZ1 = k;
      }
      if (x2 > maxX2) {
        maxX2 = x2;
        maxY2 = j;
        maxZ2 = k;
      }
    }
  }

  xtip1 = maxX1;
  ytip1 = maxY1;
  ztip1 = maxZ1;

  xtip2 = maxX2;
  ytip2 = maxY2;
  ztip2 = maxZ2;

  /*
  // Do this calculation in "int FieldsEvolution::CalculateTip()"
  if (xtip1>xtip2)
  {
      xtip=maxX1;
      ytip=maxY1;
      ztip=maxZ1;
  }
  else
  {
      xtip=maxX2;
      ytip=maxY2;
      ztip=maxZ2;
  }
  */
}