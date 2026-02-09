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



#include "io-functions.h"

// C++ & CUDA headers
#include "math.h"
#include <fstream>
#include <iostream>
#include <sstream>
// Project headers
#include "gpu-manager.h"
#include "macro.h"
#include "param-manager.h"

#define IMIN 0
#define IMAX Nx + 1

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

float FloatSwap(float f);

int IO_InitfromFile_DAT(REAL *P, REAL *U, signed char *Phase, REAL *Temperature, Constants Ct) {
  char INIT_FILE_P[LENMAX];
  char INIT_FILE_C[LENMAX];
  char INIT_FILE_Phase[LENMAX];
  sprintf(INIT_FILE_P, "Final_Psi.dat");
  sprintf(INIT_FILE_C, "Final_Compo.dat");
  sprintf(INIT_FILE_Phase, "Final_Phase.dat");

  FILE *InFile_P;
  FILE *InFile_C;
  FILE *InFile_Phase;
  InFile_P = fopen(INIT_FILE_P, "r");
  InFile_C = fopen(INIT_FILE_C, "r");
  InFile_Phase = fopen(INIT_FILE_Phase, "r");

  for (int i = 0; i < ((Nx + 2) * (Ny + 2) * (Nz + 2)); i++) {
    REAL d;
    fread((char *)&d, sizeof(REAL), 1, InFile_P);
    P[i] = d;

    REAL d1;
    fread((char *)&d1, sizeof(REAL), 1, InFile_C);
    U[i] = d1;

    signed char d2;
    fread(&d2, sizeof(signed char), 1, InFile_Phase);
    Phase[i] = d2;
  }
  fclose(InFile_P);
  fclose(InFile_C);
  fclose(InFile_Phase);

  return 0;
}

int IO_InitFromFile_VTK(REAL *P, REAL *U, signed char *Phase, REAL *Temperature, Constants Ct, Variables *Vb) {
#define pos_old(x, y, z) ((Ny_old + 2) * (Nz_old + 2) * (x) + (Nz_old + 2) * (y) + (z))
  char buff[15];
  int Nx_old, Ny_old, Nz_old, j0 = 1, k0 = 1;
  REAL *Pold, *Cold, Del0;

  char INIT_FILE_P[LENMAX];
  char INIT_FILE_C[LENMAX];
  sprintf(INIT_FILE_P, "%sPsi_%s.Final.vtk", INIT_FILE_DIR, INIT_FILE);
  sprintf(INIT_FILE_C, "%sCompo_%s.Final.vtk", INIT_FILE_DIR, INIT_FILE);

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

    Vb->x0 = 1. - Del0; // At this point (1-Delta0) is stored into (*h_Parameters).x0

    // Vb->Delta = Del0;
    //  ======================
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

  // =============================================
  // Get Phase values ==> Phase[pos(i,j,k)]
  // =============================================

  // initialize the Phase[]
  for (int k = 0; k < Nz + 2; k++) {
    for (int j = 0; j < Ny + 2; j++) {
      for (int i = 0; i < Nx + 2; i++) {

        Phase[pos(i, j, k)] = 1; // set default Phase value to 1, if no input phase file
      }
    }
  }
  //

  char INIT_FILE_Phase[LENMAX];
  sprintf(INIT_FILE_Phase, "%sGB_%s.Final.vtk", INIT_FILE_DIR, INIT_FILE);

  REAL tempha = 0.;

  std::ifstream InFilePha(INIT_FILE_Phase);
  if (InFilePha.is_open()) {
    getline(InFilePha, line); //	"# vtk DataFile Version 3.0\n"
    getline(InFilePha, line); //	"Title\n"
    getline(InFilePha, line); //	"ASCII\n"
    getline(InFilePha, line); //	"DATASET STRUCTURED_POINTS\n"
    getline(InFilePha, line); //	"DIMENSIONS %d %d %d\n"
    getline(InFilePha, line); //	"ASPECT_RATIO %f %f %f\n"
    getline(InFilePha, line); //	"ORIGIN 0 0 0\n"
    getline(InFilePha, line); //	"POINT_DATA %d\n"
    getline(InFilePha, line); //	"SCALARS Psi double 1\n"
    getline(InFilePha, line); //	"LOOKUP_TABLE default\n"
    getline(InFilePha, line);

    std::istringstream Valuespha(line);
    for (int k = 0; k < Nz_old + 2; k++) {
      for (int j = 0; j < Ny_old + 2; j++) {
        for (int i = 0; i < Nx_old + 2; i++) {

          Valuespha >> tempha;
          Phase[pos(i, j, k)] = (signed char)(tempha);

          /* if ( (Phase[pos(i,j,k)]!=-1)&&(Phase[pos(i,j,k)]!=0)&&(Phase[pos(i,j,k)]!=1) )
          {
              printf("inside read GB: (%d,%d,%d) pha=%d, tempha=%d\n",i,j,k,Phase[pos(i,j,k)],(signed char)(tempha) );

          } */
        }
      }
    }
    InFilePha.close();
  } else
    std::cout << "Unable to open file " << INIT_FILE_Phase << std::endl;

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
        phi = tanh(P[pos(inew, jnew, knew)] / Ct.sqrt2);
        U[pos(inew, jnew, knew)] = (2. * c - Ct.opk + Ct.omk * phi) / Ct.omk / (Ct.opk - Ct.omk * phi);
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
          // Phase[pos(i,j,k)]=0; // phase in liquid
        }
      }
    }
  }
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
            // Phase[pos(i,jnew,knew)]=Phase[pos(i,j,k)];
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
          for (int k = 1; k <= Nz; k++) {
            knew = k;
            P[pos(i, jnew, knew)] = P[pos(i, j, k)];
            U[pos(i, jnew, knew)] = U[pos(i, j, k)];
            // Phase[pos(i,jnew,knew)]=Phase[pos(i,j,k)];
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
      Phase[pos(IMIN, j, k)] = Phase[pos(1, j, k)];

      P[pos(IMAX, j, k)] = P[pos(Nx, j, k)];
      U[pos(IMAX, j, k)] = U[pos(Nx, j, k)];
      Phase[pos(IMAX, j, k)] = Phase[pos(Nx, j, k)];
    }
  }
  for (int i = 1; i <= Nx; i++) {
    for (int k = 1; k <= Nz; k++) {
      P[pos(i, JMIN, HEL(k))] = P[pos(i, 1, k)];
      U[pos(i, JMIN, HEL(k))] = U[pos(i, 1, k)];
      Phase[pos(i, JMIN, HEL(k))] = Phase[pos(i, 1, k)];

      P[pos(i, JMAX, HEL(k))] = P[pos(i, Ny, k)];
      U[pos(i, JMAX, HEL(k))] = U[pos(i, Ny, k)];
      Phase[pos(i, JMAX, HEL(k))] = Phase[pos(i, Ny, k)];
    }
  }
  for (int i = 1; i <= Nx; i++) {
    for (int j = 1; j <= Ny; j++) {
      P[pos(i, j, KMIN)] = P[pos(i, j, 1)];
      U[pos(i, j, KMIN)] = U[pos(i, j, 1)];
      Phase[pos(i, j, KMIN)] = Phase[pos(i, j, 1)];

      P[pos(i, SYM(j), KMAX)] = P[pos(i, j, Nz)];
      U[pos(i, SYM(j), KMAX)] = U[pos(i, j, Nz)];
      Phase[pos(i, SYM(j), KMAX)] = Phase[pos(i, j, Nz)];
    }
  }
  for (int i = 1; i <= Nx; i++) {
    P[pos(i, JMIN, HEL(KMIN))] = P[pos(i, 1, 1)];
    U[pos(i, JMIN, HEL(KMIN))] = U[pos(i, 1, 1)];
    Phase[pos(i, JMIN, HEL(KMIN))] = Phase[pos(i, 1, 1)];

    P[pos(i, JMAX, HEL(KMIN))] = P[pos(i, Ny, 1)];
    U[pos(i, JMAX, HEL(KMIN))] = U[pos(i, Ny, 1)];
    Phase[pos(i, JMAX, HEL(KMIN))] = Phase[pos(i, Ny, 1)];

    P[pos(i, SYM(JMIN), HEL(KMAX))] = P[pos(i, 1, Nz)];
    U[pos(i, SYM(JMIN), HEL(KMAX))] = U[pos(i, 1, Nz)];
    Phase[pos(i, SYM(JMIN), HEL(KMAX))] = Phase[pos(i, 1, Nz)];

    P[pos(i, SYM(JMAX), HEL(KMAX))] = P[pos(i, Ny, Nz)];
    U[pos(i, SYM(JMAX), HEL(KMAX))] = U[pos(i, Ny, Nz)];
    Phase[pos(i, SYM(JMAX), HEL(KMAX))] = Phase[pos(i, Ny, Nz)];
  }
  for (int j = 1; j <= Ny; j++) {
    P[pos(IMIN, j, KMIN)] = P[pos(1, j, 1)];
    U[pos(IMIN, j, KMIN)] = U[pos(1, j, 1)];
    Phase[pos(IMIN, j, KMIN)] = Phase[pos(1, j, 1)];

    P[pos(IMAX, j, KMIN)] = P[pos(Nx, j, 1)];
    U[pos(IMAX, j, KMIN)] = U[pos(Nx, j, 1)];
    Phase[pos(IMAX, j, KMIN)] = Phase[pos(Nx, j, 1)];

    P[pos(IMIN, SYM(j), KMAX)] = P[pos(1, j, Nz)];
    U[pos(IMIN, SYM(j), KMAX)] = U[pos(1, j, Nz)];
    Phase[pos(IMIN, SYM(j), KMAX)] = Phase[pos(1, j, Nz)];

    P[pos(IMAX, SYM(j), KMAX)] = P[pos(Nx, j, Nz)];
    U[pos(IMAX, SYM(j), KMAX)] = U[pos(Nx, j, Nz)];
    Phase[pos(IMAX, SYM(j), KMAX)] = Phase[pos(Nx, j, Nz)];
  }
  for (int k = 1; k <= Nz; k++) {
    P[pos(IMIN, JMIN, HEL(k))] = P[pos(1, 1, k)];
    U[pos(IMIN, JMIN, HEL(k))] = U[pos(1, 1, k)];
    Phase[pos(IMIN, JMIN, HEL(k))] = Phase[pos(1, 1, k)];

    P[pos(IMIN, JMAX, HEL(k))] = P[pos(1, Ny, k)];
    U[pos(IMIN, JMAX, HEL(k))] = U[pos(1, Ny, k)];
    Phase[pos(IMIN, JMAX, HEL(k))] = Phase[pos(1, Ny, k)];

    P[pos(IMAX, JMIN, HEL(k))] = P[pos(Nx, 1, k)];
    U[pos(IMAX, JMIN, HEL(k))] = U[pos(Nx, 1, k)];
    Phase[pos(IMAX, JMIN, HEL(k))] = Phase[pos(Nx, 1, k)];

    P[pos(IMAX, JMAX, HEL(k))] = P[pos(Nx, Ny, k)];
    U[pos(IMAX, JMAX, HEL(k))] = U[pos(Nx, Ny, k)];
    Phase[pos(IMAX, JMAX, HEL(k))] = Phase[pos(Nx, Ny, k)];
  }

  P[pos(IMIN, JMIN, HEL(KMIN))] = P[pos(1, 1, 1)];
  U[pos(IMIN, JMIN, HEL(KMIN))] = U[pos(1, 1, 1)];
  Phase[pos(IMIN, JMIN, HEL(KMIN))] = Phase[pos(1, 1, 1)];

  P[pos(IMIN, JMAX, HEL(KMIN))] = P[pos(1, Ny, 1)];
  U[pos(IMIN, JMAX, HEL(KMIN))] = U[pos(1, Ny, 1)];
  Phase[pos(IMIN, JMAX, HEL(KMIN))] = Phase[pos(1, Ny, 1)];

  P[pos(IMAX, JMIN, HEL(KMIN))] = P[pos(Nx, 1, 1)];
  U[pos(IMAX, JMIN, HEL(KMIN))] = U[pos(Nx, 1, 1)];
  Phase[pos(IMAX, JMIN, HEL(KMIN))] = Phase[pos(Nx, 1, 1)];

  P[pos(IMAX, JMAX, HEL(KMIN))] = P[pos(Nx, Ny, 1)];
  U[pos(IMAX, JMAX, HEL(KMIN))] = U[pos(Nx, Ny, 1)];
  Phase[pos(IMAX, JMAX, HEL(KMIN))] = Phase[pos(Nx, Ny, 1)];

  P[pos(IMIN, SYM(JMIN), HEL(KMAX))] = P[pos(1, 1, Nz)];
  U[pos(IMIN, SYM(JMIN), HEL(KMAX))] = U[pos(1, 1, Nz)];
  Phase[pos(IMIN, SYM(JMIN), HEL(KMAX))] = Phase[pos(1, 1, Nz)];

  P[pos(IMIN, SYM(JMAX), HEL(KMAX))] = P[pos(1, Ny, Nz)];
  U[pos(IMIN, SYM(JMAX), HEL(KMAX))] = U[pos(1, Ny, Nz)];
  Phase[pos(IMIN, SYM(JMAX), HEL(KMAX))] = Phase[pos(1, Ny, Nz)];

  P[pos(IMAX, SYM(JMIN), HEL(KMAX))] = P[pos(Nx, 1, Nz)];
  U[pos(IMAX, SYM(JMIN), HEL(KMAX))] = U[pos(Nx, 1, Nz)];
  Phase[pos(IMAX, SYM(JMIN), HEL(KMAX))] = Phase[pos(Nx, 1, Nz)];

  P[pos(IMAX, SYM(JMAX), HEL(KMAX))] = P[pos(Nx, Ny, Nz)];
  U[pos(IMAX, SYM(JMAX), HEL(KMAX))] = U[pos(Nx, Ny, Nz)];
  Phase[pos(IMAX, SYM(JMAX), HEL(KMAX))] = Phase[pos(Nx, Ny, Nz)];

  free(Pold);
  free(Cold);

  return 0;
}

int IO_InitfromBreak(REAL *P, REAL *U, signed char *Phase, REAL *Temperature, Constants Ct) {
  char INIT_FILE_P[LENMAX];
  char INIT_FILE_C[LENMAX];
  char INIT_FILE_Phase[LENMAX];
  sprintf(INIT_FILE_P, "Break_P.dat");
  sprintf(INIT_FILE_C, "Break_U.dat");
  sprintf(INIT_FILE_Phase, "Break_Phase.dat");

  FILE *InFile_P;
  FILE *InFile_C;
  FILE *InFile_Phase;
  InFile_P = fopen(INIT_FILE_P, "r");
  InFile_C = fopen(INIT_FILE_C, "r");
  InFile_Phase = fopen(INIT_FILE_Phase, "r");

  for (int i = 0; i < ((Nx + 2) * (Ny + 2) * (Nz + 2)); i++) {
    REAL d;
    fread((char *)&d, sizeof(REAL), 1, InFile_P);
    P[i] = d;

    REAL d1;
    fread((char *)&d1, sizeof(REAL), 1, InFile_C);
    U[i] = d1;

    signed char d2;
    fread(&d2, sizeof(signed char), 1, InFile_Phase);
    Phase[i] = d2;
  }
  fclose(InFile_P);
  fclose(InFile_C);
  fclose(InFile_Phase);

  return 0;
}

int InitIndexFile(Constants *Ct, Variables *Vb) {
  FILE *IndxT = fopen(Ct->IndexFileName, "w");
  fprintf(IndxT, "Index \tTime[s] \txoffs[dx] \n");
  fprintf(IndxT, "%d \t%g \t%g \n", 0, 0., Vb->xoffs / dx);
  fclose(IndxT);

  return 0;
}

int OutputIndexFile(Constants *Ct, Variables *Vb) {
  FILE *IndxT = fopen(Ct->IndexFileName, "a");
#if (NOUTFIELDS < 0)
  fprintf(IndxT, "%d \t%.2f \t%g \n", Vb->numoutput_f, Vb->iter * Ct->dt * Ct->Tau0_sec, Vb->xoffs / dx);
#elif (NOUTFIELDS > 0)
  fprintf(IndxT, "%d \t%.2f \t%g \n", int(Vb->iter * Ct->dt * Ct->Tau0_sec), Vb->iter * Ct->dt * Ct->Tau0_sec, Vb->xoffs / dx);
#endif
  fclose(IndxT);

  return 0;
}

int InitTipFile(char *TipFileName) {
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
  fprintf(TipF, "(18)V_{OSC}[micron/s] \t"); // 18
  fprintf(TipF, "(19)V_{tot}[micron/s] \t"); // 19
#endif
  fprintf(TipF, "\n");
  fclose(TipF);

  return 0;
}

int OutputTipFile(Constants *Ct, Variables *Vb) {
  FILE *TipF = fopen(Ct->TipFileName, "a");
  fprintf(TipF, "%g \t", Vb->iter * Ct->dt * Ct->Tau0_sec);       //  1:	Time		[s]
  fprintf(TipF, "%g \t", Vb->Delta);                              //  2:	Delta		[-]
  fprintf(TipF, "%g \t", Vb->Omega);                              //  3:	Omega		[-]
  fprintf(TipF, "%g \t", Vb->Vel * Ct->W_microns / Ct->Tau0_sec); //  4:	Vel_Tip		[micron/s]
  fprintf(TipF, "%g \t", Vb->xtip);                               //  5:	xtip
  fprintf(TipF, "%g \t", Vb->ytip);                               //  6:	ytip
  fprintf(TipF, "%g \t", Vb->ztip);                               //  7:	ztip
  fprintf(TipF, "%g \t", Vb->RadY * Ct->W_microns);               //  8:	R/y
  fprintf(TipF, "%g \t", Vb->RadZ * Ct->W_microns);               //  9:	R/z
  fprintf(TipF, "%g \t", Vb->xtip / Nx);                          // 10:	xtip/Nx
  fprintf(TipF, "%g \t", Vb->Delta * Vb->lT * Ct->W_microns);     // 11:	xtip-xL
  fprintf(TipF, "%g \t", Vb->xoffs);                              // 12:	xoffs
  REAL Grad = GRAD0;
#if (TIME0 > 0)
  Grad = (Vb->iter * Ct->dt * Ct->Tau0_sec < TIME0) ? GRAD0 : (Vb->iter * Ct->dt * Ct->Tau0_sec > TIME1) ?
                                                                                                         // GRAD1 : (GRAD0+(GRAD1-GRAD0)/(TIME1-TIME0)*(Vb->iter*Ct->dt*Ct->Tau0_sec-TIME0)) ;	// linearly varying G
                                                              GRAD1
                                                                                                         : 1. / (1. / GRAD0 + (1. / GRAD1 - 1. / GRAD0) / (TIME1 - TIME0) * (Vb->iter * Ct->dt * Ct->Tau0_sec - TIME0)); // linearly varying lT
#endif
  fprintf(TipF, "%g \t", Grad); // 13:	GradT		[K/m]

#if (OSC_Velocity)
  fprintf(TipF, "%g \t", Vb->OSCVamp * Ct->W_microns / Ct->Tau0_sec);            // 18:	oscillating velocity [micron/s]
  fprintf(TipF, "%g \t", (Ct->Vp + Vb->OSCVamp) * Ct->W_microns / Ct->Tau0_sec); // 19:	total pulling velocity [micron/s]
#endif
  fprintf(TipF, "\n");
  fclose(TipF);

  return 0;
}

int OutputGrainTip(Constants *Ct, Variables *Vb) {

  REAL tcurr = Vb->iter * Ct->dt * Ct->Tau0_sec;

  // make a file for grain tips informations
  FILE *OutFileGBTips = fopen(Vb->FileNameGBTips, "a");
  fprintf(OutFileGBTips, "%g \t", tcurr);      // 1
  fprintf(OutFileGBTips, "%g \t", Vb->xtip1);  // 2
  fprintf(OutFileGBTips, "%g \t", Vb->Delta1); // 3
  fprintf(OutFileGBTips, "%g \t", Vb->Omega1); // 4
  fprintf(OutFileGBTips, "%g \t", Vb->xtip2);  // 5
  fprintf(OutFileGBTips, "%g \t", Vb->Delta2); // 6
  fprintf(OutFileGBTips, "%g \t", Vb->Omega2); // 7
  fprintf(OutFileGBTips, "\n");
  fclose(OutFileGBTips);

  return 0;
}

void OutputParameters(GpuManager_Param dv, Constants Ct, Variables Vb) {
  // ------------------------------------------------------
  REAL a1 = 5. * sqrt(2.) / 8.;
  REAL a2 = 47. / 75.;
  // ------------------------------------------------------
  REAL mc0 = LIQSLOPE * COMPOSITION;   // |Liquidus slope|*Nominal composition, K
  REAL DT0 = mc0 / PARTITION - mc0;    // Solidification range, K
  REAL d0 = GIBBSTHOMSON / DT0 * 1.e6; // Capillarity length @ T0, microns
  REAL lTherm0 = DT0 / GRAD0 * 1.e6; // Thermal length, microns

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
  //		Output to file
  // ----------------------------------------------------
  char FileName[LENMAX];
  sprintf(FileName, "%s.Param.txt", Ct.OutputPrefix);
  FILE *OutFile;
  OutFile = fopen(FileName, "w");

  fprintf(OutFile, "----------------------------------------");
  fprintf(OutFile, "\n         SIMULATION PARAMETERS");
  fprintf(OutFile, "\n----------------------------------------\n");
  fprintf(OutFile, " Vp    = %g microns/s\n", VELOCITY);
#if (TIME0 > 0)
  fprintf(OutFile, " G     = %g to %g K/m\n", GRAD0, GRAD1);
  fprintf(OutFile, " (at t = %d to %d s)\n", TIME0, TIME1);
#else
  fprintf(OutFile, " G     = %g K/m\n", (Thot - Tcold) / Lsample * 1.e6);
#endif
  fprintf(OutFile, "\n");
  fprintf(OutFile, " m     = %g K/UnitC\n", LIQSLOPE);
  fprintf(OutFile, " c0    = %g UnitC\n", COMPOSITION);
  fprintf(OutFile, " D     = %g microns^2/s\n", DIFFUSION);
  fprintf(OutFile, " Gamma = %g Km\n", GIBBSTHOMSON);
  fprintf(OutFile, " k     = %g\n", PARTITION);
  fprintf(OutFile, " Eps4  = %g\n", ANISOTROPY);

  fprintf(OutFile, " Anisotropy rotation\n");
  fprintf(OutFile, " alpha1 = %g, \t alpha2 = %g degrees\n", AngleA1, AngleA2);
  fprintf(OutFile, " beta1  = %g, \t beta2  = %g degrees\n", AngleB1, AngleB2);
  fprintf(OutFile, " gamma1 = %g, \t gamma2 = %g degrees\n", AngleC1, AngleC2);
  fprintf(OutFile, "\n");

#if (TIME0 > 0)
  fprintf(OutFile, " lT    = %g to %g microns\n", lTherm0, lTherm1);
  fprintf(OutFile, " (at t = %d to %d s)\n", TIME0, TIME1);
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
  fprintf(OutFile, " from t             = %g s\n", OSC_t0);
  fprintf(OutFile, " Oscillation Amp    = %g [microns/s]\n", OSC_Vamp);
  fprintf(OutFile, "                    = %g \n", OSC_Vamp * d0 / DIFFUSION * a1 * a2 * E * E);

  if (OSC_Velocity == SINOSC || OSC_Velocity == STEPLIKE) {
    fprintf(OutFile, " Oscillation period = %g [s]\n", OSC_Period);
  }
#endif

  fprintf(OutFile, "-------------- Sample info --------------\n");
  fprintf(OutFile, " Tcold            = %g K \n", Tcold);
  fprintf(OutFile, " Thot             = %g K \n", Thot);
  fprintf(OutFile, " Sample length    = %g microns \n", Lsample);
  fprintf(OutFile, " Gradient         = %g K/cm \n", (Thot - Tcold) / Lsample * 1.e4);
  //fprintf(OutFile, " Tl0              = %g K \n", TL0);
  //fprintf(OutFile, " Ts0              = %g K \n", TL0 - DT0);
  //fprintf(OutFile, " Dthermal         = %g microns^2/s \n", Dtherm);
  fprintf(OutFile, "----------------------------------------\n");

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
  fprintf(OutFile, " Total time = %g \n", Vb.niter * Ct.dt);
  fprintf(OutFile, "--------- INITIAL CONDITIONS -----------\n");
#if (!INITfromFILE)
  fprintf(OutFile, "Delta_0    = %g\n", UNDERCOOL_0);
  fprintf(OutFile, "x_int_0    = %g W\n", POSITION_0 / W_microns);
  fprintf(OutFile, "           = %g dx\n", POSITION_0 / W_microns / dx);
  fprintf(OutFile, "           = %g microns\n", POSITION_0);
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
  fprintf(OutFile, " dt = %g\n", Ct.dt);
  fprintf(OutFile, "    = %g seconds\n", Ct.dt * Tau0_sec);


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
  for (int i = 0; i < NGPU; i++) // BLOCK_SIZE
  {
    int nbx = (dv.nx[i] + 2.) / dv.BLOCK_X[i];
    int nby = (Ny + 2.) / dv.BLOCK_Y[i];
    int nbz = (Nz + 2.) / dv.BLOCK_Z[i];

    fprintf(OutFile, "***** Device %d *****\n", dv.DEVICES[i]);
    fprintf(OutFile, "DEV_Nx + 2 is %d\n", dv.nx[i] + 2);
    fprintf(OutFile, "Block size /X = %d\n", dv.BLOCK_X[i]);
    fprintf(OutFile, "Block size /Y = %d\n", dv.BLOCK_Y[i]);
    fprintf(OutFile, "Block size /Z = %d\n", dv.BLOCK_Z[i]);
    fprintf(OutFile, "Thread/Block  = %d\n", dv.BLOCK_X[i] * dv.BLOCK_Y[i] * dv.BLOCK_Z[i]);
    fprintf(OutFile, "Number of Blocks /X = %d\n", nbx);
    fprintf(OutFile, "Number of Blocks /Y = %d\n", nby);
    fprintf(OutFile, "Number of Blocks /Z = %d\n", nbz);
    fprintf(OutFile, "Number of Blocks    = %d\n", nbx * nby * nbz);
  }
  fprintf(OutFile, "----------------------------------------\n");
  fprintf(OutFile, " Time step input dt = %g\n", dt0);
  if (dt0 > dx * dx / 6. / D)
    fprintf(OutFile, "                 dt > dx^2/(6*D) ...\n", Ct.dt);
  if (Ct.dt < dt0)
    fprintf(OutFile, " Adjust =>       dt = %g\n", Ct.dt);
  fprintf(OutFile, " Number of iterations: %d\n", Vb.niter);
  fprintf(OutFile, " Output movie files (compressed) every %d iterations\n", Vb.IterOutFields);
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

void WriteParam(Variables *Vb, int ifFinal) // ifFinal: if read Vb from final results
{
  FILE *f;
  if (ifFinal) {
    f = fopen("Final_Variables.dat", "w");
  } else {
    f = fopen("Break_Variables.dat", "w");
  }
  fwrite((void *)Vb, sizeof(Variables), 1, f);
  fclose(f);
}

void ReadParam(Variables *Vb, int ifFinal) // ifFinal: if read Vb from final results
{
  FILE *f;
  if (ifFinal) {
    f = fopen("Final_Variables.dat", "r");
  } else {
    f = fopen("Break_Variables.dat", "r");
  }
  fread((void *)Vb, sizeof(Variables), 1, f);
  fclose(f);
}

void WriteFields(REAL *P, REAL *U, signed char *Phase, REAL *Temperature,
                 Constants Ct, Variables Vb,
                 int index, int SVG) {
  //================================================
  // Name Output Files
  //================================================
  char FileNameF[LENMAX];
  char FileNameC[LENMAX];
  char FileNamePhase[LENMAX];
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
  printf("Writing file *%s.%s.* ... ", Ct.OutputPrefix, Indx);
  sprintf(FileNameF, "%s_%s.%s.vtk", PField, Ct.OutputPrefix, Indx);
  sprintf(FileNameC, "%s_%s.%s.vtk", CField, Ct.OutputPrefix, Indx);
  if (SVG > 0) {
    sprintf(FileNamePhase, "GB_%s.%s.vtk", Ct.OutputPrefix, Indx);
  } else {
    sprintf(FileNamePhase, "Grain_%s.%s.vtk", Ct.OutputPrefix, Indx);
  }
  sprintf(FileNameCompoX, "CompoX_%s.%s.dat", Ct.OutputPrefix, Indx);

  //================================================
  // Output Fields
  //================================================
  // REAL Delta=1.-(Vb.xtip*dx-Vb.x0+Vb.xoffs-Ct.Vp*Ct.dt*Vb.iter)/Vb.lT;
  REAL psi, phi, c, locg;
  signed char pha;

  REAL Delta_out = 1. - (Vb.xtip * dx - Vb.x0 + Vb.xoffs - Ct.Vp * Ct.dt * Vb.iter) / Vb.lT; // tip undercooling for FTA


  int XmaxC = XoutMAX;
  int XmaxF = XoutMAX;
  if (SVG > 0) {
    XmaxC = Nx + 2;
    XmaxF = Nx + 2;
  }

  //=======================
  // c Fields to files
  //=======================
  if (SVG > 0) // SVG: Psi & Compo
  {
    // .vtk header
    FILE *OutFileF;
    OutFileF = fopen(FileNameF, "w");
    fprintf(OutFileF, "# vtk DataFile Version 3.0\n");
    fprintf(OutFileF, "Delta %g\n", Delta_out);
    fprintf(OutFileF, "ASCII\n");
    fprintf(OutFileF, "DATASET STRUCTURED_POINTS\n");
    fprintf(OutFileF, "DIMENSIONS %d %d %d\n", (XmaxF - XoutMIN), Ny + 2, Nz + 2);
    fprintf(OutFileF, "ASPECT_RATIO %f %f %f\n", 1., 1., 1.);
    fprintf(OutFileF, "ORIGIN 0 0 0\n");
    fprintf(OutFileF, "POINT_DATA %d\n", (XmaxF - XoutMIN) * (Ny + 2) * (Nz + 2));
    fprintf(OutFileF, "SCALARS PF double 1\n"); //***?
    fprintf(OutFileF, "LOOKUP_TABLE default\n");

    // .vtk header
    FILE *OutFileC;
    OutFileC = fopen(FileNameC, "w");
    fprintf(OutFileC, "# vtk DataFile Version 3.0\n");
    fprintf(OutFileC, "Delta %g\n", Delta_out);
    fprintf(OutFileC, "ASCII\n");
    fprintf(OutFileC, "DATASET STRUCTURED_POINTS\n");
    fprintf(OutFileC, "DIMENSIONS %d %d %d\n", (XmaxC - XoutMIN), Ny + 2, Nz + 2);
    fprintf(OutFileC, "ASPECT_RATIO %f %f %f\n", 1., 1., 1.);
    fprintf(OutFileC, "ORIGIN 0 0 0\n");
    fprintf(OutFileC, "POINT_DATA %d\n", (XmaxC - XoutMIN) * (Ny + 2) * (Nz + 2));
    fprintf(OutFileC, "SCALARS C double 1\n");
    fprintf(OutFileC, "LOOKUP_TABLE default\n");

    // .vtk header (Phase field)
    FILE *OutFilePhase = fopen(FileNamePhase, "w");
    fprintf(OutFilePhase, "# vtk DataFile Version 3.0\n");
    fprintf(OutFilePhase, "Delta %g\n", Delta_out);
    fprintf(OutFilePhase, "ASCII\n");
    fprintf(OutFilePhase, "DATASET STRUCTURED_POINTS\n");
    fprintf(OutFilePhase, "DIMENSIONS %d %d %d\n", Nx + 2, Ny + 2, Nz + 2);
    fprintf(OutFilePhase, "ASPECT_RATIO %f %f %f\n", 1., 1., 1.);
    fprintf(OutFilePhase, "ORIGIN 0 0 0\n");
    fprintf(OutFilePhase, "POINT_DATA %d\n", (Nx + 2) * (Ny + 2) * (Nz + 2));
    fprintf(OutFilePhase, "SCALARS Phase int 1\n");
    fprintf(OutFilePhase, "LOOKUP_TABLE default\n");

    for (int k = 0; k < Nz + 2; k++) {
      for (int j = 0; j < Ny + 2; j++) {
        for (int i = 0; i < Nx + 2; i++) {
          psi = P[pos(i, j, k)];
          c = 0.5 * (Ct.opk - Ct.omk * tanh(psi / Ct.sqrt2)) * (1. + Ct.omk * U[pos(i, j, k)]);

          fprintf(OutFileF, DECIMALS_SVG, psi);
          fprintf(OutFileC, DECIMALS_SVG, c);
          fprintf(OutFilePhase, "%d ", Phase[pos(i, j, k)]); // phase
        }
      }
    }
    fclose(OutFileF);
    fclose(OutFileC);
    fclose(OutFilePhase);
  } else // Movies: PF & C (If SVG=0, output compressed movie file)
  {
    // int Xmaxf=( (XoutMAX == 0 || SVG>0) ? Nx : (int)(POSITION_0/dx+XoutMAX) );
    int Xmaxf = Nx;

    float val;
    int Freq = EVERY, nx = Xmaxf / Freq, ny = Ny / Freq, nz = Nz / Freq;

    // .vtk header
    FILE *OutFileF = fopen(FileNameF, "w");
    fprintf(OutFileF, "# vtk DataFile Version 3.0\n");
    fprintf(OutFileF, "Delta %g\n", Delta_out);
    fprintf(OutFileF, "BINARY\n");
    fprintf(OutFileF, "DATASET STRUCTURED_POINTS\n");
    fprintf(OutFileF, "DIMENSIONS %d %d %d\n", nx, ny, nz);
    fprintf(OutFileF, "ASPECT_RATIO %f %f %f\n", 1. * Freq, 1. * Freq, 1. * Freq);
    fprintf(OutFileF, "ORIGIN 0 0 0\n");
    fprintf(OutFileF, "POINT_DATA %d\n", nx * ny * nz);
    fprintf(OutFileF, "SCALARS PF float 1\n");
    fprintf(OutFileF, "LOOKUP_TABLE default\n");

    // .vtk header
    FILE *OutFileC = fopen(FileNameC, "w");
    fprintf(OutFileC, "# vtk DataFile Version 3.0\n");
    fprintf(OutFileC, "Delta %g\n", Delta_out);
    fprintf(OutFileC, "BINARY\n");
    fprintf(OutFileC, "DATASET STRUCTURED_POINTS\n");
    fprintf(OutFileC, "DIMENSIONS %d %d %d\n", nx, ny, nz);
    fprintf(OutFileC, "ASPECT_RATIO %f %f %f\n", 1. * Freq, 1. * Freq, 1. * Freq);
    fprintf(OutFileC, "ORIGIN 0 0 0\n");
    fprintf(OutFileC, "POINT_DATA %d\n", nx * ny * nz);
    fprintf(OutFileC, "SCALARS C float 1\n");
    fprintf(OutFileC, "LOOKUP_TABLE default\n");

    // .vtk header (Phase field)
    FILE *OutFilePhase = fopen(FileNamePhase, "w");
    fprintf(OutFilePhase, "# vtk DataFile Version 3.0\n");
    fprintf(OutFilePhase, "Delta %g\n", Delta_out);
    fprintf(OutFilePhase, "BINARY\n");
    fprintf(OutFilePhase, "DATASET STRUCTURED_POINTS\n");
    fprintf(OutFilePhase, "DIMENSIONS %d %d %d\n", nx, ny, nz);
    fprintf(OutFilePhase, "ASPECT_RATIO %f %f %f\n", 1. * Freq, 1. * Freq, 1. * Freq);
    fprintf(OutFilePhase, "ORIGIN 0 0 0\n");
    fprintf(OutFilePhase, "POINT_DATA %d\n", nx * ny * nz);
    fprintf(OutFilePhase, "SCALARS Grain float 1\n");
    fprintf(OutFilePhase, "LOOKUP_TABLE default\n");

    for (int k = 1; k <= nz; k++) {
      for (int j = 1; j <= ny; j++) {
        for (int i = 1; i <= nx; i++) {
          phi = (float)(tanh(P[pos(1 + (i - 1) * Freq, 1 + (j - 1) * Freq, 1 + (k - 1) * Freq)] / Ct.sqrt2));
          val = FloatSwap(phi);
          fwrite((void *)&val, sizeof(float), 1, OutFileF);

          c = (float)(0.5 * (Ct.opk - Ct.omk * tanh(P[pos(1 + (i - 1) * Freq, 1 + (j - 1) * Freq, 1 + (k - 1) * Freq)] / Ct.sqrt2)) * (1. + Ct.omk * U[pos(1 + (i - 1) * Freq, 1 + (j - 1) * Freq, 1 + (k - 1) * Freq)]));
          val = FloatSwap(c);
          fwrite((void *)&val, sizeof(float), 1, OutFileC);

          pha = Phase[pos(1 + (i - 1) * Freq, 1 + (j - 1) * Freq, 1 + (k - 1) * Freq)];
          locg = (float)(pha * 0.5 * (1 + phi));
          val = FloatSwap(locg);
          fwrite((void *)&val, sizeof(float), 1, OutFilePhase);
        }
      }
    }
    fclose(OutFileF);
    fclose(OutFileC);
    fclose(OutFilePhase);
  }

  //================================================
  // Output Composition profile at the tip location
  //================================================
  int jtip = (int)(Vb.ytip);
  int ktip = (int)(Vb.ztip);
  FILE *OutFileCompoX = fopen(FileNameCompoX, "w");
  fprintf(OutFileCompoX, "#t=%g \n", Vb.iter * Ct.dt * Ct.Tau0_sec);
  fprintf(OutFileCompoX, "#itip=%g \t", Vb.xtip);
  fprintf(OutFileCompoX, "jtip=%d \t", jtip);
  fprintf(OutFileCompoX, "ktip=%d \n", ktip);
  fprintf(OutFileCompoX, "x \t");
  fprintf(OutFileCompoX, "c/cl0 \n");
  for (int i = 0; i < Nx + 2; i++) {
    phi = tanh(P[pos(i, jtip, ktip)] / Ct.sqrt2);
    c = 0.5 * (Ct.opk - Ct.omk * phi) * (1. + Ct.omk * U[pos(i, jtip, ktip)]);
    // c=U[pos(i,jtip,ktip)];
    fprintf(OutFileCompoX, "%g \t", i * Ct.dx_microns);
    fprintf(OutFileCompoX, "%.16f \n", c);
  }
  fclose(OutFileCompoX);

  //================================================
  printf("written");
  if (SVG > 0) {
    printf(" (SVG)");
  }
  printf(".\n");

  //================================================
  // Output the final dat file
  //================================================

  if (index == IndexFINAL && SVG > 0) {
    char FileName_P[256];
    char FileName_C[256];
    char FileName_Pha[256];
    FILE *OutFile_P;
    FILE *OutFile_C;
    FILE *OutFile_Pha;
    sprintf(FileName_P, "Final_Psi.dat");
    sprintf(FileName_C, "Final_Compo.dat");
    sprintf(FileName_Pha, "Final_Phase.dat");
    OutFile_P = fopen(FileName_P, "w");
    OutFile_C = fopen(FileName_C, "w");
    OutFile_Pha = fopen(FileName_Pha, "w");

    for (int i = 0; i < Nx + 2; i++) {
      for (int j = 0; j < Ny + 2; j++) {
        for (int k = 0; k < Nz + 2; k++) {

          REAL d = P[pos(i, j, k)];
          fwrite((char *)&d, sizeof(REAL), 1, OutFile_P);

          d = U[pos(i, j, k)];
          fwrite((char *)&d, sizeof(REAL), 1, OutFile_C);

          signed char d1 = Phase[pos(i, j, k)];
          fwrite(&d1, sizeof(signed char), 1, OutFile_Pha);
        }
      }
    }
    fclose(OutFile_P);
    fclose(OutFile_C);
    fclose(OutFile_Pha);
    printf("Written Final_*.dat file. \n");
  }

  /*
  char FileName[256];
  char FileName2[256];
  char FileName3[256];
  FILE *OutFile;
  FILE *OutFile2;
  FILE *OutFile3;
sprintf(FileName,"m_Psi_binary.%s.%s.dat",Ct.OutputPrefix,Indx);
sprintf(FileName2,"m_U_binary.%s.%s.dat",Ct.OutputPrefix,Indx);
sprintf(FileName3,"m_Pha_binary.%s.%s.dat",Ct.OutputPrefix,Indx);
OutFile=fopen(FileName,"w");
OutFile2=fopen(FileName2,"w");
OutFile3=fopen(FileName3,"w");

for(int i=0; i<Nx+2; i++) {
  for(int j=0; j<Ny+2; j++) {
      for(int k=0; k<Nz+2; k++) {

          REAL d = P[pos(i,j,k)];

          fwrite((char*)&d,sizeof(REAL),1,OutFile);

          REAL d2 = U[pos(i,j,k)];

          fwrite((char*)&d2,sizeof(REAL),1,OutFile2);

          signed char d3 = Phase[pos(i,j,k)];

          fwrite(&d3,sizeof(signed char),1,OutFile3);

      }
  }
}
fclose(OutFile);
fclose(OutFile2);
fclose(OutFile3);

  printf("Written dat file %s. \n",FileName);
  */
}

void WriteBinaryFiles(REAL *P, REAL *U, signed char *Phase, REAL *Temperature, Constants Ct) {

  char FileName1[256];
  char FileName2[256];
  char FileName3[256];
  FILE *OutFile1;
  FILE *OutFile2;
  FILE *OutFile3;
  sprintf(FileName1, "Break_P.dat");
  sprintf(FileName2, "Break_U.dat");
  sprintf(FileName3, "Break_Phase.dat");
  OutFile1 = fopen(FileName1, "w");
  OutFile2 = fopen(FileName2, "w");
  OutFile3 = fopen(FileName3, "w");

  for (int i = 0; i < ((Nx + 2) * (Ny + 2) * (Nz + 2)); i++) {

    REAL Break_P = P[i];
    fwrite((char *)&Break_P, sizeof(REAL), 1, OutFile1);

    REAL Break_U = U[i];
    fwrite((char *)&Break_U, sizeof(REAL), 1, OutFile2);

    signed char Break_Phase = Phase[i];
    fwrite(&Break_Phase, sizeof(signed char), 1, OutFile3);
  }

  fclose(OutFile1);
  fclose(OutFile2);
  fclose(OutFile3);
}

// for 2D-record
#if (CutZplane != 0)
void WriteZ2Dfield(REAL *P, REAL *U,
                   signed char *Phase,
                   Constants Ct,
                   Variables Vb,
                   int zplane, int d2num, int Xmax2Df) {

  // for phi field
  char FileName2DF[LENMAX];
  sprintf(FileName2DF, "Z%dplane2DF_%s.%d.dat", zplane, Ct.OutputPrefix, d2num);
  FILE *OutFile2DF = fopen(FileName2DF, "w");

  fprintf(OutFile2DF, "#x[μm] \ty[μm] \tPF \tCompo \tPha \tGrain \n");

  REAL phi, c, locg;
  char pha;
  REAL sqrt2val = sqrt(2.);

  for (int i = 0; i < Xmax2Df; i++) {
    for (int j = 0; j < Ny + 2; j++) {
      phi = tanh(P[pos(i, j, zplane)] / sqrt2val);
      pha = Phase[pos(i, j, zplane)];
      c = 0.5 * (Ct.opk - Ct.omk * phi) * (1. + Ct.omk * U[pos(i, j, zplane)]);
      locg = (pha * 0.5 * (1 + phi));

      fprintf(OutFile2DF, "%g \t%g \t%g \t%g \t%d \t%g \n", i * Ct.dx_microns, j * Ct.dx_microns, phi, c, pha, locg);
    }
    fprintf(OutFile2DF, "\n");
  }
  fclose(OutFile2DF);

  printf("2D file (%d) is written.\n", d2num);
}
#endif

#if (COMPRESS)
int CompressFiles(Constants Ct) {
  ///////////////////////////////////////
  // Compress results to *.tar.gz files
  ///////////////////////////////////////
  printf(" Compressing *%s* files ... ", Ct.OutputPrefix);
  char Command[1024];
  sprintf(Command, "tar -czf PF_%s.tar.gz PF_%s*vtk %s*Param* %s*tip* %s_IndexTime.dat", Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix);
  system(Command);
  sprintf(Command, "rm PF_%s*vtk", Ct.OutputPrefix);
  system(Command);

  sprintf(Command, "tar -czf C_%s.tar.gz C_%s*vtk %s*Param* %s*tip* %s_IndexTime.dat", Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix);
  system(Command);
  sprintf(Command, "rm C_%s*vtk", Ct.OutputPrefix);
  system(Command);

  sprintf(Command, "tar -czf CompoX_%s.tar.gz CompoX_%s*dat %s*Param* %s*tip*", Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix);
  system(Command);
  sprintf(Command, "rm CompoX_%s*dat", Ct.OutputPrefix);
  system(Command);

  sprintf(Command, "tar -czf Grain_%s.tar.gz Grain_%s*vtk %s*Param* %s*tip* %s_IndexTime.dat", Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix);
  system(Command);
  sprintf(Command, "rm Grain_%s*vtk", Ct.OutputPrefix);
  system(Command);

#if (CutZplane != 0)
  sprintf(Command, "tar -czf Z%d_%s.tar.gz Z%dplane*_%s*dat %s*Param* %s*tip* %s_IndexTimeFor2D.dat", Zloc, Ct.OutputPrefix, Zloc, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix, Ct.OutputPrefix);
  system(Command);
  sprintf(Command, "rm Z%dplane*_%s*dat", Zloc, Ct.OutputPrefix);
  system(Command);
#endif

  printf("done! \n");
  return 0;
}
#endif // end of if(COMPRESS)

float FloatSwap(float f) {
  union {
    float f;
    unsigned char b[4];
  } dat1, dat2;

  dat1.f = f;
  dat2.b[0] = dat1.b[3];
  dat2.b[1] = dat1.b[2];
  dat2.b[2] = dat1.b[1];
  dat2.b[3] = dat1.b[0];
  return dat2.f;
};
