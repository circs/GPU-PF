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
 * 1) Karma, A. Phase-field formulation for quantitative modeling of alloy solidification. Phys Rev Lett 87, 115701 (2001).
 *
 * 2) Echebarria, B., Folch, R., Karma, A. & Plapp, M. Quantitative phase-field model of alloy solidification. \
 * Phys Rev E Stat Nonlin Soft Matter Phys 70, 061604 (2004).
 *--------------------------------------------------------------- */

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include <curand_kernel.h>
#include <curand.h>

#define NewP 			2	// Set 1 to use the Compute_P with vanishing 2nd order error; Set 2 to use the form with isotropy at the 2nd order
#define NewU 			1	// Set 1 to use the invariant form of Compute_U; Set 2 for invariant form with approximation

// -----------------------
//		PROCESS
// -----------------------
#define ANGLE1DEG	(0.)
#define ANGLE2DEG	(0.)
#define ANGLE1INIT	(1.)

#define GRADIENT	(24917.97507820541)  // K/m
#define VELOCITY	(182.85310024827348)  //microns/s
#define TIMESEC		(1.0833952093187644)

#define NUMWRITE	50
#define TIPWRITE	120

// -----------------------
//		SIMULATION
// -----------------------
#define E		22.62741699796952	// E=W/d0

#define Nx		384     // Dimension x
#define Ny		128   // Dimension y

#define dx		0.78125	        // Space step
#define dt0		0.002		// Time step (input, may be reajusted)
#define Kdt		0.8		    // Time step adjustment criterion (dt<=Kdt*dx*dx/4/D)
#define U_off 	0.9
#define x00		5.


#define REAL	float

#define UNDERCOOL_0 	(0.1)   
#define POSITION_0 		(480.)	// The position of pullback [micron]
#define IfTemporal 		0 		// set to 1 if there is a lateral temperature difference, for initialzing a half cell
// -----------------------
//	 BOUNDARY CONDITIONS
// -----------------------
#define BOUND_COND		NOFLUX
// #define SYMMETRY_Y0 	1		// no need it, for noflux it's also symmetrical (When NOFLUX BC, set 1 if symmetric with respect to axis j=0)

// -----------------------
//		  NOISE
// -----------------------
#define NOISE	WITHOUT
#define Fnoise	0.

// -----------------------
//		  OUTPUT
// -----------------------
#define PREFIX			"PRISMS_DEFAULT_"	// Prefix of Output File Name

#define OUTPUT_TIP		1
#define Npol			10	
#define OUTPUT_Psi		1
#define OUTPUT_C		1
#define OUTPUT_U		1
#define OUTPUT_Pha		0
#define OUTPUT_Phi		1
#define OUTPUT_Grains	0
#define	CheckMass		0			// set to 1 to check the mass conservation
#define	CheckGrad		0 			// Check the norm of gradient 

#define COORDINATES		MICRONS

#define INDEX_SECONDS	1
#define IndexFINAL		999999

// -----------------------
//		  INPUT
// -----------------------
#define INITfromFILE	0
#define INITfromPrefix	"SCN_100_"
#define	MIRROR_Y 		0
#define MULTIPLY_Y 		0 			// set tp a positive integer (>=2) for multiplying in y direction, mirror with respect to the original j=Ny axis (after MIRROR_Y)
#define ASYMMETRY_Y 	0			// set a integer (>0) to input the fields asymmetrially: input (1 to ASYMMETRY_Y) and create empty (ASYMMETRY_Y+1 to Ny) on the right
									// cannot use ASYMMETRY_Y together with MIRROR_Y and MULTIPLY_Y. Make sure ASYMMETRY_Y < Ny

// -----------------------
//		  GPU
// -----------------------
#define BSIZEMAX	64
#define BLOCKMAX	512

// -----------------------
//	  Mapping function
// -----------------------
#define pos(x,y)	((Nx+2)*(y)+(x))

// -------------------------
#define WITHOUT		0
#define NOFLUX		1
#define PERIODIC	2
#define GRID		1
#define MICRONS		2
#define FLAT		1
#define GAUSSIAN	2
// -------------------------
#define LENMAX		256
// -------------------------

// ----------------------------
// Parameters copied on the GPU
struct Constants
{  	
	REAL sqrt2;
	REAL PI;
	REAL W_d0;
	int NX;
	int NY;
	REAL Dx;
	REAL dt;
	REAL TotalTime_sec;	
	int niter;
	int IterOutFields;	
	REAL kcoeff;
	REAL omk;
	REAL opk;
	REAL Eps4;
	REAL D;
	REAL Vp;
	REAL lT;
	REAL Lambda;
	REAL W_microns;
	REAL dx_microns;
	REAL Tau0_sec;
	REAL xinit;
	REAL x0;
	REAL xint;
	REAL amp;
	REAL iq;
	REAL Alpha[2];
	REAL sAlpha[2];
	REAL cAlpha[2];
	REAL sAlpha2[2];
	REAL cAlpha2[2];
	REAL s2Alpha[2];
	REAL c2Alpha[2];
	REAL cAlphasAlpha[2];
	REAL GradT;	
	REAL Vpull;	
	REAL Diff;	
	REAL d0;		
	REAL lTherm;	
	int NwriteF;	
	
	// variables
	double xoffs;	
	int iter;
};

////////////////////////////
// GPU computing kernels
////////////////////////////
__global__ void Init(REAL *P1,REAL *P2,REAL *Cu1,REAL *Cu2,int *Pha,Constants *Param);
__global__ void Compute_P(REAL *Pcurr,REAL *Pnext,REAL *F,REAL *Cu,int *Pha,Constants *Param,curandState *state);
__global__ void Compute_U(REAL *Cucurr,REAL *Cunext,REAL *Pcurr,REAL *Pnext,REAL *F,Constants *Param);
__global__ void PullBack(REAL *P,REAL *Cu,int *Pha,REAL *F, Constants *Param,int off);
__global__ void	setup_kernel(unsigned long long seed,curandState *state);

////////////////////////////
// CPU I/O function
////////////////////////////
void InitFromFile(REAL *P,REAL *U,int *Pha,Constants *Param);
void WriteFields(char Prefix[LENMAX],int index,REAL *P,REAL *F,REAL *Cu,int *Pha,Constants *Param);
REAL WriteTip(char FileName[LENMAX],REAL xprev,REAL *P,REAL *Cu,int itPrev,Constants *Param);
REAL PolInt(REAL *XA,REAL *YA,REAL X);
REAL RTBIS(REAL *xp,REAL *yp);
int FindMaxX(REAL *P);

////////////////////////////
// Cuda Device Management
////////////////////////////
void DisplayDeviceProperties(int Ndev);
void GetMemUsage(int *Array,int Num);
int GetFreeDevice(int Num);
void AutoBlockSize(int *Bloc);

////////////////////////////////////////////////
//              Main CPU program              //
////////////////////////////////////////////////
int main(int argc, char **argv)
{
	clock_t begin=clock();
	
	// Attributing a GPU device
	int CudaDevice=0;	
	if(argc>1)
	{
		CudaDevice=atoi(argv[1]);
	}
	else
	{
		// Looking for a free device
		int Ndevices=-1;
		cudaGetDeviceCount(&Ndevices);
		if(Ndevices>1)
		{
			CudaDevice=GetFreeDevice(Ndevices);
			if(CudaDevice<0)
			{
				return 1;
			}
		}
	}
	cudaSetDevice(CudaDevice);
	DisplayDeviceProperties(CudaDevice);
	
	// Constants
	char OutputPrefix[LENMAX]=PREFIX;
	REAL PI=4.*atan(1.);
	REAL a1=5.*sqrt(2.)/8.;
	REAL a2=47./75.;
	
	// ------------------------------------------------------
	// ---------------- SIMULATION PARAMETERS ---------------
	// ------------------------------------------------------
	// ----------------- Physical parameters ----------------
	REAL m		=	1.5;		// |Liquidus slope|
	REAL c0		=	3.0;		// Nominal composition
	REAL kcoeff	=	0.14;		// Partition coefficient, K
	REAL Diff	=	12.7;		// Diffusion, microns^2/s
	REAL Eps4	=	0.01;
	REAL Gamma	=	6.4e-8;
	// ------------------ Process parameters -----------------   
	REAL GradT		=	GRADIENT;			// Temperature gradient, K/m
	REAL Vpull		=	VELOCITY;			// Pulling speed, microns/s
	REAL Alpha[2]	=	{ANGLE1DEG*PI/180.,ANGLE2DEG*PI/180.};	// rotation angle of anisotropy
	// ------------------------------------------------------
	REAL mc0	=	m*c0;				// |Liquidus slope|*Nominal composition, K
	REAL DT0	=	mc0/kcoeff-mc0;		// Solidification range, K
	REAL lTherm	=	DT0/GradT*1e6;		// microns
	REAL d0		=	Gamma/DT0*1.e6;		// capillarity length, microns
	// -------------- Non-dimensional parameters -------------   
	REAL nu=d0/lTherm;
	REAL D=a1*a2*E;								
	REAL Lambda=a1*E;
	REAL Vp=Vpull*d0/Diff*a1*a2*E*E;	
	REAL lT=1./(E*nu);							
	REAL lD=D/Vp;							
	REAL W_microns=E*d0;					// [microns]
	REAL dx_microns=W_microns*dx;			// [microns]
	REAL Tau0_sec=Vp/Vpull*W_microns;		// [seconds]	
	// ------------------------------------------------------   
	REAL Delta0 =	 0.;                    // Delta0=0 sets the concentration of the liquid as c_infty and of the solid as k c_infty
	REAL xinit  =	 1.-Delta0;				// Initial composition profile, 0: exp., 1: planar, <0: input file
	REAL xint   =	 POSITION_0/W_microns;	// Initial interface position [/W]
	REAL x0		=	 xint-(1.-UNDERCOOL_0)*lT;
	REAL iq    =	 1.;					// Wavenumber of the initial perturbation
	REAL amp   =	 1.;				// Amplitude of the initial perturbation
	// ----------------- Computational ----------------------
	REAL TotalTime_sec	=	TIMESEC;	// [seconds]	
	int NwriteF			=	NUMWRITE;	// Number of fields output
#if(OUTPUT_TIP)
	int NwriteTip		=	TIPWRITE;
#endif
	REAL TotalTime =	TotalTime_sec/Tau0_sec;	// [/Tau0]	
	REAL dt=dt0;
	if (dt>Kdt*dx*dx/4./D) 
	{
		dt=Kdt*dx*dx/4./D;
	}
	int niter=int(TotalTime/dt);
	int IterPull=int(dx/Vp/dt);
	int IterOutFields=niter/NwriteF;	
	int IterOutTip=niter/NwriteTip;
	// ------------------------------------------------------   

	// -------------------------------------------     
	// -------------- CPU Memory -----------------
	// -------------------------------------------     
	REAL xoffs=0.;
	int iter=0;
	
	// Arrays
	size_t SizeGrid = (Nx+2)*(Ny+2);
	REAL *h_Psi=(REAL*)malloc(SizeGrid*sizeof(REAL)) ;
#if(CheckGrad)
	REAL *h_Phi=(REAL*)malloc(SizeGrid*sizeof(REAL)) ;
#else
	REAL *h_Phi;
	h_Phi=h_Psi;
#endif	
	REAL *h_U=(REAL*)malloc(SizeGrid*sizeof(REAL)) ;	
	int *h_Phase=(int*)malloc(SizeGrid*sizeof(int)) ;	
	
	// ----------- h_Parameters storage -----------
	Constants h_Parameters[1];	
	
	(*h_Parameters).iter=0;
	(*h_Parameters).xoffs=0.;	
	
	(*h_Parameters).sqrt2=sqrt(2.);
	(*h_Parameters).PI=4.*atan(1.);
	(*h_Parameters).W_d0=E;
	(*h_Parameters).NX=Nx;
	(*h_Parameters).NY=Ny;
	(*h_Parameters).Dx=dx;
	(*h_Parameters).dt=dt;
	(*h_Parameters).TotalTime_sec=TotalTime_sec;	
	(*h_Parameters).niter=niter;
	(*h_Parameters).IterOutFields=IterOutFields;	
	(*h_Parameters).kcoeff=kcoeff;
	(*h_Parameters).omk=1.-kcoeff;
	(*h_Parameters).opk=1.+kcoeff;
	(*h_Parameters).Eps4=Eps4;
	(*h_Parameters).D=D;
	(*h_Parameters).Vp=Vp;
	(*h_Parameters).lT=lT; 
	(*h_Parameters).Lambda=Lambda; 
	(*h_Parameters).W_microns=W_microns;
	(*h_Parameters).dx_microns=dx_microns;
	(*h_Parameters).Tau0_sec=Tau0_sec;
	(*h_Parameters).xinit=xinit;
	(*h_Parameters).xint=xint;
	(*h_Parameters).amp=amp;
	(*h_Parameters).iq=iq;
	(*h_Parameters).x0=x0;
	for(int p=0; p<=1; p++)
	{
		(*h_Parameters).Alpha[p]=Alpha[p];
		(*h_Parameters).sAlpha[p]=sin(Alpha[p]);
		(*h_Parameters).cAlpha[p]=cos(Alpha[p]);
		(*h_Parameters).sAlpha2[p]=sin(Alpha[p])*sin(Alpha[p]);
		(*h_Parameters).cAlpha2[p]=cos(Alpha[p])*cos(Alpha[p]);
		(*h_Parameters).s2Alpha[p]=sin(2.*Alpha[p]);
		(*h_Parameters).c2Alpha[p]=cos(2.*Alpha[p]);
		(*h_Parameters).cAlphasAlpha[p]=cos(Alpha[p])*sin(Alpha[p]);
	}
	(*h_Parameters).TotalTime_sec=TotalTime_sec;
	(*h_Parameters).niter=niter;
	(*h_Parameters).IterOutFields=IterOutFields;
	(*h_Parameters).W_microns=W_microns;
	(*h_Parameters).dx_microns=dx_microns;
	(*h_Parameters).Tau0_sec=Tau0_sec;
	(*h_Parameters).GradT=GradT;
	(*h_Parameters).Vpull=Vpull;
	(*h_Parameters).Diff=Diff;
	(*h_Parameters).d0=d0;
	(*h_Parameters).lTherm=lTherm;
	(*h_Parameters).NwriteF=NwriteF;
	
	// -------------------------------------------     
	// -------------- GPU Memory -----------------
	// -------------------------------------------     
	Constants *Parameters;	
	cudaMalloc((void**)&Parameters,sizeof(Constants));
	
	REAL *Phi,*Psi1,*U1,*Psi2,*U2;
	cudaMalloc((void**)&Phi,SizeGrid*sizeof(REAL));
	
	cudaMalloc((void**)&Psi1,SizeGrid*sizeof(REAL));
	cudaMalloc((void**)&Psi2,SizeGrid*sizeof(REAL));
	cudaMalloc((void**)&U1,SizeGrid*sizeof(REAL));
	cudaMalloc((void**)&U2,SizeGrid*sizeof(REAL));
	
	REAL *Unext,*Ucurr,*Ubuff;
	Ucurr=U1;
	Unext=U2;
	
	REAL *Psinext,*Psicurr,*Psibuff;
	Psicurr=Psi1;
	Psinext=Psi2;
	
	int *Phase;
	cudaMalloc((void**)&Phase,SizeGrid*sizeof(int));
	
	curandState *devStates;
	cudaMalloc((void**)&devStates,SizeGrid*sizeof(curandState));
	
	// -------------- GPU BLOCKS -----------------  
	int BX=1,BY=1;
	for(int bx=1;bx<=BSIZEMAX;bx++)
	{
		if(((Nx+2.)/bx)==((Nx+2)/bx))
		{
			for(int by=1;by<=BSIZEMAX && bx*by<=BLOCKMAX;by++)
			{
				if(((Ny+2.)/by)==((Ny+2)/by))
				{
					if(bx*by==BX*BY)
					{
						int VAR=(BX-BY)*(BX-BY);
						int var=(bx-by)*(bx-by);
						if(var<VAR)
						{
							BX=bx; BY=by;
						}
					}
					else if(bx*by>BX*BY)
					{
						BX=bx; BY=by;
					}
				}
			}
		}
	}
	const int BLOCK_SIZE_X=BX;
	const int BLOCK_SIZE_Y=BY;
	
	dim3 SizeBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 NumBlocks((Nx+2)/BLOCK_SIZE_X,(Ny+2)/BLOCK_SIZE_Y);
	
	dim3 SizeBlockPull(1);
	dim3 NumBlocksPull(Ny+2);
	
	cudaFuncSetCacheConfig(Compute_P,cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(Compute_U,cudaFuncCachePreferL1);
	
	// ------------- Initializations -------------  
#if(INITfromFILE)	
	printf("Initializing fields from files...\n");
    cudaMemcpy(Parameters, h_Parameters, sizeof(Constants), cudaMemcpyHostToDevice);
	InitFromFile(h_Psi,h_U,h_Phase,h_Parameters);
    cudaMemcpy(Ucurr, h_U, SizeGrid*sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMemcpy(Psicurr, h_Psi, SizeGrid*sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMemcpy(Phase, h_Phase, SizeGrid*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Parameters, h_Parameters, sizeof(Constants), cudaMemcpyHostToDevice);
#else
	printf("Initializing fields.\n");
	cudaMemcpy(Parameters, h_Parameters, sizeof(Constants), cudaMemcpyHostToDevice);	
	Init<<<NumBlocks,SizeBlock>>>(Psi1,Psi2,U1,U2,Phase,Parameters);
	cudaMemcpy(h_U, Ucurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Psi, Psicurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Phase, Phase, SizeGrid*sizeof(int), cudaMemcpyDeviceToHost);
#endif	
	
	setup_kernel<<<NumBlocks,SizeBlock>>>(time(NULL),devStates);
	
#if(OUTPUT_TIP)
	REAL xtipbef=xint;
	int iTipPrev=0;
	
	// Output Tip /time & Steady state test
	char TipFileName[LENMAX];
	sprintf(TipFileName,"%s.tip.dat",OutputPrefix);
	FILE *TipF=fopen(TipFileName,"w");	
	fprintf(TipF,"Time[s] \t");					// 1	
	fprintf(TipF,"Delta[-] \t");				// 2	
	fprintf(TipF,"Omega[-] \t");				// 3	
	fprintf(TipF,"Rho_{Tip}[microns] \t");		// 4	
	fprintf(TipF,"Vel_{Tip}[microns/s] \t");	// 5	
	fprintf(TipF,"Peclet_{Tip} \t");			// 6	
	fprintf(TipF,"Sigma_{Tip} \t");				// 7	
	fprintf(TipF,"d_{0,Tip}[microns] \t");		// 8	
	fprintf(TipF,"x_{Tip}[microns] \t");		// 9	
	fprintf(TipF,"y_{Tip}[microns] \t");		// 10	
	fprintf(TipF,"x_{offset}[microns] \t");		// 11	
	fprintf(TipF,"i_{Tip} \t");					// 12	
	fprintf(TipF,"j_{Tip} \t");					// 13	
	fprintf(TipF,"x_{Tip}/N_x \t");				// 14	
	fprintf(TipF,"y_{Tip}/N_y \t");				// 15	
	fprintf(TipF,"x_{Tip}/Mat[microns] \t");	// 16
#if(CheckMass)	
	fprintf(TipF,"cl/c0l Ave. \t");				// 17	
#endif	
	fprintf(TipF,"\n");
	fclose(TipF);
#endif
	
	// ----------------------------------------------------     
	printf("\n----------------------------------------");
	printf("\n         SIMULATION PARAMETERS");
	printf("\n----------------------------------------\n");
	printf(" Vp   = %g microns/s\n",Vpull);
	printf(" G    = %g K/m\n",GradT);
	printf(" D    = %g microns^2/s\n",Diff);
	printf(" lT   = %g microns\n",lTherm);
	printf(" lD   = %g microns\n",Diff/Vpull);
	printf(" d0   = %g microns\n",d0);
	printf(" Eps4 = %g\n",Eps4);
	printf(" k    = %g\n",kcoeff);
	printf("-------------- DIMENSIONS --------------\n");	
	printf(" Dimension/X = %g microns\n",Nx*dx_microns);
	printf(" Dimension/Y = %g microns\n",Ny*dx_microns);	
	printf(" Total time  = %g seconds\n",TotalTime_sec);
	printf("------------ DIMENSIONLESS -------------\n");
	printf(" W/d0 = %g\n",E);
	printf(" W    = %g microns\n",W_microns);
	printf(" Tau0 = %g seconds\n",Tau0_sec);
	printf("----------------------------------------\n");
	printf(" D          = %g\n",D);
	printf(" Vp         = %g\n",Vp);
	printf(" lT         = %g\n",lT);
	printf(" lD         = %g\n",lD);
	printf(" Lambda     = %g\n",Lambda);
	printf(" Total time = %g \n",TotalTime);
	printf("------------ COMPUTATIONAL -------------\n");
	printf(" Nx+2 = %d\n",Nx+2);
	printf(" Ny+2 = %d\n",Ny+2);
	printf(" dx   = %g\n",dx);
	printf("      = %g microns\n",dx_microns);
	printf(" dt   = %g\n",dt);
	printf("      = %g seconds\n",dt*Tau0_sec);
#if(BOUND_COND==NOFLUX)
	printf(" Boundary Conditions /y : No-Flux\n");
#endif
#if(BOUND_COND==PERIODIC)
	printf(" Boundary Conditions /y : Periodic\n");
#endif
#if(NOISE!=WITHOUT)
#if(NOISE==FLAT)
	printf(" Noise: Flat distribution\n");
	printf("        Amplitude = %g\n",Fnoise);
#endif
#if(NOISE==GAUSSIAN)
	printf(" Noise: Gaussian distribution\n");
	printf("        Amplitude = %g\n",Fnoise);
#endif
#endif
	printf("----------------- GPU ------------------\n");
	printf("Block size /X = %d\n",BLOCK_SIZE_X);
	printf("Block size /Y = %d\n",BLOCK_SIZE_Y);
	printf("Thread/Block  = %d\n",BLOCK_SIZE_X*BLOCK_SIZE_Y);
	printf("Number of Blocks /X = %g\n",((Nx+2.)/BLOCK_SIZE_X));
	printf("Number of Blocks /Y = %g\n",((Ny+2.)/BLOCK_SIZE_Y));
	printf("Number of Blocks    = %d\n",((Nx+2)/BLOCK_SIZE_X*(Ny+2)/BLOCK_SIZE_Y));
	printf("----------------------------------------\n");
	printf(" Time step input dt = %g\n",dt0);
	if (dt0>dx*dx/4./D) 
	printf("                 dt > dx^2/(4*D) ...\n");
	if (dt<dt0)
	printf("       =>        dt=%g\n",dt);
	printf(" Number of iterations: %d\n",niter);
	printf(" Output every %d iterations\n",IterOutFields);
	printf(" Checking tip position every %d iterations\n",IterPull);
	printf("----------------------------------------\n\n");
	// ----------------------------------------------------     
	//return 1;
	
	printf("Writing file ***_%s.%d.*** ... ",OutputPrefix,0);	
	WriteFields(OutputPrefix,0,h_Psi,h_Phi,h_U,h_Phase,h_Parameters);
	printf("written.\n");	
	//return 1;
		
	// ----------------------------
	//			Time loop	
	// ----------------------------
    
    // tempo
    // niter=1;
    
	for(iter=1;iter<=niter;iter=iter+1)
	{	
		// ================
		// Fields evolution
		// ================
		Compute_P<<<NumBlocks,SizeBlock>>>(Psicurr,Psinext,Phi,Ucurr,Phase,Parameters,devStates);	
		Compute_U<<<NumBlocks,SizeBlock>>>(Ucurr,Unext,Psicurr,Psinext,Phi,Parameters);
		
		Psibuff=Psinext;
		Psinext=Psicurr;
		Psicurr=Psibuff;
		
		Ubuff=Unext;
		Unext=Ucurr;
		Ucurr=Ubuff;
		
		// =========
		// Pull back
		// =========

		// if(iter%IterPull==0)  
		// {
		// 	cudaMemcpy(h_Psi, Psicurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
		// 	int off=(int)(FindMaxX(h_Psi))-(int)(POSITION_0/dx_microns);
		// 	if(off)
		// 	{
		// 		PullBack<<<NumBlocksPull,SizeBlockPull>>>(Psicurr,Ucurr,Phase,Phi,Parameters,off);
		// 		xoffs+=off*dx;
		// 	}
		// }
		
		// =============
		// Output fields
		// =============
		if(iter%IterOutFields==0 && iter<niter)  
		{
			cudaMemcpy(h_U, Ucurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Psi, Psicurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Phase, Phase, SizeGrid*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);
			int indx=(iter+1)/IterOutFields;
#if(OUTINDEXSEC)
			indx=int(iter*dt*Tau0_sec);
#endif
			printf("Writing file ***_%s.%d.*** ... ",OutputPrefix,indx);	
			WriteFields(OutputPrefix,indx,h_Psi,h_Phi,h_U,h_Phase,h_Parameters);
			printf("written.\n");	
		}		

		// ===============
		// Output tip data
		// ===============
#if(OUTPUT_TIP)
		if(iter%IterOutTip==0)
		{	
			cudaMemcpy(h_Psi, Psicurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_U, Ucurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);
			xtipbef=WriteTip(TipFileName,xtipbef,h_Psi,h_U,iTipPrev,h_Parameters);
			iTipPrev=iter;
		}
#endif

	} // end main time loop
	//---------------------
		
	// Writing final results
	printf("Writing file ***_%s.Final.*** ... ",OutputPrefix);	
	cudaMemcpy(h_U, Ucurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Psi, Psicurr, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Phase, Phase, SizeGrid*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Parameters, Parameters, sizeof(Constants), cudaMemcpyDeviceToHost);
#if(CheckGrad)
	cudaMemcpy(h_Phi, Phi, SizeGrid*sizeof(REAL), cudaMemcpyDeviceToHost);
#endif	
	WriteFields(OutputPrefix,IndexFINAL,h_Psi,h_Phi,h_U,h_Phase,h_Parameters);
	printf("written.\n");	
	
	cudaFree(Phi) ;
	cudaFree(Psi1) ;
	cudaFree(Psi2) ;
	cudaFree(U1) ;
	cudaFree(U2) ;
	cudaFree(Phase) ;
	cudaFree(Parameters) ;
	
	cudaFree(devStates) ;
	
	free(h_Psi) ;
	free(h_U) ;
	free(h_Phase) ;
#if(CheckGrad)
	free(h_Phi);
#endif	
	
	return EXIT_SUCCESS;
}
//------------------------ End Main Program ------------------------

// Constants
#define sqrt2			(*Param).sqrt2
#define PI				(*Param).PI
#define dt				(*Param).dt
#define niter			(*Param).niter
#define kcoeff			(*Param).kcoeff
#define omk				(*Param).omk
#define opk				(*Param).opk
#define Eps4			(*Param).Eps4
#define D				(*Param).D
#define Vp				(*Param).Vp
#define lT				(*Param).lT
#define Lambda			(*Param).Lambda
#define xinit			(*Param).xinit
#define xint			(*Param).xint
#define x0 				(*Param).x0
#define amp				(*Param).amp
#define iq				(*Param).iq
#define Alpha			(*Param).Alpha
#define sAlpha			(*Param).sAlpha
#define cAlpha			(*Param).cAlpha
#define sAlpha2			(*Param).sAlpha2
#define cAlpha2			(*Param).cAlpha2
#define s2Alpha			(*Param).s2Alpha
#define c2Alpha			(*Param).c2Alpha
#define cAlphasAlpha	(*Param).cAlphasAlpha
#define TotalTime_sec	(*Param).TotalTime_sec
#define niter			(*Param).niter
#define IterOutFields	(*Param).IterOutFields
#define W_microns		(*Param).W_microns
#define dx_microns		(*Param).dx_microns
#define Tau0_sec		(*Param).Tau0_sec
#define GradT			(*Param).GradT
#define Vpull			(*Param).Vpull
#define Diff			(*Param).Diff
#define d0				(*Param).d0
#define lTherm			(*Param).lTherm
#define NwriteF			(*Param).NwriteF

#define iter			(*Param).iter
#define xoffs			(*Param).xoffs

// Boundary Conditions
#define IMIN	0
#define IMAX	(Nx+1)

#if(BOUND_COND==NOFLUX)
#define JMIN	0
#define JMAX	(Ny+1)
#endif

#if(BOUND_COND==PERIODIC)
#define JMIN	(Ny+1)
#define JMAX	0
#endif

/////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Initializations ///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
// __global__ void Init(REAL *P1,REAL *P2,REAL *Cu1,REAL *Cu2,int *Pha,Constants *Param)
// {
// 	int i = threadIdx.x + blockIdx.x * blockDim.x;
// 	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
// 	REAL xi=(xint/dx)*dx;
// 	// REAL xinty=xi/dx-amp*.5*(1.+cos(iq*3.14159*(Ny-j+1)/Ny));
    
//     REAL xinty=xi/dx + 0.5*amp*(1.+cos(iq*3.14159*(Ny-j+1)/Ny));
//     // REAL xinty=xi/dx + 10.*sqrt(1./2./PI/16.)*(exp(-(float)j/2./16.));	// initial shape is a Gaussian distribution
	
// 	P1[pos(i,j)]=P2[pos(i,j)]=-(i-xinty)*dx;
	
// 	REAL phi=tanh(P1[pos(i,j)]/sqrt2);
// 	REAL c = (i<xinty)? (0.5*(opk-omk*phi))*(1.-xinit*omk) :
// 	(0.5*(opk-omk*phi))*(kcoeff+omk*(1.-xinit)*exp(-(i-xinty)*dx*Vp/D)) ;
	
// 	Cu1[pos(i,j)]=Cu2[pos(i,j)]=(2.*c-opk+omk*phi)/omk/(opk-omk*phi);
	
// 	Pha[pos(i,j)]=0;
// 	if (P1[pos(i,j)]>0.)
// 	{
// 		if (j<=(Ny+1)*ANGLE1INIT)
// 		{	Pha[pos(i,j)]=-1;	}
// 		else
// 		{	Pha[pos(i,j)]=1;	}
// 	}
// }

// From the initial condition of the PRISMS-PF
__global__ void Init(REAL *P1,REAL *P2,REAL *Cu1,REAL *Cu2,int *Pha,Constants *Param)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	REAL xi = (xint/dx) * dx;
	// REAL xinty=xi/dx-amp*.5*(1.+cos(iq*3.14159*(Ny-j+1)/Ny));
    
    REAL xinty=xi/dx + 0.5*amp*(1.+cos(iq*3.14159*(Ny-j+1)/Ny));
    // REAL xinty=xi/dx + 10.*sqrt(1./2./PI/16.)*(exp(-(float)j/2./16.));	// initial shape is a Gaussian distribution
	
    REAL dist = sqrt(static_cast<REAL>((i-0.5) * (i-0.5) + (j-0.5) * (j-0.5))) * dx; 
    REAL rad = 5.; //[W]

	P1[pos(i,j)]=P2[pos(i,j)]=(rad-dist);
	
	REAL phi=tanh(P1[pos(i,j)]/sqrt2);

	REAL c = (dist<rad)? (0.5*(opk-omk*phi))*(1.-xinit*omk) :
	(0.5*(opk-omk*phi))*(kcoeff+omk*(1.-xinit)*exp(-(dist/dx-rad/dx)*dx*Vp/D)) ;
	
	// Cu1[pos(i,j)]=Cu2[pos(i,j)]=(2.*c-opk+omk*phi)/omk/(opk-omk*phi);
	Cu1[pos(i,j)] = Cu2[pos(i,j)] = -1.;
	
	Pha[pos(i,j)]=0;
	if (P1[pos(i,j)]>0.)
	{
		if (j<=(Ny+1)*ANGLE1INIT)
		{	Pha[pos(i,j)]=-1;	}
		else
		{	Pha[pos(i,j)]=1;	}
	}
}



void InitFromFile(REAL *P,REAL *U,int *Pha,Constants *Param)
{
#define pos_old(x,y)    ((Nx_old+2)*(y)+(x))

    printf("initialzing starts ");
    char buff[15];
    int Nx_old,Ny_old;
    REAL *Pold,*Cold,Del0;
    
    char INIT_FILE_P[LENMAX];
    char INIT_FILE_C[LENMAX];
    sprintf(INIT_FILE_P,"%s_Psi.Final.dat",INITfromPrefix);
    sprintf(INIT_FILE_C,"%s_C.Final.dat",INITfromPrefix);
	// ======================================
	// Get Initial Psi values ==> Pold[pos(i,j)]
	// ======================================
    std::string line;
    std::ifstream InFileP(INIT_FILE_P);
    
	if (InFileP.is_open())
	{
        //get Nx
        getline (InFileP,line);
        std::istringstream NxIn(line);
        NxIn >> buff;
        NxIn >> Nx_old;
        //get Ny
        getline (InFileP,line);
        std::istringstream NyIn(line);
        NyIn >> buff;
        NyIn >> Ny_old;
        //get Delta
        getline (InFileP,line);
        std::istringstream DeltaIn(line);
        DeltaIn >> buff;
        DeltaIn >> Del0;
        xinit=1.-Del0; // (1-Del0) is stored into (*h_parameters).x0

        getline (InFileP,line); //dx
        getline (InFileP,line); //coordinates: #x y Psi
        
        Pold=(REAL*)malloc(((Nx_old+2)*(Ny_old+2))*sizeof(REAL));
        
        for (int i=0; i<Nx_old+2; i++)
        {
            for (int j=0; j<Ny_old+2; j++)
            {
                getline(InFileP,line);
                std::istringstream ValuesP(line);
                ValuesP >> buff; //x coordinate
                ValuesP >> buff; //y coordinate
                ValuesP >> Pold[pos_old(i,j)]; //Old Psi value at (i,j)
            }
            getline(InFileP,line);
        }
        InFileP.close();
	}
	else 
	{
		std::cout << "Unable to open file "<<INIT_FILE_P<< std::endl;
	}
	
	// ====================================
	// Get Initial C values ==> Cold[pos(i,j)]
	// ====================================
	std::ifstream InFileC (INIT_FILE_C);
	if (InFileC.is_open())
	{
        getline (InFileC,line); // for Nx
        getline (InFileC,line); // for Ny
        getline (InFileC,line); // for Delta
        getline (InFileC,line); // for dx
        getline (InFileC,line); // for coordinates: #x y Psi
        
        Cold=(REAL*)malloc(((Nx_old+2)*(Ny_old+2))*sizeof(REAL));
        
        for (int i=0; i<Nx_old+2; i++)
        {
            for (int j=0; j<Ny_old+2; j++)
            {
                getline(InFileC,line);
                std::istringstream ValuesC(line);
                ValuesC >> buff; //x coordinate
                ValuesC >> buff; //y coordinate
                ValuesC >> Cold[pos_old(i,j)]; //Old Psi value at (i,j)
            }
            getline(InFileC,line);
        }
        InFileC.close();
	}
	else 
	{
		std::cout << "Unable to open file "<<INIT_FILE_C<< std::endl;
	}
	
    //======================================
    // Translate
    // Pold[pos_old(i,j)] => P[pos(i,j)]
    // Cold[pos_old(i,j)] => C[pos(i,j)]
    //======================================
    REAL phi;
    REAL P0,P1;
    REAL C0,C1,c;
    REAL yold,yi;
    int iold,jold,inew,jnew;
    int imax=((Nx<=Nx_old)?Nx:Nx_old);
#if(ASYMMETRY_Y)
    int jmax=ASYMMETRY_Y;    // if use ASYMMETRY_Y, import the old field and interpolate from 1 to ASYMMETRY_Y
#else    
	int jmax=(MULTIPLY_Y>1)? Ny/MULTIPLY_Y : Ny ;
#endif    

    for (int i=1; i<=imax; i++)
    {
        for (int j=1; j<=jmax; j++)
        {
            jnew=(MIRROR_Y)? (jmax+1-j):j;
            inew=i;
            iold=i;

            yold=1.+(REAL)(Ny_old-2.)/(REAL)(jmax-2.)*(j-1.);
            jold=(int)(yold);
            yi=yold-jold;
            
            //Psi
            P0=Pold[pos_old(iold,jold)];
            P1=Pold[pos_old(iold,jold+1)];
            
            P[pos(inew,jnew)]=P0+(P1-P0)*yi;

            //composition
            C0=Cold[pos_old(iold,jold)];
            C1=Cold[pos_old(iold,jold+1)];
            
            c=C0+(C1-C0)*yi;
            
            phi=tanh(P[pos(inew,jnew)]/sqrt2);
            U[pos(inew,jnew)]=(2.*c-opk+omk*phi)/omk/(opk-omk*phi);
        }
    }

#if(ASYMMETRY_Y)    // if use ASYMMETRY_Y, fill from (ASYMMETRY_Y+1) to Ny by the fields at j=ASYMMETRY_Y
    for (int i=1; i<=imax; i++)
    {
        for (int j=ASYMMETRY_Y+1; j<=Ny; j++)
        {
            P[pos(i,j)]=P[pos(i,ASYMMETRY_Y)];
            U[pos(i,j)]=U[pos(i,ASYMMETRY_Y)];
        }
    }    
#endif   

	// ==========
	// Multiply/Y
	// ==========
	int My = MULTIPLY_Y ;
	if(MULTIPLY_Y>1)
	{
		for(int i=1; i<=Nx; i++) 
		{    
			for (int iy=2;iy<=My;iy++)
			{
				for(int j=1; j<=jmax; j++) 
				{	
					// Mirror on y if(iy%2==0)
					jnew = (iy%2==0)? jmax*(iy-1)+jmax+1-j : jmax*(iy-1)+j ;

					P[pos(i,jnew)]=P[pos(i,j)];
					U[pos(i,jnew)]=U[pos(i,j)];
				}
			}
		}
	}  

    //Boundary Conditions
    for (int j=1; j<=Ny; j++)
    {
        //on IMIN
        P[pos(IMIN,j)]=P[pos(1,j)];
        U[pos(IMIN,j)]=U[pos(1,j)];
        //on IMAX
        P[pos(IMAX,j)]=P[pos(Nx,j)];
        U[pos(IMAX,j)]=U[pos(Nx,j)];
    }
    for (int i=1; i<=Nx; i++)
    {
        //on JMIN
        P[pos(i,JMIN)]=P[pos(i,1)];
        U[pos(i,JMIN)]=U[pos(i,1)];
        //on JMAX
        P[pos(i,JMAX)]=P[pos(i,Ny)];
        U[pos(i,JMAX)]=U[pos(i,Ny)];
    }
    
    //Edge points
    P[pos(IMIN,JMIN)]=P[pos(1,1)]; //IMIN JMIN
    U[pos(IMIN,JMIN)]=U[pos(1,1)];
    P[pos(IMIN,JMAX)]=P[pos(1,Ny)]; //IMIN JMAX
    U[pos(IMIN,JMAX)]=U[pos(1,Ny)];
    P[pos(IMAX,JMIN)]=P[pos(Nx,1)]; //IMAX JMIN
    U[pos(IMAX,JMIN)]=U[pos(Nx,1)];
    P[pos(IMAX,JMAX)]=P[pos(Nx,Ny)]; //IMAX JMAX
    U[pos(IMAX,JMAX)]=U[pos(Nx,Ny)];
    
    for (int i=0; i<Nx+2; i++)
    {
        for (int j=0; j<Ny+2; j++)
        {
            Pha[pos(i,j)]=(P[pos(i,j)]<0.)? 0 : -1 ;
        }
    }
    
    free(Pold);
    free(Cold);
}

__global__ void	setup_kernel(unsigned long long seed,curandState *state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	curand_init(seed,pos(i,j),0,&state[pos(i,j)]);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Computation /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
__global__ void Compute_P(REAL *Pcurr,REAL *Pnext,REAL *F,REAL *Cu,int *Pha,Constants *Param,curandState *state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(i!=0 && i!=Nx+1 && j!=0 && j!=Ny+1)
	{
		REAL dphix,dphiy,phxx,phyy,phxy,dphixx,dphiyy,dphixy,NormG4,thx,thy,c4,s4,anis;
		REAL Tau,temp;
		
		REAL phi=F[pos(i,j)]=tanh(Pcurr[pos(i,j)]/sqrt2);	
		
		REAL psi=Pcurr[pos(i,j)];		
		REAL omp2=1.-phi*phi;
		int pha=Pha[pos(i,j)];		
		
		Pnext[pos(i,j)]= Pcurr[pos(i,j)];

#if(NewP==1)	// the new Compute_P with the second order error vanishing
		// need to use i pm 2 and j pm 2
		int ip2=i+2;
		int im2=i-2;
		int jp2=j+2;
		int jm2=j-2;
		if(i==1) 	im2=0;
		if(i==Nx) 	ip2=Nx+1;
	#if(BOUND_COND==NOFLUX)	
		// if(j==1) 	jm2=(SYMMETRY_Y0)?2:0;
		// if(j==Ny)	jp2=Ny+1;
		if(j==1) 	jm2=2;
		if(j==Ny)	jp2=Ny-1;		
	#elif(BOUND_COND==PERIODIC)
		if(j==1) 	jm2=Ny-1;
		if(j==Ny)	jp2=2;	
	#endif	

		// ------------
		// Gradient h_Psi
		// ------------		
		REAL phx=2.*(Pcurr[pos(i+1,j)]-Pcurr[pos(i-1,j)])/(3.*dx)-(Pcurr[pos(ip2,j)]-Pcurr[pos(im2,j)])/(12.*dx); // 2nd order error vanishes
		REAL phy=2.*(Pcurr[pos(i,j+1)]-Pcurr[pos(i,j-1)])/(3.*dx)-(Pcurr[pos(i,jp2)]-Pcurr[pos(i,jm2)])/(12.*dx); // 2nd order error vanishes
		REAL NormG2=phx*phx+phy*phy;

		phxx=(-5.*psi/2.+4.*Pcurr[pos(i+1,j)]/3.+4.*Pcurr[pos(i-1,j)]/3.-Pcurr[pos(ip2,j)]/12.-Pcurr[pos(im2,j)]/12.)/(dx*dx); // 2nd order error vanishes
		phyy=(-5.*psi/2.+4.*Pcurr[pos(i,j+1)]/3.+4.*Pcurr[pos(i,j-1)]/3.-Pcurr[pos(i,jp2)]/12.-Pcurr[pos(i,jm2)]/12.)/(dx*dx); // 2nd order error vanishes			
		REAL phy2h = 2.*(Pcurr[pos(i+1,j+1)]-Pcurr[pos(i+1,j-1)])/(3.*dx)-(Pcurr[pos(i+1,jp2)]-Pcurr[pos(i+1,jm2)])/(12.*dx);
		REAL phym2h= 2.*(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i-1,j-1)])/(3.*dx)-(Pcurr[pos(i-1,jp2)]-Pcurr[pos(i-1,jm2)])/(12.*dx);

		REAL dpy1,dpy2,dpy3,phyh,phymh; // for calculating dpdy at (h,0) and (-h,0)
		dpy1 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(i,j-1)])+(Pcurr[pos(i,j+1)]-Pcurr[pos(i+1,j-1)]) )/(4.*dx);
		dpy2 = ( (Pcurr[pos(ip2,j+1)]-Pcurr[pos(i-1,j-1)])+(Pcurr[pos(i-1,j+1)]-Pcurr[pos(ip2,j-1)]) )/(4.*dx);
		dpy3 = ( (Pcurr[pos(i+1,jp2)]-Pcurr[pos(i,jm2)])+(Pcurr[pos(i,jp2)]-Pcurr[pos(i+1,jm2)]) )/(8.*dx);
		phyh = 35.*dpy1/24.-dpy2/8.-dpy3/3.;

		dpy1 = ( (Pcurr[pos(i,j+1)]-Pcurr[pos(i-1,j-1)])+(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i,j-1)]) )/(4.*dx);
		dpy2 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(im2,j-1)])+(Pcurr[pos(im2,j+1)]-Pcurr[pos(i+1,j-1)]) )/(4.*dx);
		dpy3 = ( (Pcurr[pos(i,jp2)]-Pcurr[pos(i-1,jm2)])+(Pcurr[pos(i-1,jp2)]-Pcurr[pos(i,jm2)]) )/(8.*dx);
		phymh = 35.*dpy1/24.-dpy2/8.-dpy3/3.;

		phxy = 4.*(phyh-phymh)/(3.*dx)-(phy2h-phym2h)/(6.*dx); // 2nd order error vanishes

		// ----------
		// Anisotropy
		// ----------				
		if(fabs(omp2)>=1.e-6 && fabs(NormG2)>1.e-15) 
		{
			if (!pha && fabs(omp2)>=.01)
			{
				int sum=Pha[pos(i+1,j)]+Pha[pos(i-1,j)]+Pha[pos(i,j+1)]+Pha[pos(i,j-1)]
				+Pha[pos(i+1,j+1)]+Pha[pos(i+1,j-1)]+Pha[pos(i-1,j+1)]+Pha[pos(i-1,j-1)];
				if (sum)	
				Pha[pos(i,j)]=pha=(sum>0)-(sum<0);
			}
			
			if (pha)
			{
				pha=(1+pha)/2;	// From now pha is 0 or 1 (not -1 or 1)
				
				dphix=phx*cAlpha[pha]+phy*sAlpha[pha];
				dphiy=-phx*sAlpha[pha]+phy*cAlpha[pha];
				
				//phxx=(Pcurr[pos(i+1,j)]-2.*psi+Pcurr[pos(i-1,j)])/(dx*dx);
				//phyy=(Pcurr[pos(i,j+1)]-2.*psi+Pcurr[pos(i,j-1)])/(dx*dx);
				//phxy=(Pcurr[pos(i+1,j+1)]-Pcurr[pos(i-1,j+1)]-Pcurr[pos(i+1,j-1)]+Pcurr[pos(i-1,j-1)])/(4.*dx*dx);

				dphixx=phyy*sAlpha2[pha]+phxy*s2Alpha[pha]+phxx*cAlpha2[pha];
				dphiyy=phyy*cAlpha2[pha]-phxy*s2Alpha[pha]+phxx*sAlpha2[pha];
				dphixy=phyy*cAlphasAlpha[pha]+phxy*c2Alpha[pha]-phxx*cAlphasAlpha[pha];
				NormG4=NormG2*NormG2;
				thx=(dphix*dphixy-dphiy*dphixx)/NormG2;
				thy=(dphix*dphiyy-dphiy*dphixy)/NormG2;
				c4=-8.*dphix*dphix*dphiy*dphiy/NormG4+1.;
				s4=4.*(dphix*dphix*dphix*dphiy-dphiy*dphiy*dphiy*dphix)/NormG4;
				anis=c4*(2.+Eps4*c4)*(phxx+phyy - sqrt2*phi*NormG2 )  		// 2nd term (incomplete)
				-8.*s4*(1.+Eps4*c4)*(thx*dphix+thy*dphiy) 					// 1st term
				-16.*(c4+Eps4*(c4*c4-s4*s4))*(thy*dphix-thx*dphiy); 		// 3rd term
			}
			else
			{
				anis=0.;
				c4=0.;
			}
			
#if(NOISE!=WITHOUT)
			curandState localState;
			localState=state[pos(i,j)];
			REAL ran1=curand_uniform_double(&localState);
#if(NOISE==GAUSSIAN)
			REAL ran2=curand_uniform_double(&localState);
			Pnext[pos(i,j)] += Fnoise*sqrt(dt)*sqrt(-2.0*log(1.0-ran1))*cos(2.0*PI*ran2);
#endif
#if(NOISE==FLAT)
			Pnext[pos(i,j)] += Fnoise*sqrt(dt)*(ran1-.5);
#endif
			state[pos(i,j)]=localState;
#endif
			
		}
		else
		{
			anis=0.;
			c4=0.;
		}
		
		// ------
		// dPsidt
		// ------
    
    #if(IfTemporal)    
		// temporal update
		temp=(i*dx +0.5*j +xoffs-xint+xinit*lT-Vp*dt*iter)/lT;
	#else    
        // original
		temp=(i*dx+xoffs-x0-Vp*dt*iter)/lT;
	#endif				
        
		Tau=(temp>1.)? kcoeff : (1.-omk*temp);
		
		Pnext[pos(i,j)]+= dt*(phxx+phyy 									// the remaining 2nd term when computing "anis"
							  +sqrt2*phi*(1.-NormG2) 						// 4th term + the remaining 2nd term when computing "anis"
							  -sqrt2*omp2*Lambda*(Cu[pos(i,j)]+temp)		// the 5th term
							  +Eps4*anis)/((1.+Eps4*c4)*(1.+Eps4*c4)*Tau); 	// 1st + 2nd(incomplete) + 3rd terms
#elif(NewP==2)	// isotropic at the 2nd order
		REAL phx=(Pcurr[pos(i+1,j)]-Pcurr[pos(i-1,j)])/(2.*dx);
		REAL phy=(Pcurr[pos(i,j+1)]-Pcurr[pos(i,j-1)])/(2.*dx);
		REAL NormG2=phx*phx+phy*phy;

		phxx=(Pcurr[pos(i+1,j)]-2.*psi+Pcurr[pos(i-1,j)])/(dx*dx);
		phyy=(Pcurr[pos(i,j+1)]-2.*psi+Pcurr[pos(i,j-1)])/(dx*dx);
		phxy=(Pcurr[pos(i+1,j+1)]-Pcurr[pos(i-1,j+1)]-Pcurr[pos(i+1,j-1)]+Pcurr[pos(i-1,j-1)])/(4.*dx*dx);			

		// ----------
		// Anisotropy
		// ----------				
		if(fabs(omp2)>=1.e-6 && fabs(NormG2)>1.e-15) 
		{
			if (!pha && fabs(omp2)>=.01)
			{
				int sum=Pha[pos(i+1,j)]+Pha[pos(i-1,j)]+Pha[pos(i,j+1)]+Pha[pos(i,j-1)]
				+Pha[pos(i+1,j+1)]+Pha[pos(i+1,j-1)]+Pha[pos(i-1,j+1)]+Pha[pos(i-1,j-1)];
				if (sum)	
				Pha[pos(i,j)]=pha=(sum>0)-(sum<0);
			}
			
			if (pha)
			{
				pha=(1+pha)/2;	// From now pha is 0 or 1 (not -1 or 1)
				
				dphix=phx*cAlpha[pha]+phy*sAlpha[pha];
				dphiy=-phx*sAlpha[pha]+phy*cAlpha[pha];
				
				dphixx=phyy*sAlpha2[pha]+phxy*s2Alpha[pha]+phxx*cAlpha2[pha];
				dphiyy=phyy*cAlpha2[pha]-phxy*s2Alpha[pha]+phxx*sAlpha2[pha];
				dphixy=phyy*cAlphasAlpha[pha]+phxy*c2Alpha[pha]-phxx*cAlphasAlpha[pha];
				NormG4=NormG2*NormG2;
				thx=(dphix*dphixy-dphiy*dphixx)/NormG2;
				thy=(dphix*dphiyy-dphiy*dphixy)/NormG2;
				c4=-8.*dphix*dphix*dphiy*dphiy/NormG4+1.;
				s4=4.*(dphix*dphix*dphix*dphiy-dphiy*dphiy*dphiy*dphix)/NormG4;
				anis=c4*(2.+Eps4*c4)*(phxx+phyy - sqrt2*phi*NormG2 )  		// 2nd term (incomplete)
				-8.*s4*(1.+Eps4*c4)*(thx*dphix+thy*dphiy) 					// 1st term
				-16.*(c4+Eps4*(c4*c4-s4*s4))*(thy*dphix-thx*dphiy); 		// 3rd term
			}
			else
			{
				anis=0.;
				c4=0.;
			}
			
#if(NOISE!=WITHOUT)
			curandState localState;
			localState=state[pos(i,j)];
			REAL ran1=curand_uniform_double(&localState);
#if(NOISE==GAUSSIAN)
			REAL ran2=curand_uniform_double(&localState);
			Pnext[pos(i,j)] += Fnoise*sqrt(dt)*sqrt(-2.0*log(1.0-ran1))*cos(2.0*PI*ran2);
#endif
#if(NOISE==FLAT)
			Pnext[pos(i,j)] += Fnoise*sqrt(dt)*(ran1-.5);
#endif
			state[pos(i,j)]=localState;
#endif
			
		}
		else
		{
			anis=0.;
			c4=0.;
		}
		
		// Invariant Laplacian
		REAL Lap9 = 2.*(Pcurr[pos(i+1,j)]+Pcurr[pos(i-1,j)]+Pcurr[pos(i,j+1)]+Pcurr[pos(i,j-1)])/3.			// the <10> lattice shell
					+ (Pcurr[pos(i+1,j+1)]+Pcurr[pos(i-1,j+1)]+Pcurr[pos(i+1,j-1)]+Pcurr[pos(i-1,j-1)])/6.	// the <11> lattice
					- 10.*psi/3.;
		Lap9 = Lap9/(dx*dx);			

		// Invariant Gradient
		NormG2 = ( (Pcurr[pos(i+1,j)]-Pcurr[pos(i-1,j)])*(Pcurr[pos(i+1,j)]-Pcurr[pos(i-1,j)])+(Pcurr[pos(i,j+1)]-Pcurr[pos(i,j-1)])*(Pcurr[pos(i,j+1)]-Pcurr[pos(i,j-1)]) )/6.					// the <10> lattice shell
				+( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(i-1,j-1)])*(Pcurr[pos(i+1,j+1)]-Pcurr[pos(i-1,j-1)])+(Pcurr[pos(i+1,j-1)]-Pcurr[pos(i-1,j+1)])*(Pcurr[pos(i+1,j-1)]-Pcurr[pos(i-1,j+1)]) )/24.;	// the <11> lattice shell
		NormG2 = NormG2/(dx*dx);

		// ------
		// dPsidt
		// ------
    
    #if(IfTemporal)    
		// temporal update
		temp=(i*dx +0.5*j +xoffs-xint+xinit*lT-Vp*dt*iter)/lT;
	#else    
        // original
		//temp=(i*dx+xoffs-x0-Vp*dt*iter)/lT;
        temp=(i*dx+xoffs-x00-Vp*dt*iter)/lT+U_off;
	#endif				
        
		//Tau=(temp>1.)? kcoeff : (1.-omk*temp);
		Tau = (Cu[pos(i,j)] > 1.)? kcoeff : (1. + omk * Cu[pos(i,j)]);
		Pnext[pos(i,j)]+= dt*(Lap9 + sqrt2*phi*(1.-NormG2) 					// the invariant discretization
							  // +sqrt2*phi*(1.-NormG2) 						// 4th term + the remaining 2nd term when computing "anis"
							  -sqrt2*omp2*Lambda*(Cu[pos(i,j)]+temp)		// the 5th term
							  +Eps4*anis)/((1.+Eps4*c4)*(1.+Eps4*c4)*Tau); 	// 1st + 2nd(incomplete) + 3rd terms
#else	// the old variant compute_P
		REAL phx=(Pcurr[pos(i+1,j)]-Pcurr[pos(i-1,j)])/(2.*dx);
		REAL phy=(Pcurr[pos(i,j+1)]-Pcurr[pos(i,j-1)])/(2.*dx);
		REAL NormG2=phx*phx+phy*phy;

		phxx=(Pcurr[pos(i+1,j)]-2.*psi+Pcurr[pos(i-1,j)])/(dx*dx);
		phyy=(Pcurr[pos(i,j+1)]-2.*psi+Pcurr[pos(i,j-1)])/(dx*dx);
		phxy=(Pcurr[pos(i+1,j+1)]-Pcurr[pos(i-1,j+1)]-Pcurr[pos(i+1,j-1)]+Pcurr[pos(i-1,j-1)])/(4.*dx*dx);			

		// ----------
		// Anisotropy
		// ----------				
		if(fabs(omp2)>=1.e-6 && fabs(NormG2)>1.e-15) 
		{
			// if (!pha && fabs(omp2)>=.01)
			// {
			// 	int sum=Pha[pos(i+1,j)]+Pha[pos(i-1,j)]+Pha[pos(i,j+1)]+Pha[pos(i,j-1)]
			// 	+Pha[pos(i+1,j+1)]+Pha[pos(i+1,j-1)]+Pha[pos(i-1,j+1)]+Pha[pos(i-1,j-1)];
			// 	if (sum)	
			// 	Pha[pos(i,j)]=pha=(sum>0)-(sum<0);
			// }
			
			// if (pha)
			{
				pha=(1+pha)/2;	// From now pha is 0 or 1 (not -1 or 1)
				
				dphix=phx*cAlpha[pha]+phy*sAlpha[pha];
				dphiy=-phx*sAlpha[pha]+phy*cAlpha[pha];
				
				dphixx=phyy*sAlpha2[pha]+phxy*s2Alpha[pha]+phxx*cAlpha2[pha];
				dphiyy=phyy*cAlpha2[pha]-phxy*s2Alpha[pha]+phxx*sAlpha2[pha];
				dphixy=phyy*cAlphasAlpha[pha]+phxy*c2Alpha[pha]-phxx*cAlphasAlpha[pha];
				NormG4=NormG2*NormG2;
				thx=(dphix*dphixy-dphiy*dphixx)/NormG2;
				thy=(dphix*dphiyy-dphiy*dphixy)/NormG2;
				c4=-8.*dphix*dphix*dphiy*dphiy/NormG4+1.;
				s4=4.*(dphix*dphix*dphix*dphiy-dphiy*dphiy*dphiy*dphix)/NormG4;
				anis=c4*(2.+Eps4*c4)*(phxx+phyy - sqrt2*phi*NormG2 )  		// 2nd term (incomplete)
				-8.*s4*(1.+Eps4*c4)*(thx*dphix+thy*dphiy) 					// 1st term
				-16.*(c4+Eps4*(c4*c4-s4*s4))*(thy*dphix-thx*dphiy); 		// 3rd term
			}
			// else
			// {
			// 	anis=0.;
			// 	c4=0.;
			// }
			
#if(NOISE!=WITHOUT)
			curandState localState;
			localState=state[pos(i,j)];
			REAL ran1=curand_uniform_double(&localState);
#if(NOISE==GAUSSIAN)
			REAL ran2=curand_uniform_double(&localState);
			Pnext[pos(i,j)] += Fnoise*sqrt(dt)*sqrt(-2.0*log(1.0-ran1))*cos(2.0*PI*ran2);
#endif
#if(NOISE==FLAT)
			Pnext[pos(i,j)] += Fnoise*sqrt(dt)*(ran1-.5);
#endif
			state[pos(i,j)]=localState;
#endif
			
		}
		else
		{
			anis=0.;
			c4=0.;
		}
		
		// ------
		// dPsidt
		// ------
    
    #if(IfTemporal)    
		// temporal update
		temp=(i*dx +0.5*j +xoffs-xint+xinit*lT-Vp*dt*iter)/lT;
	#else    
        // original
        temp=(i*dx+xoffs-x00-Vp*dt*iter)/lT+U_off;
		//temp=(i*dx+xoffs-x0-Vp*dt*iter)/lT;
	#endif				
        
		//Tau=(temp>1.)? kcoeff : (1.-omk*temp);
		Tau = (Cu[pos(i,j)] > 1.)? kcoeff : (1. + omk * Cu[pos(i,j)]);
		Pnext[pos(i,j)]+= dt*(phxx+phyy 									// the remaining 2nd term when computing "anis"
							  +sqrt2*phi*(1.-NormG2) 						// 4th term + the remaining 2nd term when computing "anis"
							  -sqrt2*omp2*Lambda*(Cu[pos(i,j)]+temp)		// the 5th term
							  +Eps4*anis)/((1.+Eps4*c4)*(1.+Eps4*c4)*Tau); 	// 1st + 2nd(incomplete) + 3rd terms
#endif		
		
		// BC at (x=0.5) or (x=Nx+0.5)
		if(i==1)
		{
			Pnext[pos(IMIN,j)]=Pnext[pos(1,j)];
			F[pos(IMIN,j)]=F[pos(1,j)];
			Pha[pos(IMIN,j)]=Pha[pos(1,j)];
			Cu[pos(IMIN,j)]=Cu[pos(1,j)];			
			if(j==1)
			{
				Pnext[pos(IMIN,JMIN)]=Pnext[pos(1,1)];
				F[pos(IMIN,JMIN)]=F[pos(1,1)];
				Pha[pos(IMIN,JMIN)]=Pha[pos(1,1)];
				Cu[pos(IMIN,JMIN)]=Cu[pos(1,1)];
			}
			else if(j==Ny)
			{
				Pnext[pos(IMIN,JMAX)]=Pnext[pos(1,Ny)];
				F[pos(IMIN,JMAX)]=F[pos(1,Ny)];
				Pha[pos(IMIN,JMAX)]=Pha[pos(1,Ny)];
				Cu[pos(IMIN,JMAX)]=Cu[pos(1,Ny)];
			}			
		}
		else if(i==Nx)
		{
			Pnext[pos(IMAX,j)]=Pnext[pos(Nx,j)];
			F[pos(IMAX,j)]=F[pos(Nx,j)];
			Pha[pos(IMAX,j)]=Pha[pos(Nx,j)];
			Cu[pos(IMAX,j)]=Cu[pos(Nx,j)];
			if(j==1)
			{
				Pnext[pos(IMAX,JMIN)]=Pnext[pos(Nx,1)];
				F[pos(IMAX,JMIN)]=F[pos(Nx,1)];
				Pha[pos(IMAX,JMIN)]=Pha[pos(Nx,1)];
				Cu[pos(IMAX,JMIN)]=Cu[pos(Nx,1)];
			}
			else if(j==Ny)
			{
				Pnext[pos(IMAX,JMAX)]=Pnext[pos(Nx,Ny)];
				F[pos(IMAX,JMAX)]=F[pos(Nx,Ny)];
				Pha[pos(IMAX,JMAX)]=Pha[pos(Nx,Ny)];
				Cu[pos(IMAX,JMAX)]=Cu[pos(Nx,Ny)];
			}			
		}
		
		// BC at (y=0.5) or (y=Ny+0.5)
		if(j==1)
		{
			Pnext[pos(i,JMIN)]=Pnext[pos(i,1)];
			F[pos(i,JMIN)]=F[pos(i,1)];
			Pha[pos(i,JMIN)]=Pha[pos(i,1)];
			Cu[pos(i,JMIN)]=Cu[pos(i,1)];
		}
		else if(j==Ny)
		{
			Pnext[pos(i,JMAX)]=Pnext[pos(i,Ny)];
			F[pos(i,JMAX)]=F[pos(i,Ny)];
			Pha[pos(i,JMAX)]=Pha[pos(i,Ny)];
			Cu[pos(i,JMAX)]=Cu[pos(i,Ny)];
		}
	}
}

__global__ void Compute_U(REAL *Cucurr,REAL *Cunext,REAL *Pcurr,REAL *Pnext,REAL *F,Constants *Param)
{

#if(NewU==1)	// the new Compute_U

// pay attention to here djxr... are not declared
#define	djxr	dpxr 
#define	djxl	dpxl
#define	djyu	dpyu
#define	djyd	dpyd

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(i!=0 && i!=Nx+1 && j!=0 && j!=Ny+1)
	{
		REAL S00,S10,Sm10,S01,S0m1,S11,Sm11,S1m1,Sm1m1,Shh,Shmh,Smhh,Smhmh;
		REAL dpxr,dnormr,dpxl,dnorml,dpyu,dnormu,dpyd,dnormd;
		REAL dpx1,dpy1,dpx2,dpy2,dpx3,dpy3,dpxSum,dpySum;
		REAL src,srcd,DivompGradU,DivompGradUd;	
		
		REAL dpdt=(Pnext[pos(i,j)]-Pcurr[pos(i,j)])/dt;
		REAL u=Cucurr[pos(i,j)];
		REAL phi=F[pos(i,j)];
		REAL omp2=1.-phi*phi;

		// The scalar field in the anti-trapping term
		S00 = omp2*dpdt*(1.+omk*u)/2.;
		S10 = (1.-F[pos(i+1,j)]*F[pos(i+1,j)])*(Pnext[pos(i+1,j)]-Pcurr[pos(i+1,j)])*(1.+omk*Cucurr[pos(i+1,j)])/(2.*dt);
		Sm10= (1.-F[pos(i-1,j)]*F[pos(i-1,j)])*(Pnext[pos(i-1,j)]-Pcurr[pos(i-1,j)])*(1.+omk*Cucurr[pos(i-1,j)])/(2.*dt);
		S01 = (1.-F[pos(i,j+1)]*F[pos(i,j+1)])*(Pnext[pos(i,j+1)]-Pcurr[pos(i,j+1)])*(1.+omk*Cucurr[pos(i,j+1)])/(2.*dt);
		S0m1= (1.-F[pos(i,j-1)]*F[pos(i,j-1)])*(Pnext[pos(i,j-1)]-Pcurr[pos(i,j-1)])*(1.+omk*Cucurr[pos(i,j-1)])/(2.*dt);
		S11 = (1.-F[pos(i+1,j+1)]*F[pos(i+1,j+1)])*(Pnext[pos(i+1,j+1)]-Pcurr[pos(i+1,j+1)])*(1.+omk*Cucurr[pos(i+1,j+1)])/(2.*dt);
		Sm11= (1.-F[pos(i-1,j+1)]*F[pos(i-1,j+1)])*(Pnext[pos(i-1,j+1)]-Pcurr[pos(i-1,j+1)])*(1.+omk*Cucurr[pos(i-1,j+1)])/(2.*dt);
		S1m1= (1.-F[pos(i+1,j-1)]*F[pos(i+1,j-1)])*(Pnext[pos(i+1,j-1)]-Pcurr[pos(i+1,j-1)])*(1.+omk*Cucurr[pos(i+1,j-1)])/(2.*dt);
		Sm1m1=(1.-F[pos(i-1,j-1)]*F[pos(i-1,j-1)])*(Pnext[pos(i-1,j-1)]-Pcurr[pos(i-1,j-1)])*(1.+omk*Cucurr[pos(i-1,j-1)])/(2.*dt);

		// need to use i pm 2 and j pm 2
		int ip2=i+2;
		int im2=i-2;
		int jp2=j+2;
		int jm2=j-2;
		if(i==1) 	im2=2;
		if(i==Nx) 	ip2=Nx-1;
	#if(BOUND_COND==NOFLUX)	
		// if(j==1) 	jm2=(SYMMETRY_Y0)?2:0;
		// if(j==Ny)	jp2=Ny+1;
		if(j==1) 	jm2=2;
		if(j==Ny)	jp2=Ny-1;		
	#elif(BOUND_COND==PERIODIC)
		if(j==1) 	jm2=Ny-1;
		if(j==Ny)	jp2=2;	
	#endif
		
		// -----------------------------------------------
		// src <- div[(dphi/dt)gradphi/|gradphi|)]/sqrt(2)
		// i.e. the source term for the diffusion field
		// -----------------------------------------------
		if(fabs(omp2)>=1.e-6) 
		{				
			// The unit gradient may be computed either by Grad(h_Psi) or Grad(h_Phi), but Psi field is more stable computationally.
			dpxr=Pcurr[pos(i+1,j)]-Pcurr[pos(i,j)] ;
			dpxl=Pcurr[pos(i,j)]-Pcurr[pos(i-1,j)] ;
			dpyu=Pcurr[pos(i,j+1)]-Pcurr[pos(i,j)] ;
			dpyd=Pcurr[pos(i,j)]-Pcurr[pos(i,j-1)] ;

			dpx1 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(i,j-1)])-(Pcurr[pos(i,j+1)]-Pcurr[pos(i+1,j-1)]) )/2.;
			dpy1 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(i,j-1)])+(Pcurr[pos(i,j+1)]-Pcurr[pos(i+1,j-1)]) )/4.;
			dpx2 = ( (Pcurr[pos(ip2,j+1)]-Pcurr[pos(i-1,j-1)])-(Pcurr[pos(i-1,j+1)]-Pcurr[pos(ip2,j-1)]) )/6.;
			dpy2 = ( (Pcurr[pos(ip2,j+1)]-Pcurr[pos(i-1,j-1)])+(Pcurr[pos(i-1,j+1)]-Pcurr[pos(ip2,j-1)]) )/4.;
			dpx3 = ( (Pcurr[pos(i+1,jp2)]-Pcurr[pos(i,jm2)])-(Pcurr[pos(i,jp2)]-Pcurr[pos(i+1,jm2)]) )/2.;
			dpy3 = ( (Pcurr[pos(i+1,jp2)]-Pcurr[pos(i,jm2)])+(Pcurr[pos(i,jp2)]-Pcurr[pos(i+1,jm2)]) )/8.;
			dpxSum = 35.*dpx1/24.-dpx2/8.-dpx3/3.;
			dpySum = 35.*dpy1/24.-dpy2/8.-dpy3/3.;
			dnormr=sqrt(dpxSum*dpxSum+dpySum*dpySum);

			dpx1 = ( (Pcurr[pos(i,j+1)]-Pcurr[pos(i-1,j-1)])-(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i,j-1)]) )/2.;
			dpy1 = ( (Pcurr[pos(i,j+1)]-Pcurr[pos(i-1,j-1)])+(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i,j-1)]) )/4.;
			dpx2 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(im2,j-1)])-(Pcurr[pos(im2,j+1)]-Pcurr[pos(i+1,j-1)]) )/6.;
			dpy2 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(im2,j-1)])+(Pcurr[pos(im2,j+1)]-Pcurr[pos(i+1,j-1)]) )/4.;
			dpx3 = ( (Pcurr[pos(i,jp2)]-Pcurr[pos(i-1,jm2)])-(Pcurr[pos(i-1,jp2)]-Pcurr[pos(i,jm2)]) )/2.;
			dpy3 = ( (Pcurr[pos(i,jp2)]-Pcurr[pos(i-1,jm2)])+(Pcurr[pos(i-1,jp2)]-Pcurr[pos(i,jm2)]) )/8.;
			dpxSum = 35.*dpx1/24.-dpx2/8.-dpx3/3.;
			dpySum = 35.*dpy1/24.-dpy2/8.-dpy3/3.;
			dnorml=sqrt(dpxSum*dpxSum+dpySum*dpySum);			

			dpy1 = ( (Pcurr[pos(i-1,j+1)]-Pcurr[pos(i+1,j)])-(Pcurr[pos(i-1,j)]-Pcurr[pos(i+1,j+1)]) )/2.;
			dpx1 = -((Pcurr[pos(i-1,j+1)]-Pcurr[pos(i+1,j)])+(Pcurr[pos(i-1,j)]-Pcurr[pos(i+1,j+1)]) )/4.;
			dpy2 = ( (Pcurr[pos(i-1,jp2)]-Pcurr[pos(i+1,j-1)])-(Pcurr[pos(i-1,j-1)]-Pcurr[pos(i+1,jp2)]) )/6.;
			dpx2 = -((Pcurr[pos(i-1,jp2)]-Pcurr[pos(i+1,j-1)])+(Pcurr[pos(i-1,j-1)]-Pcurr[pos(i+1,jp2)]) )/4.;
			dpy3 = ( (Pcurr[pos(im2,j+1)]-Pcurr[pos(ip2,j)])-(Pcurr[pos(im2,j)]-Pcurr[pos(ip2,j+1)]) )/2.;
			dpx3 = -((Pcurr[pos(im2,j+1)]-Pcurr[pos(ip2,j)])+(Pcurr[pos(im2,j)]-Pcurr[pos(ip2,j+1)]) )/8.;
			dpySum = 35.*dpy1/24.-dpy2/8.-dpy3/3.;
			dpxSum = 35.*dpx1/24.-dpx2/8.-dpx3/3.;
			dnormu=sqrt(dpxSum*dpxSum+dpySum*dpySum);

			dpy1 = ( (Pcurr[pos(i-1,j)]-Pcurr[pos(i+1,j-1)])-(Pcurr[pos(i-1,j-1)]-Pcurr[pos(i+1,j)]) )/2.;
			dpx1 = -((Pcurr[pos(i-1,j)]-Pcurr[pos(i+1,j-1)])+(Pcurr[pos(i-1,j-1)]-Pcurr[pos(i+1,j)]) )/4.;
			dpy2 = ( (Pcurr[pos(i-1,j+1)]-Pcurr[pos(i+1,jm2)])-(Pcurr[pos(i-1,jm2)]-Pcurr[pos(i+1,j+1)]) )/6.;
			dpx2 = -((Pcurr[pos(i-1,j+1)]-Pcurr[pos(i+1,jm2)])+(Pcurr[pos(i-1,jm2)]-Pcurr[pos(i+1,j+1)]) )/4.;
			dpy3 = ( (Pcurr[pos(im2,j)]-Pcurr[pos(ip2,j-1)])-(Pcurr[pos(im2,j-1)]-Pcurr[pos(ip2,j)]) )/2.;
			dpx3 = -((Pcurr[pos(im2,j)]-Pcurr[pos(ip2,j-1)])+(Pcurr[pos(im2,j-1)]-Pcurr[pos(ip2,j)]) )/8.;
			dpySum = 35.*dpy1/24.-dpy2/8.-dpy3/3.;
			dpxSum = 35.*dpx1/24.-dpx2/8.-dpx3/3.;
			dnormd=sqrt(dpxSum*dpxSum+dpySum*dpySum);						

			//if((dnormr*dnorml*dnormu*dnormd)>1.e-15)
			{
				// Shh is the scalar fields at (i+1/2,j+1/2). Smhmh is the scalar fields at (i-1/2,j-1/2) and so on.
				Shh = (S11+S10+S01+S00)/4.;
				Shmh= (S10+S00+S1m1+S0m1)/4.;
				Smhh= (Sm11+S01+Sm10+S00)/4.;
				Smhmh=(Sm10+Sm1m1+S0m1+S00)/4.;

				djxr=(dnormr>=1.e-6)? (S10+S00+Shh+Shmh)*dpxr/(4.*dnormr) : 0.;
				djxl=(dnorml>=1.e-6)? (S00+Sm10+Smhh+Smhmh)*dpxl/(4.*dnorml) : 0.;
				djyu=(dnormu>=1.e-6)? (S01+S00+Shh+Smhh)*dpyu/(4.*dnormu) : 0.;
				djyd=(dnormd>=1.e-6)? (S00+S0m1+Shmh+Smhmh)*dpyd/(4.*dnormd) : 0.;

				src=(djxr-djxl+djyu-djyd)/dx;
			}
			//else
			//{
			//	src=0.;
			//}
		}
		else
		{
			src=0.;
		}

		// The diagonal contribution to the source term
		if(fabs(omp2)>=1.e-6) 
		{				
			// The unit gradient may be computed either by Grad(h_Psi) or Grad(h_Phi), but Psi field is more stable computationally.
			dpxr=(Pcurr[pos(i+1,j+1)]-Pcurr[pos(i,j)])/sqrt2;
			dpxl=(Pcurr[pos(i,j)]-Pcurr[pos(i-1,j-1)])/sqrt2;
			dpyu=(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i,j)])/sqrt2;
			dpyd=(Pcurr[pos(i,j)]-Pcurr[pos(i+1,j-1)])/sqrt2;

			dpx1 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(i,j)])-(Pcurr[pos(i,j+1)]-Pcurr[pos(i+1,j)]) )/2.;
			dpy1 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(i,j)])+(Pcurr[pos(i,j+1)]-Pcurr[pos(i+1,j)]) )/2.;
			dpx2 = ( (Pcurr[pos(ip2,j+1)]-Pcurr[pos(i-1,j)])-(Pcurr[pos(i-1,j+1)]-Pcurr[pos(ip2,j)]) )/6.;
			dpy2 = ( (Pcurr[pos(ip2,j+1)]-Pcurr[pos(i-1,j)])+(Pcurr[pos(i-1,j+1)]-Pcurr[pos(ip2,j)]) )/2.;
			dpx3 = ( (Pcurr[pos(i+1,jp2)]-Pcurr[pos(i,j-1)])-(Pcurr[pos(i,jp2)]-Pcurr[pos(i+1,j-1)]) )/2.;
			dpy3 = ( (Pcurr[pos(i+1,jp2)]-Pcurr[pos(i,j-1)])+(Pcurr[pos(i,jp2)]-Pcurr[pos(i+1,j-1)]) )/6.;
			dpxSum = 5.*dpx1/4.-dpx2/8.-dpx3/8.;
			dpySum = 5.*dpy1/4.-dpy2/8.-dpy3/8.;
			dnormr=sqrt(dpxSum*dpxSum+dpySum*dpySum);

			dpx1 = ( (Pcurr[pos(i,j+1)]-Pcurr[pos(i-1,j)])-(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i,j)]) )/2.;
			dpy1 = ( (Pcurr[pos(i,j+1)]-Pcurr[pos(i-1,j)])+(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i,j)]) )/2.;
			dpx2 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(im2,j)])-(Pcurr[pos(im2,j+1)]-Pcurr[pos(i+1,j)]) )/6.;
			dpy2 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(im2,j)])+(Pcurr[pos(im2,j+1)]-Pcurr[pos(i+1,j)]) )/2.;
			dpx3 = ( (Pcurr[pos(i,jp2)]-Pcurr[pos(i-1,j-1)])-(Pcurr[pos(i-1,jp2)]-Pcurr[pos(i,j-1)]) )/2.;
			dpy3 = ( (Pcurr[pos(i,jp2)]-Pcurr[pos(i-1,j-1)])+(Pcurr[pos(i-1,jp2)]-Pcurr[pos(i,j-1)]) )/6.;
			dpxSum = 5.*dpx1/4.-dpx2/8.-dpx3/8.;
			dpySum = 5.*dpy1/4.-dpy2/8.-dpy3/8.;
			dnormu=sqrt(dpxSum*dpxSum+dpySum*dpySum);

			dpx1 = ( (Pcurr[pos(i+1,j)]-Pcurr[pos(i,j-1)])-(Pcurr[pos(i,j)]-Pcurr[pos(i+1,j-1)]) )/2.;
			dpy1 = ( (Pcurr[pos(i+1,j)]-Pcurr[pos(i,j-1)])+(Pcurr[pos(i,j)]-Pcurr[pos(i+1,j-1)]) )/2.;
			dpx2 = ( (Pcurr[pos(ip2,j)]-Pcurr[pos(i-1,j-1)])-(Pcurr[pos(i-1,j)]-Pcurr[pos(ip2,j-1)]) )/6.;
			dpy2 = ( (Pcurr[pos(ip2,j)]-Pcurr[pos(i-1,j-1)])+(Pcurr[pos(i-1,j)]-Pcurr[pos(ip2,j-1)]) )/2.;
			dpx3 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(i,jm2)])-(Pcurr[pos(i,j+1)]-Pcurr[pos(i+1,jm2)]) )/2.;
			dpy3 = ( (Pcurr[pos(i+1,j+1)]-Pcurr[pos(i,jm2)])+(Pcurr[pos(i,j+1)]-Pcurr[pos(i+1,jm2)]) )/6.;
			dpxSum = 5.*dpx1/4.-dpx2/8.-dpx3/8.;
			dpySum = 5.*dpy1/4.-dpy2/8.-dpy3/8.;
			dnormd=sqrt(dpxSum*dpxSum+dpySum*dpySum);

			dpx1 = ( (Pcurr[pos(i,j)]-Pcurr[pos(i-1,j-1)])-(Pcurr[pos(i-1,j)]-Pcurr[pos(i,j-1)]) )/2.;
			dpy1 = ( (Pcurr[pos(i,j)]-Pcurr[pos(i-1,j-1)])+(Pcurr[pos(i-1,j)]-Pcurr[pos(i,j-1)]) )/2.;
			dpx2 = ( (Pcurr[pos(i+1,j)]-Pcurr[pos(im2,j-1)])-(Pcurr[pos(im2,j)]-Pcurr[pos(i+1,j-1)]) )/6.;
			dpy2 = ( (Pcurr[pos(i+1,j)]-Pcurr[pos(im2,j-1)])+(Pcurr[pos(im2,j)]-Pcurr[pos(i+1,j-1)]) )/2.;
			dpx3 = ( (Pcurr[pos(i,j+1)]-Pcurr[pos(i-1,jm2)])-(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i,jm2)]) )/2.;
			dpy3 = ( (Pcurr[pos(i,j+1)]-Pcurr[pos(i-1,jm2)])+(Pcurr[pos(i-1,j+1)]-Pcurr[pos(i,jm2)]) )/6.;
			dpxSum = 5.*dpx1/4.-dpx2/8.-dpx3/8.;
			dpySum = 5.*dpy1/4.-dpy2/8.-dpy3/8.;
			dnorml=sqrt(dpxSum*dpxSum+dpySum*dpySum);

			//if((dnormr*dnorml*dnormu*dnormd)>1.e-15)
			{
				djxr=(dnormr>=1.e-6)? (S11+S00+S01+S10)*dpxr/(4.*dnormr) : 0.;
				djxl=(dnorml>=1.e-6)? (S00+Sm1m1+Sm10+S0m1)*dpxl/(4.*dnorml) : 0.;
				djyu=(dnormu>=1.e-6)? (Sm11+S00+S01+Sm10)*dpyu/(4.*dnormu) : 0.;
				djyd=(dnormd>=1.e-6)? (S00+S1m1+S10+S0m1)*dpyd/(4.*dnormd) : 0.;

				srcd=(djxr-djxl+djyu-djyd)/sqrt2/dx;
			}
			//else
			//{
			//	srcd=0.;
			//}
		}
		else
		{
			srcd=0.;
		}
		
		// The scalar field in the diffusion term
		S00 = 1.-phi;
		S10 = (1.-F[pos(i+1,j)]);
		Sm10= (1.-F[pos(i-1,j)]);
		S01 = (1.-F[pos(i,j+1)]);
		S0m1= (1.-F[pos(i,j-1)]);
		S11 = (1.-F[pos(i+1,j+1)]);
		Sm11= (1.-F[pos(i-1,j+1)]);
		S1m1= (1.-F[pos(i+1,j-1)]);
		Sm1m1=(1.-F[pos(i-1,j-1)]);

		// -----------------------------------
		// DivompGradU <- Div((1-h_Phi)*Grad(h_U))
		// -----------------------------------							
		if((1.-phi)>=1.e-6) 
		{
			// Shh is the scalar fields at (i+1/2,j+1/2). Smhmh is the scalar fields at (i-1/2,j-1/2) and so on.
			Shh = (S11+S10+S01+S00)/4.;
			Shmh= (S10+S00+S1m1+S0m1)/4.;
			Smhh= (Sm11+S01+Sm10+S00)/4.;
			Smhmh=(Sm10+Sm1m1+S0m1+S00)/4.;

			djxr=(S10+S00+Shh+Shmh)*(Cucurr[pos(i+1,j)]-u)/(4.*dx);
			djxl=(S00+Sm10+Smhh+Smhmh)*(u-Cucurr[pos(i-1,j)])/(4.*dx);
			djyu=(S01+S00+Shh+Smhh)*(Cucurr[pos(i,j+1)]-u)/(4.*dx);
			djyd=(S00+S0m1+Shmh+Smhmh)*(u-Cucurr[pos(i,j-1)])/(4.*dx);

			DivompGradU=(djxr-djxl+djyu-djyd)/dx;			
		}
		else
		{
			DivompGradU=0.;
		}

		// The diagonal contribution
		if((1.-phi)>=1.e-6) 
		{
			djxr=(S11+S00+S01+S10)*(Cucurr[pos(i+1,j+1)]-u)/(4.*sqrt2*dx);
			djxl=(S00+Sm1m1+Sm10+S0m1)*(u-Cucurr[pos(i-1,j-1)])/(4.*sqrt2*dx);
			djyu=(Sm11+S00+S01+Sm10)*(Cucurr[pos(i-1,j+1)]-u)/(4.*sqrt2*dx);
			djyd=(S00+S1m1+S10+S0m1)*(u-Cucurr[pos(i+1,j-1)])/(4.*sqrt2*dx);

			DivompGradUd=(djxr-djxl+djyu-djyd)/sqrt2/dx;			
		}
		else
		{
			DivompGradUd=0.;
		}

		
		// ----
		// h_dUdt
		// ----		
		REAL DivJ = D*(2.*DivompGradU/3.+DivompGradUd/3.)+(2.*src/3.+srcd/3.) ;
		REAL phinext = tanh(Pnext[pos(i,j)]/sqrt2);

		//Cunext[pos(i,j)]=Cucurr[pos(i,j)]+dt*(D*(2.*DivompGradU/3.+DivompGradUd/3.)+(2.*src/3.+srcd/3.)+(1.+omk*u)*omp2/sqrt2*dpdt)/(opk-omk*phi) ;
		Cunext[pos(i,j)]=( DivJ*dt+(Cucurr[pos(i,j)]+1./omk)*(opk-omk*phi) )/(opk-omk*phinext) - 1./omk;
	}
#elif(NewU==2)	// New Compute_U with approximation
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(i!=0 && i!=Nx+1 && j!=0 && j!=Ny+1)
	{
		REAL S00,S10,Sm10,S01,S0m1,S11,Sm11,S1m1,Sm1m1,Shh,Shmh,Smhh,Smhmh;
		REAL djxr,djxl,djyu,djyd;
		REAL src,srcd,DivompGradU,DivompGradUd;	
		
		REAL dpdt=(Pnext[pos(i,j)]-Pcurr[pos(i,j)])/dt;
		REAL u=Cucurr[pos(i,j)];
		REAL phi=F[pos(i,j)];
		REAL psi=Pcurr[pos(i,j)];
		REAL omp2=1.-phi*phi;

		// The scalar field in the anti-trapping term
		S00 = omp2*dpdt*(1.+omk*u)/2.;
		S10 = (1.-F[pos(i+1,j)]*F[pos(i+1,j)])*(Pnext[pos(i+1,j)]-Pcurr[pos(i+1,j)])*(1.+omk*Cucurr[pos(i+1,j)])/(2.*dt);
		Sm10= (1.-F[pos(i-1,j)]*F[pos(i-1,j)])*(Pnext[pos(i-1,j)]-Pcurr[pos(i-1,j)])*(1.+omk*Cucurr[pos(i-1,j)])/(2.*dt);
		S01 = (1.-F[pos(i,j+1)]*F[pos(i,j+1)])*(Pnext[pos(i,j+1)]-Pcurr[pos(i,j+1)])*(1.+omk*Cucurr[pos(i,j+1)])/(2.*dt);
		S0m1= (1.-F[pos(i,j-1)]*F[pos(i,j-1)])*(Pnext[pos(i,j-1)]-Pcurr[pos(i,j-1)])*(1.+omk*Cucurr[pos(i,j-1)])/(2.*dt);
		S11 = (1.-F[pos(i+1,j+1)]*F[pos(i+1,j+1)])*(Pnext[pos(i+1,j+1)]-Pcurr[pos(i+1,j+1)])*(1.+omk*Cucurr[pos(i+1,j+1)])/(2.*dt);
		Sm11= (1.-F[pos(i-1,j+1)]*F[pos(i-1,j+1)])*(Pnext[pos(i-1,j+1)]-Pcurr[pos(i-1,j+1)])*(1.+omk*Cucurr[pos(i-1,j+1)])/(2.*dt);
		S1m1= (1.-F[pos(i+1,j-1)]*F[pos(i+1,j-1)])*(Pnext[pos(i+1,j-1)]-Pcurr[pos(i+1,j-1)])*(1.+omk*Cucurr[pos(i+1,j-1)])/(2.*dt);
		Sm1m1=(1.-F[pos(i-1,j-1)]*F[pos(i-1,j-1)])*(Pnext[pos(i-1,j-1)]-Pcurr[pos(i-1,j-1)])*(1.+omk*Cucurr[pos(i-1,j-1)])/(2.*dt);
		
		// -----------------------------------------------
		// src <- div[(dphi/dt)gradphi/|gradphi|)]/sqrt(2)
		// i.e. the source term for the diffusion field
		// -----------------------------------------------
		if((1.-phi)>=1.e-6) 
		{
			// Shh is the scalar fields at (i+1/2,j+1/2). Smhmh is the scalar fields at (i-1/2,j-1/2) and so on.
			Shh = (S11+S10+S01+S00)/4.;
			Shmh= (S10+S00+S1m1+S0m1)/4.;
			Smhh= (Sm11+S01+Sm10+S00)/4.;
			Smhmh=(Sm10+Sm1m1+S0m1+S00)/4.;

			djxr=(S10+S00+Shh+Shmh)*(Pcurr[pos(i+1,j)]-psi)/(4.*dx);
			djxl=(S00+Sm10+Smhh+Smhmh)*(psi-Pcurr[pos(i-1,j)])/(4.*dx);
			djyu=(S01+S00+Shh+Smhh)*(Pcurr[pos(i,j+1)]-psi)/(4.*dx);
			djyd=(S00+S0m1+Shmh+Smhmh)*(psi-Pcurr[pos(i,j-1)])/(4.*dx);

			src=(djxr-djxl+djyu-djyd)/dx;			
		}
		else
		{
			src=0.;
		}

		// The diagonal contribution
		if((1.-phi)>=1.e-6) 
		{
			djxr=(S11+S00+S01+S10)*(Pcurr[pos(i+1,j+1)]-psi)/(4.*sqrt2*dx);
			djxl=(S00+Sm1m1+Sm10+S0m1)*(psi-Pcurr[pos(i-1,j-1)])/(4.*sqrt2*dx);
			djyu=(Sm11+S00+S01+Sm10)*(Pcurr[pos(i-1,j+1)]-psi)/(4.*sqrt2*dx);
			djyd=(S00+S1m1+S10+S0m1)*(psi-Pcurr[pos(i+1,j-1)])/(4.*sqrt2*dx);

			srcd=(djxr-djxl+djyu-djyd)/sqrt2/dx;			
		}
		else
		{
			srcd=0.;
		}
		
		// The scalar field in the diffusion term
		S00 = 1.-phi;
		S10 = (1.-F[pos(i+1,j)]);
		Sm10= (1.-F[pos(i-1,j)]);
		S01 = (1.-F[pos(i,j+1)]);
		S0m1= (1.-F[pos(i,j-1)]);
		S11 = (1.-F[pos(i+1,j+1)]);
		Sm11= (1.-F[pos(i-1,j+1)]);
		S1m1= (1.-F[pos(i+1,j-1)]);
		Sm1m1=(1.-F[pos(i-1,j-1)]);

		// -----------------------------------
		// DivompGradU <- Div((1-h_Phi)*Grad(h_U))
		// -----------------------------------							
		if((1.-phi)>=1.e-6) 
		{
			// Shh is the scalar fields at (i+1/2,j+1/2). Smhmh is the scalar fields at (i-1/2,j-1/2) and so on.
			Shh = (S11+S10+S01+S00)/4.;
			Shmh= (S10+S00+S1m1+S0m1)/4.;
			Smhh= (Sm11+S01+Sm10+S00)/4.;
			Smhmh=(Sm10+Sm1m1+S0m1+S00)/4.;

			djxr=(S10+S00+Shh+Shmh)*(Cucurr[pos(i+1,j)]-u)/(4.*dx);
			djxl=(S00+Sm10+Smhh+Smhmh)*(u-Cucurr[pos(i-1,j)])/(4.*dx);
			djyu=(S01+S00+Shh+Smhh)*(Cucurr[pos(i,j+1)]-u)/(4.*dx);
			djyd=(S00+S0m1+Shmh+Smhmh)*(u-Cucurr[pos(i,j-1)])/(4.*dx);

			DivompGradU=(djxr-djxl+djyu-djyd)/dx;			
		}
		else
		{
			DivompGradU=0.;
		}

		// The diagonal contribution
		if((1.-phi)>=1.e-6) 
		{
			djxr=(S11+S00+S01+S10)*(Cucurr[pos(i+1,j+1)]-u)/(4.*sqrt2*dx);
			djxl=(S00+Sm1m1+Sm10+S0m1)*(u-Cucurr[pos(i-1,j-1)])/(4.*sqrt2*dx);
			djyu=(Sm11+S00+S01+Sm10)*(Cucurr[pos(i-1,j+1)]-u)/(4.*sqrt2*dx);
			djyd=(S00+S1m1+S10+S0m1)*(u-Cucurr[pos(i+1,j-1)])/(4.*sqrt2*dx);

			DivompGradUd=(djxr-djxl+djyu-djyd)/sqrt2/dx;			
		}
		else
		{
			DivompGradUd=0.;
		}

		
		// ----
		// h_dUdt
		// ----		
		REAL DivJ = D*(2.*DivompGradU/3.+DivompGradUd/3.)+(2.*src/3.+srcd/3.) ;
		REAL phinext = tanh(Pnext[pos(i,j)]/sqrt2);

		//Cunext[pos(i,j)]=Cucurr[pos(i,j)]+dt*(D*(2.*DivompGradU/3.+DivompGradUd/3.)+(2.*src/3.+srcd/3.)+(1.+omk*u)*omp2/sqrt2*dpdt)/(opk-omk*phi) ;
		Cunext[pos(i,j)]=( DivJ*dt+(Cucurr[pos(i,j)]+1./omk)*(opk-omk*phi) )/(opk-omk*phinext) - 1./omk;
	}
#else	// the old Compute_U

#define	dpyl	dpyr
#define	dpxu	dpyr
#define	dpxd	dpyr
#define	djxr	dpxr
#define	djxl	dpxl
#define	djyu	dpyu
#define	djyd	dpyd
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(i!=0 && i!=Nx+1 && j!=0 && j!=Ny+1)
	{
		
		double dpxr,dpyr,dnormr,dpxl,dnorml,dpyu,dnormu,dpyd,dnormd;
		double src,DivompGradU;	
		
		double u=Cucurr[pos(i,j)];
		double phi=F[pos(i,j)];
		double omp2=1.-phi*phi;
		double dphidt=omp2/sqrt2*(Pnext[pos(i,j)]-Pcurr[pos(i,j)])/dt;
		
		// -----------------------------------------------
		// src <- div[(dphi/dt)gradphi/|gradphi|)]/sqrt(2)
		// i.e. the source term for the diffusion field
		// -----------------------------------------------
		if(fabs(omp2)>=1.e-6) 
		{				
			// The unit gradient may be computed either by Grad(h_Psi) or Grad(h_Phi)
			dpxr=F[pos(i+1,j)]-phi ;
			dpyr=(F[pos(i+1,j+1)]+F[pos(i,j+1)]-F[pos(i+1,j-1)]-F[pos(i,j-1)])/4.;
			dnormr=sqrt(dpxr*dpxr+dpyr*dpyr);
			dpxl=phi-F[pos(i-1,j)] ;
			dpyl=(F[pos(i-1,j+1)]+F[pos(i,j+1)]-F[pos(i-1,j-1)]-F[pos(i,j-1)])/4.;
			dnorml=sqrt(dpxl*dpxl+dpyl*dpyl);
			dpyu=F[pos(i,j+1)]-phi ;
			dpxu=(F[pos(i+1,j+1)]+F[pos(i+1,j)]-F[pos(i-1,j+1)]-F[pos(i-1,j)])/4.;
			dnormu=sqrt(dpxu*dpxu+dpyu*dpyu);
			dpyd=phi-F[pos(i,j-1)] ;
			dpxd=(F[pos(i+1,j-1)]+F[pos(i+1,j)]-F[pos(i-1,j-1)]-F[pos(i-1,j)])/4.;
			dnormd=sqrt(dpxd*dpxd+dpyd*dpyd);
			if((dnormr*dnorml*dnormu*dnormd)>1.e-15) 
			{
				djxr=0.5*((1.-F[pos(i+1,j)]*F[pos(i+1,j)])/sqrt2*(Pnext[pos(i+1,j)]-Pcurr[pos(i+1,j)])/dt*(1.+omk*Cucurr[pos(i+1,j)])
						  +dphidt*(1.+omk*u))*dpxr/dnormr;
				djxl=0.5*((1.-F[pos(i-1,j)]*F[pos(i-1,j)])/sqrt2*(Pnext[pos(i-1,j)]-Pcurr[pos(i-1,j)])/dt*(1.+omk*Cucurr[pos(i-1,j)])
						  +dphidt*(1.+omk*u))*dpxl/dnorml;
				djyu=0.5*((1.-F[pos(i,j+1)]*F[pos(i,j+1)])/sqrt2*(Pnext[pos(i,j+1)]-Pcurr[pos(i,j+1)])/dt*(1.+omk*Cucurr[pos(i,j+1)])
						  +dphidt*(1.+omk*u))*dpyu/dnormu;
				djyd=0.5*((1.-F[pos(i,j-1)]*F[pos(i,j-1)])/sqrt2*(Pnext[pos(i,j-1)]-Pcurr[pos(i,j-1)])/dt*(1.+omk*Cucurr[pos(i,j-1)])
						  +dphidt*(1.+omk*u))*dpyd/dnormd;
				src=(djxr-djxl+djyu-djyd)/dx/sqrt2;
			}
			else
			{
				src=0.;
			}
		}
		else
		{
			src=0.;
		}
		
		// -----------------------------------
		// DivompGradU <- Div((1-h_Phi)*Grad(h_U))
		// -----------------------------------							
		if((1.-phi)>=1.e-6) 
		{
			DivompGradU=0.5*( (2.-F[pos(i+1,j)]-phi)*(Cucurr[pos(i+1,j)]-u)
							 -(2.-F[pos(i-1,j)]-phi)*(u-Cucurr[pos(i-1,j)])
							 +(2.-F[pos(i,j+1)]-phi)*(Cucurr[pos(i,j+1)]-u)
							 -(2.-F[pos(i,j-1)]-phi)*(u-Cucurr[pos(i,j-1)])
							 )/dx/dx;
		}
		else
		{
			DivompGradU=0.;
		}
		
		// ----
		// h_dUdt
		// ----		
		Cunext[pos(i,j)]=Cucurr[pos(i,j)]+dt*(D*DivompGradU+src+(1.+omk*u)*dphidt)/(opk-omk*phi) ;
	}
#endif

	
	if(i==0 && j==0)
	{
		iter++;
	}
	
}

__global__ void PullBack(REAL *P,REAL *Cu,int *Pha,REAL *F, Constants *Param,int off)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(int i=0;i<Nx+1-off;i++)
	{
		Cu[pos(i,j)]=Cu[pos(i+off,j)];
		P[pos(i,j)]=P[pos(i+off,j)];
		Pha[pos(i,j)]=Pha[pos(i+off,j)];
	}
	for(int i=Nx+1-off;i<=Nx+1;i++)
	{
		Pha[pos(i,j)]=0;
		Cu[pos(i,j)]=-1.;
		P[pos(i,j)]=P[pos(i-1,j)]-dx;
	}
	if(j==0)
	{
		xoffs+=off*dx;
	}
	
}

int FindMaxX(REAL *P)
{
	int max=0;
	for(int i=0;i<=Nx;i++)
	{
		for(int j=0;j<=Ny;j++)
		{
			if(P[pos(i,j)]*P[pos(i+1,j)]<0. && i>max)
			{
				max=i;
			}
		}
	}
	return max;
}

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Output /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
void WriteFields(char Prefix[LENMAX],int index,REAL *P,REAL *F,REAL *Cu,int *Pha,Constants *Param)
{
	int i,j;
	char FileName[LENMAX];
	char Extension[LENMAX]="dat";
	FILE *OutFile;
	float columnsize=(float)(Nx+2);
	float floati,floatj,floatVal;
	
	float coord=1.;

#if(COORDINATES==MICRONS)
	coord=dx_microns;
#endif

	REAL DeltaF=1.-(xinit*lT+xoffs-Vp*dt*iter)/lT;
	

	if (OUTPUT_U || index == IndexFINAL){
		if (index == IndexFINAL){
			sprintf(FileName,"%s_U.Final.%s",Prefix,Extension);
		}
		else if (index >=0){
			sprintf(FileName,"%s_U.%d.%s",Prefix,index,Extension);
		}
		else{
			sprintf(FileName,"%s_U.Error.%s",Prefix,Extension);
		}

		OutFile=fopen(FileName,"w");
		if(index==IndexFINAL)
		{ 
			fprintf(OutFile, "#Nx\t%d\n",Nx);				
			fprintf(OutFile, "#Ny\t%d\n",Ny);				
			fprintf(OutFile, "#Delta\t%lf\n",DeltaF);				
			fprintf(OutFile, "#dx\t%g microns\n",dx_microns);				
			fprintf(OutFile, "#x\ty\tU\n");				
			for(int i=0; i<Nx+2; i++) 
			{    
				for(int j=0; j<Ny+2; j++) 
				{	
					fprintf(OutFile,"%g\t%g\t%g\n",i*coord,j*coord,Cu[pos(i,j)]);
				}
				fprintf(OutFile,"\n");
			}	
		}
		else{
			fprintf(OutFile, "#Nx\t%d\n",Nx);				
			fprintf(OutFile, "#Ny\t%d\n",Ny);				
			fprintf(OutFile, "#Delta\t%lf\n",DeltaF);				
			fprintf(OutFile, "#dx\t%g microns\n",dx_microns);				
			fprintf(OutFile, "#x\ty\tU\n");				
			for(int i=0; i<Nx+2; i++) 
			{    
				for(int j=0; j<Ny+2; j++) 
				{	
					fprintf(OutFile,"%g\t%g\t%g\n",i*coord,j*coord,Cu[pos(i,j)]);
				}
				fprintf(OutFile,"\n");
			}	
		}
		fclose(OutFile);
	}

	if(OUTPUT_Psi || index==IndexFINAL)
	{
		// Output Psi
		// ----------
		if(index==IndexFINAL)
		{ sprintf(FileName,"%s_Psi.Final.%s",Prefix,Extension); }
		else if (index>=0) 
		{ sprintf(FileName,"%s_Psi.%d.%s",Prefix,index,Extension); }
		else 
		{ sprintf(FileName,"%s_Psi.Error.%s",Prefix,Extension); }
		
		OutFile=fopen(FileName,"w");
		if(index==IndexFINAL)
		{ 
			fprintf(OutFile, "#Nx\t%d\n",Nx);				
			fprintf(OutFile, "#Ny\t%d\n",Ny);				
			fprintf(OutFile, "#Delta\t%lf\n",DeltaF);				
			fprintf(OutFile, "#dx\t%g microns\n",dx_microns);				
			fprintf(OutFile, "#x\ty\tPsi\n");				
			for(int i=0; i<Nx+2; i++) 
			{    
				for(int j=0; j<Ny+2; j++) 
				{	
					fprintf(OutFile,"%g\t%g\t%g\n",i*coord,j*coord,P[pos(i,j)]);
				}
				fprintf(OutFile,"\n");
			}	
		}
		// else
		// {
		// 	fwrite((char*)&columnsize,sizeof(float),1,OutFile);
		// 	for(i=0;i<Nx+2;i++) 
		// 	{
		// 		floati=(float)(i*coord);
		// 		fwrite((char*)&floati,sizeof(float),1,OutFile);
		// 	}
			
		// 	for(j=0;j<Ny+2;j++) 
		// 	{
		// 		floatj=(float)(j*coord);
		// 		fwrite((char*)&floatj,sizeof(float),1,OutFile);
		// 		for(i=0;i<Nx+2;i++) 
		// 		{
		// 			floatVal=(float)(P[pos(i,j)]);			
		// 			fwrite((char*)&floatVal,sizeof(float),1,OutFile);
		// 		}
		// 	}
		// }
		else
		{
					
			fprintf(OutFile, "#Nx\t%d\n",Nx);				
			fprintf(OutFile, "#Ny\t%d\n",Ny);				
			fprintf(OutFile, "#Delta\t%lf\n",DeltaF);				
			fprintf(OutFile, "#dx\t%g microns\n",dx_microns);				
			fprintf(OutFile, "#x\ty\tPsi\n");				
			for(int i=0; i<Nx+2; i++) 
			{    
				for(int j=0; j<Ny+2; j++) 
				{	
					fprintf(OutFile,"%g\t%g\t%g\n",i*coord,j*coord,P[pos(i,j)]);
				}
				fprintf(OutFile,"\n");
			}	
		
		}
		fclose(OutFile);
	}	
	
	if(OUTPUT_C || index==IndexFINAL)
	{
		// Output C
		// ----------
		if(index==IndexFINAL)
		{ sprintf(FileName,"%s_C.Final.%s",Prefix,Extension); }
		else if (index>=0) 
		{ sprintf(FileName,"%s_C.%d.%s",Prefix,index,Extension); }
		else 
		{ sprintf(FileName,"%s_C.Error.%s",Prefix,Extension); }
		
		OutFile=fopen(FileName,"w");
		if(index==IndexFINAL)
		{ 
			fprintf(OutFile, "#Nx\t%d\n",Nx);				
			fprintf(OutFile, "#Ny\t%d\n",Ny);				
			fprintf(OutFile, "#Delta\t%lf\n",DeltaF);				
			fprintf(OutFile, "#dx\t%g microns\n",dx_microns);				
			fprintf(OutFile, "#x\ty\tC\n");				
			for(int i=0; i<Nx+2; i++) 
			{    
				for(int j=0; j<Ny+2; j++) 
				{	
					fprintf(OutFile,"%g\t%g\t%g\n",i*coord,j*coord,(0.5*(opk-omk*tanh(P[pos(i,j)]/sqrt2))*(1.+omk*Cu[pos(i,j)])));
				}
				fprintf(OutFile,"\n");
			}	
		}
		// else
		// {
		// 	fwrite((char*)&columnsize,sizeof(float),1,OutFile);
		// 	for(i=0;i<Nx+2;i++) 
		// 	{
		// 		floati=(float)(i*coord);
		// 		fwrite((char*)&floati,sizeof(float),1,OutFile);
		// 	}
			
		// 	for(j=0;j<Ny+2;j++) 
		// 	{
		// 		floatj=(float)(j*coord);
		// 		fwrite((char*)&floatj,sizeof(float),1,OutFile);
		// 		for(i=0;i<Nx+2;i++) 
		// 		{
		// 			floatVal=(float)(0.5*(opk-omk*tanh(P[pos(i,j)]/sqrt2))*(1.+omk*Cu[pos(i,j)]));			
		// 			fwrite((char*)&floatVal,sizeof(float),1,OutFile);
		// 		}
		// 	}
		// }

		else{
			fprintf(OutFile, "#Nx\t%d\n",Nx);				
			fprintf(OutFile, "#Ny\t%d\n",Ny);				
			fprintf(OutFile, "#Delta\t%lf\n",DeltaF);				
			fprintf(OutFile, "#dx\t%g microns\n",dx_microns);				
			fprintf(OutFile, "#x\ty\tC\n");				
			for(int i=0; i<Nx+2; i++) 
			{    
				for(int j=0; j<Ny+2; j++) 
				{	
					fprintf(OutFile,"%g\t%g\t%g\n",i*coord,j*coord,(0.5*(opk-omk*tanh(P[pos(i,j)]/sqrt2))*(1.+omk*Cu[pos(i,j)])));
				}
				fprintf(OutFile,"\n");
			}	
		}
		fclose(OutFile);
	}
    
    
    
    
    ///
    /// temporal
    ///
    /// compoX
    ///
    
    if(index==IndexFINAL)
    {
        // file name
        sprintf(FileName,"CompoX_%s.Final.%s",Prefix,Extension);
        
        REAL xmax = 0.;
        int advjtip = 1;
        
        // find tip position
        
        for(int j=1;j<=Ny;j=j+1)
        {
            REAL imax=0;
            for(int i=0;i<Nx;i=i+1)
            {
                if (P[pos(i,j)]*P[pos(i+1,j)]<=0.)
                {
                    imax = i-P[pos(i,j)]/(P[pos(i+1,j)]-P[pos(i,j)]);
                }
                
            }
            
            // check max value
            if(imax > xmax)
            {
                xmax=imax;
                advjtip=j;
            }
            
            
        }
        
        //
        
        OutFile=fopen(FileName,"w");
        fprintf(OutFile, "#(xmax,j) = (%g, %g) \n",xmax*coord, advjtip*coord);
        fprintf(OutFile, "#x\tC\n");
        
        for(int i=0; i<Nx+2; i++)
        {
            fprintf(OutFile,"%g\t%g\n",i*coord,(0.5*(opk-omk*tanh(P[pos(i,advjtip)]/sqrt2))*(1.+omk*Cu[pos(i,advjtip)])));
        }
        
        fclose(OutFile);

#if(CheckGrad)
		// file name
        sprintf(FileName,"Grad_%s.Final.%s",Prefix,Extension);        
        
        //
        OutFile=fopen(FileName,"w");
        fprintf(OutFile, "#(xmax,j) = (%g, %d) \n",xmax, advjtip);
        fprintf(OutFile, "#i\tNormG\tNormG_Approx\tratio\n");
        
        for(int i=1; i<=Nx; i++)
        {
			int	j=advjtip;
			if(j<1) 	j=1;
			if(j>Ny)	j=Ny;
			// ------------
			// Gradient h_Phi
			// ------------		
			REAL phx=(F[pos(i+1,j)]-F[pos(i-1,j)])/(2.*dx);
			REAL phy=(F[pos(i,j+1)]-F[pos(i,j-1)])/(2.*dx);
			REAL NormG=sqrt(phx*phx+phy*phy);

			REAL NormG_Approx=(1.-F[pos(i,j)]*F[pos(i,j)])/sqrt2;
			REAL NormG_ratio;
			if(NormG_Approx<1.e-6)
			{
				NormG_ratio=1.;
			}
			else
			{
				NormG_ratio=NormG/NormG_Approx;
			}
			       	
            fprintf(OutFile,"%d\t%g\t%g\t%g\n",i,NormG,NormG_Approx,NormG_ratio);
        }
        fclose(OutFile);
#endif
        
    }
   
    ///
    ///
    ///
    
	
	if(OUTPUT_Pha || index==IndexFINAL)
	{
		// Output Phase
		// ----------
		if(index==IndexFINAL)
		{ sprintf(FileName,"%s_Pha.Final.%s",Prefix,Extension); }
		else if (index>=0) 
		{ sprintf(FileName,"%s_Pha.%d.%s",Prefix,index,Extension); }
		else 
		{ sprintf(FileName,"%s_Pha.Error.%s",Prefix,Extension); }
		
		OutFile=fopen(FileName,"w");
		if(index==IndexFINAL)
		{
			fprintf(OutFile, "#Nx\t%d\n",Nx);				
			fprintf(OutFile, "#Ny\t%d\n",Ny);				
			fprintf(OutFile, "#Delta\t%lf\n",DeltaF);				
			fprintf(OutFile, "#dx\t%g microns\n",dx_microns);				
			fprintf(OutFile, "#x\ty\tPhase\n");				
			for(int i=0; i<Nx+2; i++) 
			{    
				for(int j=0; j<Ny+2; j++) 
				{	
					fprintf(OutFile,"%g\t%g\t%d\n",i*coord,j*coord,Pha[pos(i,j)]);
				}
				fprintf(OutFile,"\n");
			}	
		}
		else
		{
			fwrite((char*)&columnsize,sizeof(float),1,OutFile);
			for(i=0;i<Nx+2;i++) 
			{
				floati=(float)(i*coord);
				fwrite((char*)&floati,sizeof(float),1,OutFile);
			}
			
			for(j=0;j<Ny+2;j++) 
			{
				floatj=(float)(j*coord);
				fwrite((char*)&floatj,sizeof(float),1,OutFile);
				for(i=0;i<Nx+2;i++) 
				{
					floatVal=(float)(Pha[pos(i,j)]);			
					fwrite((char*)&floatVal,sizeof(float),1,OutFile);
				}
			}
		}
		fclose(OutFile);
	}
	
#if(OUTPUT_Phi)
	// Output Phi
	// ----------
	if (index>=0) 
	sprintf(FileName,"%s_Phi.%d.%s",Prefix,index,Extension);
	else 
	sprintf(FileName,"%s_Phi.Error.%s",Prefix,Extension);

	OutFile=fopen(FileName,"w");
	
	fprintf(OutFile, "#Nx\t%d\n",Nx);				
	fprintf(OutFile, "#Ny\t%d\n",Ny);				
	fprintf(OutFile, "#Delta\t%lf\n",DeltaF);				
	fprintf(OutFile, "#dx\t%g microns\n",dx_microns);				
	fprintf(OutFile, "#x\ty\tPhi\n");				
	for(int i=0; i<Nx+2; i++) 
	{    
		for(int j=0; j<Ny+2; j++) 
		{	
			fprintf(OutFile,"%g\t%g\t%g\n",i*coord,j*coord,(tanh(P[pos(i,j)]/sqrt2)));
		}
		fprintf(OutFile,"\n");
	}
	// fwrite((char*)&columnsize,sizeof(float),1,OutFile);
	// for(i=0;i<Nx+2;i++) 
	// {
	// 	floati=(float)(i*coord);
	// 	fwrite((char*)&floati,sizeof(float),1,OutFile);
	// }
	
	// for(j=0;j<Ny+2;j++) 
	// {
	// 	floatj=(float)(j*coord);
	// 	fwrite((char*)&floatj,sizeof(float),1,OutFile);
	// 	for(i=0;i<Nx+2;i++) 
	// 	{
	// 		floatVal=(float)(tanh(P[pos(i,j)]/sqrt2));			
	// 		fwrite((char*)&floatVal,sizeof(float),1,OutFile);
	// 	}
	// }
	fclose(OutFile);
#endif

	
#if(OUTPUT_Grains)
	// Output Grains
	// -------------
	if (index>=0) 
	sprintf(FileName,"%s_Grains.%d.%s",Prefix,index,Extension);
	else 
	sprintf(FileName,"%s_Grains.Error.%s",Prefix,Extension);
	OutFile=fopen(FileName,"w");
	
	fwrite((char*)&columnsize,sizeof(float),1,OutFile);
	for(i=0;i<Nx+2;i++) 
	{
		floati=(float)(i*coord);
		fwrite((char*)&floati,sizeof(float),1,OutFile);
	}
	
	for(j=0;j<Ny+2;j++) 
	{
		floatj=(float)(j*coord);
		fwrite((char*)&floatj,sizeof(float),1,OutFile);
		for(i=0;i<Nx+2;i++) 
		{
			floatVal=(float)(Pha[pos(i,j)]*(1.+tanh(P[pos(i,j)]/sqrt2))/2.);			
			fwrite((char*)&floatVal,sizeof(float),1,OutFile);
		}
	}
	fclose(OutFile);
#endif
}


REAL WriteTip(char FileName[LENMAX],REAL xprev,REAL *P,REAL *Cu,int itPrev,Constants *Param)
{	
	REAL dh=.1;	

	int error=0;
	int itip=0,jtip=0;	
	REAL xp[Npol],yp[Npol],zp[Npol];
	REAL xsec,pzer,ppl,pmi;
	REAL Vel,Delta,Omega,d0star,Rho,Sigma,Pe;
	REAL xtip,ytip,xtip_RefMat;	
	REAL Zero=0.;	
	FILE *TipF;
	
	// ---------
	// itip,jtip
	// ---------
	REAL i0=-1.,imax=0.;
	for(int i=0;i<Nx;i=i+1)
	{
		for(int j=1;j<=Ny;j=j+1)
		{
			if (P[pos(i,j)]*P[pos(i+1,j)]<=0.)
			{
				i0=i-P[pos(i,j)]/(P[pos(i+1,j)]-P[pos(i,j)]);
				if(i0>imax)
				{
					imax=i0;
					jtip=j;
				}
			}
		}
	}
	itip=(int)(imax);
	
	// ------------------------
	// Boundary conditions on y
	// ------------------------
	int jloc[Npol];
	for(int h=0;h<Npol;h++)
	{
		jloc[h]=jtip-Npol/2+h;
		if(jloc[h]<=0)
		{		
#if(BOUND_COND==NOFLUX)
				jloc[h]=1-jloc[h]; 
#endif	
#if(BOUND_COND==PERIODIC)
				jloc[h]+=Ny; 
#endif	
		}
		else if(jloc[h]>=Ny+1)
		{
#if(BOUND_COND==NOFLUX)
				jloc[h]=2*Ny+1-jloc[h]; 
#endif	
#if(BOUND_COND==PERIODIC)
				jloc[h]-=Ny; 
#endif	
		}
	}	
	
	// ---------------------------------
	// Store xtip(j-jtip) in zp[](xp[])
	// ---------------------------------
	for(int j=0;j<Npol;j=j+1)
	{
		itip=-1;
		for(int i=0;i<Nx;i=i+1)
		{
			if (P[pos(i,jloc[j])]*P[pos(i+1,jloc[j])]<=0.)
			{
				itip=i;
			}
		}
		for(int i=0;i<Npol;i=i+1)
		{
			xp[i]=1.*(itip-Npol/2+i);
			yp[i]=P[pos(itip-Npol/2+i,jloc[j])];				
		}
		xsec=RTBIS(xp,yp);
		if(xsec<0.){error=1;}
		zp[j]=xsec;
	}
	
	// ---------------
	// Find YTip [/dx]
	// ---------------
	for(int j=0;j<Npol;j=j+1)
	{
		xp[j]=1.*(jtip-Npol/2+j);
	}
	pzer=zp[0];
	ppl=PolInt(xp,zp,1.*(jtip-Npol/2+0)+dh);
	yp[0]=(ppl-pzer)/dh;
	for(int j=1;j<Npol-1;j=j+1)
	{
		pmi=PolInt(xp,zp,1.*(jtip-Npol/2+j)-dh);
		ppl=PolInt(xp,zp,1.*(jtip-Npol/2+j)+dh);
		yp[j]=(ppl-pmi)/(2.*dh);			
	}
	pzer=zp[Npol-1];
	pmi=PolInt(xp,zp,1.*(jtip-Npol/2+Npol-1)-dh);
	yp[Npol-1]=(pzer-pmi)/dh;
	ytip=RTBIS(xp,yp);
	if(ytip<0.){error=1;}

	// ---------
	// XTip [/dx]
	// ---------
	xtip=PolInt(xp,zp,ytip);
	itip=(int)(xtip);
	xtip_RefMat=xtip*dx+xoffs;				

	// -------------------------------
	// Delta [-], Omega [-], d_0* [/W]
	// -------------------------------
	Delta=1.-(xtip*dx-xint+xinit*lT+xoffs-Vp*dt*iter)/lT;
	Omega=1./omk*(1.-kcoeff/(kcoeff+omk*Delta));
	d0star=1./(E*(kcoeff+omk*Delta));		// d0/W=1/E

	// ------
	// Radius
	// ------
	double Num=0.,Denom=0.;
	for(int p=0;p<Npol;p++)
	{
		double xi=xp[p]-ytip;
		double yi=xtip-zp[p];
		Num+=xi*xi*yi;
		Denom+=xi*xi*xi*xi;
	}
	double A=Num/Denom;
	Rho=dx/2./A;

	// ---------------------------
	// Velocity, SigmaStar, Peclet
	// ---------------------------
	if (itPrev<=1) 
	{
		Vel=Sigma=Pe=1./Zero;
	}
	else
	{
		Vel=(xtip_RefMat-xprev)/((iter-itPrev)*dt);	
		Pe=Rho*Vel/(2.*D);			
		Sigma=2.*D*d0star/(Rho*Rho*Vel);	
	}

	// Check the mass conservation
#if(CheckMass)
	REAL cl_c0l;
	REAL cl_c0l_Ave=0.;
	for(int i=1;i<=Nx;i=i+1)
	{
		for(int j=1;j<=Ny;j=j+1)
		{
			cl_c0l = (0.5*(opk-omk*tanh(P[pos(i,j)]/sqrt2))*(1.+omk*Cu[pos(i,j)]));
			cl_c0l_Ave += cl_c0l;
		}
	}
	cl_c0l_Ave = cl_c0l_Ave/(1.*Nx*Ny);
#endif	

	// ------
	// Output
	// ------
	if(!error)
	{
		TipF=fopen(FileName,"a");
		fprintf(TipF,"%g \t",iter*dt*Tau0_sec);			// 1:	Time		[s]
		fprintf(TipF,"%g \t",Delta);					// 2:	Delta		[-]
		fprintf(TipF,"%g \t",Omega);					// 3:	Omega		[-]
		fprintf(TipF,"%g \t",Rho*W_microns);			// 4:	Rho_Tip		[microns]
		fprintf(TipF,"%g \t",Vel*W_microns/Tau0_sec);	// 5:	Vel_Tip		[microns/s]
		fprintf(TipF,"%g \t",Pe);						// 6:	Peclet_Tip	[-]
		fprintf(TipF,"%g \t",Sigma);					// 7:	SigmaStar	[-]
		fprintf(TipF,"%g \t",d0star*W_microns);			// 8:	d0Star		[microns]
		fprintf(TipF,"%g \t",xtip*dx_microns);			// 9:	x_tip		[microns]
		fprintf(TipF,"%g \t",ytip*dx_microns);			// 10:	y_tip		[microns]
		fprintf(TipF,"%g \t",xoffs*W_microns);			// 11:	x_offset	[microns]
		fprintf(TipF,"%g \t",xtip);						// 12:	i_tip
		fprintf(TipF,"%g \t",ytip);						// 13:	j_tip
		fprintf(TipF,"%g \t",xtip/Nx);					// 14:	x_tip/Nx
		fprintf(TipF,"%g \t",ytip/Ny);					// 15:	y_tip/Ny
		fprintf(TipF,"%g \t",xtip_RefMat*W_microns);	// 16:	x_tip/Mat	[microns]
	#if(CheckMass)
		fprintf(TipF,"%g \t",cl_c0l_Ave);				// 17:	total mass average
	#endif		
		fprintf(TipF,"\n");	
		fclose(TipF);
	}
	return xtip_RefMat;	
}

//------------------------ Interpolation ------------------------
#define VERYSMALL	(1.e-12)

REAL PolInt(REAL *XA,REAL *YA,REAL X)
{
	REAL DY=0.;
	REAL Y=-1.;

	int NMAX=Npol;
	REAL CC[NMAX],DD[NMAX];
	
	int NS=0;
	REAL DIF=fabs(X-XA[0]);
	
	for(int I=0;I<Npol;I++)
	{
        REAL DIFT=fabs(X-XA[I]);
        if(DIFT<DIF)
		{
			NS=I;
			DIF=DIFT;
        }
        CC[I]=YA[I];
        DD[I]=YA[I];
	}

	Y=YA[NS];
	NS=NS-1;
	
	for(int M=1;M<=Npol-1;M++)
	{
        for(int I=0;I<Npol-M;I++)
		{
			REAL HO=XA[I]-X;
			REAL HP=XA[I+M]-X;
			REAL W=CC[I+1]-DD[I];
			REAL DEN=HO-HP;
			if(fabs(DEN)<VERYSMALL)
			{
				printf("POLINT - Fatal error, DEN=0.0\n");
				return Y;
			}
			DEN=W/DEN;
			DD[I]=HP*DEN;
			CC[I]=HO*DEN;
		}
        if(2*NS<Npol-M)
		{
			DY=CC[NS+1];
        }
		else
		{
			DY=DD[NS];
			NS=NS-1;
        }
        Y=Y+DY;
	}
	return Y;
}

REAL RTBIS(REAL *xp,REAL *yp)
{
	REAL xacc=VERYSMALL;
	const int IterMax=40;
	int it;
	REAL f,fmid,h,xmid,xval;
	
	int im=1;
	int iM=Npol-2;
	REAL x1=xp[im]+xacc;
	REAL x2=xp[iM]-xacc;
	f=PolInt(xp,yp,x1);
	fmid=PolInt(xp,yp,x2);
	
	if(f*fmid>0.)
	{	
		int two=1;
		do{
			
			two++;
			im+=two%2;
			iM-=(two+1)%2;
			x1=xp[im];
			x2=xp[iM];
			f=PolInt(xp,yp,x1);
			fmid=PolInt(xp,yp,x2);

		}while(f*fmid>0. && im<iM);
		if(f*fmid>0.)
		{	
			printf("RTBIS: Root must be bracketed for bisection...\n");
			return -1.;	
		}
	}
	
	if(f<0.)
	{
		xval=x1;
		h=x2-x1;
	}
	else
	{
		xval=x2;
		h=x1-x2;
	}
	for(it=1;it<=IterMax;it++)
	{
		h=h*.5;
		xmid=xval+h;
		fmid=PolInt(xp,yp,xmid);

		if(fmid<=0.)
		{	xval=xmid;	}
		if(fabs(h)<xacc || fmid==0.)
		{	return xval;	}
	}
	return xval;
}

/////////////////////////////////////////////////////////////////////////////////
///////////////////////////  Cuda Device Management  ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
void DisplayDeviceProperties(int Ndev)
{
	cudaDeviceProp deviceProp;
	memset( &deviceProp, 0, sizeof(deviceProp));
	if( cudaSuccess == cudaGetDeviceProperties(&deviceProp, Ndev))
	{
		printf( "==============================================================");
		printf( "\nDevice Name \t %s ", deviceProp.name );
		printf( "\nDevice Index\t %d ", Ndev );
		printf( "\n==============================================================");
		printf( "\nTotal Global Memory                  \t %ld KB", (long int)(deviceProp.totalGlobalMem/1024) );
		printf( "\nShared memory available per block    \t %ld KB", (long int)(deviceProp.sharedMemPerBlock/1024) );
		printf( "\nNumber of registers per thread block \t %d", deviceProp.regsPerBlock );
		printf( "\nWarp size in threads             \t %d", deviceProp.warpSize );
		printf( "\nMemory Pitch                     \t %ld bytes", (long int)(deviceProp.memPitch) );
		printf( "\nMaximum threads per block        \t %d", deviceProp.maxThreadsPerBlock );
		printf( "\nMaximum Thread Dimension (block) \t %d * %d * %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );
		printf( "\nMaximum Thread Dimension (grid)  \t %d * %d * %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] );
		printf( "\nTotal constant memory            \t %ld bytes", (long int)(deviceProp.totalConstMem) );
		printf( "\nCUDA ver                         \t %d.%d", deviceProp.major, deviceProp.minor );
		printf( "\nClock rate                       \t %d KHz", deviceProp.clockRate );
		printf( "\nTexture Alignment                \t %ld bytes", (long int)(deviceProp.textureAlignment) );
		printf( "\nDevice Overlap                   \t %s", deviceProp. deviceOverlap?"Allowed":"Not Allowed" );
		printf( "\nNumber of Multi processors       \t %d", deviceProp.multiProcessorCount );
		printf( "\n==============================================================\n");
	}
	else
	{
		printf( "\nCould not get properties for device %d.....\n", Ndev);
	}
}
void GetMemUsage(int *Array,int Num)
{
	char buffer[LENMAX];
	std::string StrUse = "";
	FILE* pipe = popen("nvidia-smi -q --display=MEMORY | grep Used ", "r");
    while(!feof(pipe)) { if(fgets(buffer,LENMAX,pipe) != NULL){ StrUse+=buffer; } }
    pclose(pipe);
	for(int dev=0;dev<Num;dev++)
	{
		std::istringstream iss(StrUse.substr(StrUse.find(":")+1,StrUse.find("MB")-StrUse.find(":")-1));
		iss >> Array[dev];
		StrUse=StrUse.substr(StrUse.find("\n")+1,StrUse.length()-StrUse.find("\n")-1);
	}
}
int GetFreeDevice(int Num)
{
	int FreeDev=-1;
	int MemFree=15;
	int *Memory_Use = new int[Num] ;
	
	// Check utilization of Devices
    GetMemUsage(Memory_Use,Num);
	// See if one is free
	int dev=0;
	do{
		if(Memory_Use[dev]<MemFree)
		{ 
			// Found one...
			FreeDev=dev; 
			// Check if it is really free...
			system("sleep 1s");
			GetMemUsage(Memory_Use,Num);
			if(Memory_Use[dev]>MemFree){ FreeDev=-1; }
			// twice...
			system("sleep 1s");
			GetMemUsage(Memory_Use,Num);
			if(Memory_Use[dev]>MemFree){ FreeDev=-1; }
		}	
		dev++;
	}while(FreeDev==-1 && dev<Num);

	delete [] Memory_Use;
	
	if(FreeDev==-1)
	{
		printf("=======================================\n");
		system("nvidia-smi -q --display=MEMORY |grep U");
		printf("=======================================\n");
		printf("NO AVAILABLE GPU: SIMULATION ABORTED...\n");
		printf("=======================================\n\n");
	}
	return FreeDev;
}
void AutoBlockSize(int *Bloc)
{
	for(int bx=1;bx<=BSIZEMAX;bx++)
	{
		if(((Nx+2.)/bx)==((Nx+2)/bx))
		{
			for(int by=1;by<=BSIZEMAX && bx*by<=BLOCKMAX;by++)
			{
				if(((Ny+2.)/by)==((Ny+2)/by))
				{
                
                    if(bx*by==Bloc[0]*Bloc[1])
                    {
                        int VAR=(Bloc[0]-Bloc[1])*(Bloc[0]-Bloc[1]);
                        int var=(bx-by)*(bx-by);
                        if(var<VAR)
                        {
                            Bloc[0]=bx;
                            Bloc[1]=by;
                        }
                    }
                    else if(bx*by>Bloc[0]*Bloc[1])
                    {
                        Bloc[0]=bx;
                        Bloc[1]=by;
                    }
				
                }
			}
		}
	}
}



