import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import vtk
import pyvista as pv
import os,sys

#Create folder
MYDIR = ("./contours")
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)
    print("created folder : %s".center(112)% MYDIR)

else:
    print("%s folder already exists.".center(112)%MYDIR)

# Dictionary for the simulations to make contours
# Possible field selections: Phi, Psi, U, C, default is Phi=0 contour
# Below is an example of choose three folders with different dx to check the spatial convergence

dict_phi = [
    {
    "name": "ISO_U_dx0.5",
    "Nx":600,
    "Ny":200,
    "ratio":2,
    "dx":0.026194,
    "label": "2D_GPU_dx_0.5",
    "prefix": "SCN_418_",
    "field": "Phi"
    }
    , 
    {
    "name": "ISO_U_dx0.7",
    "Nx":384,
    "Ny":128,
    "ratio":384./300.,
    "dx":0.0409281,
    "label": "2D GPU-PF",
    "prefix": "SCN_418_",
    "field": "Phi"
    }
    , 
    {
    "name": "ISO_U_dx1.0",
    "Nx":300,
    "Ny":100,
    "ratio":1.,
    "dx": 0.052388,
    "label": "2D_GPU_dx_1.0",
    "prefix": "SCN_418_",
    "field": "Phi"
    }
]

Fontsize = 25
Fontlabel = 25
#this flag control whether to include PRISMS-PF solution in the contour if using PRISMS-PF as well, use flag_prisms=1
flag_prisms = 0

NOUTPUT=50
tmax=400 #[tau]

#Colors and fake lines for legend
# Create a color map
colormap = plt.cm.get_cmap('viridis', len(dict_phi))
lines = [] # List to store the fake lines for the legend


# t is the Output index
# i.e, if NOUTPUT=50, then t=0~49+Final=999999 is the same as output t=49
#For Final output as well, use t in range (0, NOUTPUT+1)
for t in range (0, NOUTPUT):
    lines = []
    if (t < NOUTPUT):
        t_pf = t
        if (t == 0):
            vtk_name = "000000"
        else:
            vtk_name = int(100000*t/NOUTPUT)
            if (vtk_name < 10000):
                vtk_name = "00{}".format(vtk_name)
            elif (vtk_name < 100000):
                vtk_name = "0{}".format(vtk_name)
    else:
        vtk_name = 200000
        t_pf = 999999
    plt.figure(figsize=(15,5),facecolor='white',dpi=150)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylabel('X[W]',fontsize=Fontsize)
    plt.xlabel('Y[W]',fontsize=Fontsize)
    
    if t <= NOUTPUT:
        plt.title(r"$\phi = 0$ contours at t = {} $\tau_0$".format(np.floor(t/NOUTPUT*tmax)),fontsize=Fontsize)
    else:
        plt.title(r"$\phi = 0$ contours at t = {} $\tau_0$".format(tmax),fontsize=Fontsize)
        
    if (flag_prisms == 1):
        filename = "./PRISMS/refine_7_default_parameters/solution-{}.vtu".format(vtk_name)

        # Read the vtu file
        mesh = pv.read(filename)
        # Create a contour where Phi field equals to 0
        contour = mesh.contour([0], scalars="phi")
        # Get the points of the contour
        pts = contour.points
        line = mlines.Line2D([], [], color="orange", label="2D PRISMS-PF")
        lines.append(line)
        plt.scatter(pts[:,1], pts[:,0],marker=".",lw = 0.2, c="orange")

    for j in range (len(dict_phi)):
        filename = './{}/{}_{}.{}.dat'.format(dict_phi[0]["name"],dict_phi[0]["prefix"], dict_phi[0]["field"],t_pf)
        data = np.loadtxt(filename, skiprows=5)
        nx = dict_phi[j]["Nx"]
        ny = dict_phi[j]["Ny"]
        dx = dict_phi[j]["dx"]
        field = np.zeros([ny, nx])
        for i in range(data.shape[0]):
            x = int(np.round(data[i,0]/dx))
            y = int(np.round(data[i,1]/dx))
            if 0 <= x < nx and 0 <= y < ny:
                field[y,x] = data[i,2]

        # Prepare grid
        x = np.arange(nx) / dict_phi[j]["ratio"]
        y = np.arange(ny) / dict_phi[j]["ratio"]
        X, Y = np.meshgrid(x, y)
        color = colormap(j)
        contour = plt.contour(X, Y, field, levels=[0.], colors=[color])
        # Create a fake line with the same color for the legend
        line = mlines.Line2D([], [], color=color, label=dict_phi[j]["label"])
        lines.append(line)   
    plt.legend(handles=lines,loc='upper right',fontsize=Fontlabel)
    plt.savefig("{}/contour_{}.png".format(MYDIR,t), bbox_inches='tight')
    plt.close()
    
print("Complete")