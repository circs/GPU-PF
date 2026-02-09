import numpy as np
import matplotlib.pyplot as plt
import os,sys



# Dictionary of the simulation
# For field generation, keep single element in dict_phi
dict_phi = [
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
]

#Create folder
MYDIR = ("./{}").format(dict_phi[0]["field"])
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)
    print("created folder : %s".center(112)% MYDIR)

else:
    print("%s folder already exists.".center(112)%MYDIR)
    
Fontsize = 25
Fontlabel = 25

NOUTPUT=50
tmax= 400 #[tau]


#####Main Loop
# t is the Output index
# i.e, if NOUTPUT=50, then t=0~49+Final=999999 is the same as output t=49
#For Final output as well, use t in range (0, NOUTPUT+1)
for t in range (0, NOUTPUT):
    lines = []
    if (t < NOUTPUT):
        t_pf = t
    else:
        t_pf = 999999
        
    plt.figure(figsize=(15,5),facecolor='white',dpi=150)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylabel('Y[W]',fontsize=Fontsize)
    plt.xlabel('X[W]',fontsize=Fontsize)
    
    #Change file name according to prefix, or desired fields like C or U or Psi
    filename = './{}/{}_{}.{}.dat'.format(dict_phi[0]["name"],dict_phi[0]["prefix"], dict_phi[0]["field"],t_pf)
    data = np.loadtxt(filename, skiprows=5)
    nx = dict_phi[0]["Nx"]
    ny = dict_phi[0]["Ny"]
    dx = dict_phi[0]["dx"]
    field = np.zeros([ny, nx])
    for i in range(data.shape[0]):
        x = int(np.round(data[i,0]/dx))
        y = int(np.round(data[i,1]/dx))
        if 0 <= x < nx and 0 <= y< ny:
            field[y,x] = data[i,2]

    # Prepare grid
    x = np.arange(nx) / dict_phi[0]["ratio"]
    y = np.arange(ny) / dict_phi[0]["ratio"]
    
    plt.imshow(field, origin='lower')
    plt.colorbar(label=r'$\phi$')
    plt.savefig("{}/Phi{}.png".format(MYDIR,t), bbox_inches='tight')
    plt.close()
    #plt.show();
    
print("Done")
