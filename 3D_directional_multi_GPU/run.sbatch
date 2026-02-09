#!/bin/bash

# Use command "sbatch run.sbatch" to submit job
# This is an example for submitting sbatch job to discovery cluster
# Please refer to "http://nuweb12.neu.edu/rc/?page_id=18#gpujobs" for more details

#SBATCH --job-name=V3
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH -N 1                        # number of node (1 GPU per node in Discovery).
#SBATCH -n 2                       # number of MPI process
#SBATCH -t 1-00:00:00
#SBATCH --mem=64Gb
##SBATCH --exclusive
#SBATCH --partition multigpu             # Submit job to the new partition
#SBATCH --gres=gpu:v100-sxm2:4
##SBATCH -w, --nodelist=d1002
#SBATCH --exclude=d1005,c2190             # Exclude specific node

name=Multi_PF3D
nvcc -arch=sm_70 src/*.cu -o $name


############################################################ input parameters

ln=0;

### initial conditions

ngpu=0123; # GPU setting

CompTime=0;
tlimit=83000;
iter=0;
niter=1;


## if file does not exist, then create the file
if [[ !(-f "Break_info.txt") ]]
then

    echo -e "CompTime[s] \tTimelimit[s] \titer \tniter" > "Break_info.txt"
    echo -e "$CompTime \t$tlimit \t$iter \t$niter" >> "Break_info.txt"

else

    echo -e "Break_info.txt file exits."

    ### Read last line
    ln=0;
    tail -n1 Break_info.txt | tr "\t" "\n" > Break_temp.txt
    ###

    ### save input values information
    while read line
    do

        Readinfo[ln]="$line"

        # numbering
        ln=$(($ln+1))

    done < "Break_temp.txt"

    ### re-directing information
    CompTime=${Readinfo[0]}
    tlimit=${Readinfo[1]}
    iter=${Readinfo[2]}
    niter=${Readinfo[3]}

fi

### run the compiled cuda code
### check if the simulation is already finished
if [[ $iter -lt $niter ]]
then
    echo "Simulation starting times" >> Break_SimTimes.txt
    date >> Break_SimTimes.txt
    ### run the compiled cuda code
    # [complied file] ngpu iter niter Tlimit rep
    ./$name "$ngpu" "$tlimit" "$iter" "$niter" >> output.txt


    ### Read last line
    ln=0;
    tail -n1 Break_info.txt | tr "\t" "\n" > Break_temp.txt
    ###

    ### save input values information
    while read line
    do

        Readinfo[ln]="$line"

        # numbering
        ln=$(($ln+1))

    done < "Break_temp.txt"

    ### re-directing information
    CompTime=${Readinfo[0]}
    tlimit=${Readinfo[1]}
    iter=${Readinfo[2]}
    niter=${Readinfo[3]}
else
    echo "Simulation is already finished." >> Break_SimTimes.txt
fi    