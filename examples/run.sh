# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
#-----------------------------------------------------------------------------------------------------------------------
#!/bin/bash

if [ $1 = "sudoku" ]
then
    NAME=$2 # use name of puzzle from: easy, hard or AI_escargot
    EXE="sudoku.py"
    PARAM=$3
    VALUE=$4
elif [ $1 = "spin" ]
then
    NAME=$2 # use name of lattice from: AF3D, FM3D, SG50, SG10 or ring
    EXE="spin_lattice.py"
    PARAM=$3
    VALUE=$4
elif [ $1 = "world" ]
then
    NAME=world
    EXE="cmp_world.py"
    PARAM=$2
    VALUE=$3
elif [ $1 = "australia" ]
then
    NAME=australia
    EXE="cmp_australia.py"
    PARAM=$2
    VALUE=$3
fi


for i in {1..30}
do
    if [ -e  results/${NAME}_trial_${i}_spikes_binary ]
    then
        echo results/${NAME}_trial_${i}_spikes_binary is already present.
    else
        echo running trial_${i} for ${NAME}
        python3 $EXE $NAME --name ${NAME}_trial_${i} $PARAM $VALUE
    fi
done