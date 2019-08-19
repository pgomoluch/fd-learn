#PBS -lwalltime=01:20:00
#PBS -lselect=2:ncpus=32:mem=32gb:mpiprocs=16

module load anaconda3/personal
cd $PBS_O_WORKDIR
mkdir tmpproblems
python $HOME/code/fd-learn-hpc/learning/param_search.py $HOME/IPC/own-transport/domain.pddl tmpproblems 3600
