#!/bin/bash
# Usage: transport_data_gen.sh <no. plans> <starting seed> <starting id> <data directory>
# Specific configuration set by the following variables.

ipc_domain="../../../IPC/own-transport/domain.pddl"
ipc_generator_command="python ../../../IPC/own-transport/generator14L/city-generator.py 10 1000 2 100 2 4"

data_dir=$4

mkdir $data_dir
mkdir $data_dir/labels
counter=0
while [ $counter -lt $1 ]
do
    seed=$(($2+counter))
    $ipc_generator_command $seed > tmp-problem.pddl
    ../fast-downward.py --build="release64" $ipc_domain tmp-problem.pddl --search "astar(const)"
    id=$((counter+$3))
    mv states.txt $data_dir/$id.txt
    mv labels.txt $data_dir/labels/$id.txt
    mv output $data_dir/output${id}
    # Uncomment to keep verification data
    #mv features.txt data-v/features-$spec-$id.txt
    
    counter=$((counter+1))
done

rm city-sequential*seed.tex
rm tmp-problem.pddl
rm output.sas
