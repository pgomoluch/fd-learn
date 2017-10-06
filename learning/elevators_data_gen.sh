#!/bin/bash
# Usage: elevators_data_gen.sh <no. plans> <starting id> <data directory>
# Specific configuration set by the following variables.

ipc_domain="../../../IPC/own-elevators/domain.pddl"

# Like p1.5 (between p1 and p2)
#ipc_generator_command1="../../../IPC/own-elevators/generator/generate_elevators 3 3 1 1 1 12 6 1 2 4 4"
#ipc_generator_command2="../../../IPC/own-elevators/generator/generate_pddl 12 12 1 3 3 1 1 1"

# Like p11.5 (between p11 and p12)
ipc_generator_command1="../../../IPC/own-elevators/generator/generate_elevators 3 3 1 1 1 12 4 1 2 4 4"
ipc_generator_command2="../../../IPC/own-elevators/generator/generate_pddl 12 12 1 3 3 1 1 1"

generated_problem_file="p12_3_1.pddl"

data_dir=$3

mkdir $data_dir
mkdir $data_dir/labels
counter=0
while [ $counter -lt $1 ]
do
    $ipc_generator_command1
    $ipc_generator_command2
    ../fast-downward.py --build="release64" $ipc_domain $generated_problem_file --search "astar(const)"
    id=$((counter+$2))
    mv states.txt $data_dir/$id.txt
    mv labels.txt $data_dir/labels/$id.txt
    mv output $data_dir/output${id}
    # Uncomment to keep verification data
    #mv features.txt data-v/features-$spec-$id.txt
    
    counter=$((counter+1))
done

rm $generated_problem_file
rm output.sas
