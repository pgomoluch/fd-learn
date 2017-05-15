#!/bin/bash
# Usage: parking_data_gen.sh <no. plans> <no. curbs> <no. cars> <starting id> <data directory>
# 5 curbs, 8 cars recommended
# E.g. parking_data_gen.sh 100 5 8 0 data-park-5-8

ipc_domain="../../../IPC/own-parking/domain.pddl"
ipc_generator="../../../IPC/own-parking/parking-generator.pl"

data_dir=$5

mkdir $data_dir
mkdir $data_dir/labels
counter=0
while [ $counter -lt $1 ]
do
    perl $ipc_generator p $2 $3 > tmp-problem.pddl
    ../fast-downward.py --build="release64" $ipc_domain tmp-problem.pddl --search "astar(const)"
    id=$((counter+$4))
    mv states.txt $data_dir/$id.txt
    mv labels.txt $data_dir/labels/$id.txt
    mv output $data_dir/output$id
    # Uncomment to keep verification data
    #mv features.txt data-v/features-$spec-$id.txt
    counter=$((counter+1))
done
    
rm tmp-problem.pddl output.sas sas_plan features.txt
