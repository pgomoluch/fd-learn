#!/bin/bash
# usage: random_driver.sh <domain directory> <problem name> <walk length> <sample number> <initial sample id> <output data directory>

time_limit=30
min_length=3

base_path=$1
problem=$2
walk_length=$3
n_samples=$4
initial_id=$5
data_dir=$6

mkdir ${data_dir}
mkdir ${data_dir}/labels

counter=0

# Translate and preprocess
../fast-downward.py --build="release64" --translate --preprocess ${base_path}domain.pddl ${base_path}${problem}.pddl
# Make a copy for the altered version of the problem
cp output output2

while [ $counter -lt $n_samples ]
do
    # Random walk of walk_length steps from the goal
    ../builds/release64/bin/random_walker $walk_length < output > rd_tmp
    python replace_initial_state.py output2 rd_tmp
    search_command="../fast-downward.py --build=release64 output2 --search astar(const(0))"
    timeout $time_limit $search_command
    if [ $? -eq 0 ]
    then
        length=$(wc -l < labels.txt)
        if [ $length -ge $min_length ]
        then
            file_id=$((counter+initial_id))
            mv labels.txt ${data_dir}/labels/${file_id}.txt
            mv states.txt ${data_dir}/${file_id}.txt
            cp output2 ${data_dir}/output${file_id}
            # Uncomment to keep verification data
            #mv features.txt data-v/features-${dirname}-${problem}-rw${walk_length}-$file_id.txt
            counter=$((counter+1))
        fi
    fi
done

rm output output2 output.sas rd_tmp sas_plan features.txt
