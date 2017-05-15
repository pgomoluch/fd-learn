#!/bin/bash
# Usage: covtest.sh <problem directory> <search config> <timeout>
# E.g.: covtest.sh ../../IPC/transport "eager_greedy(cea)" 60

base_counter=0
counter=0
results=()
n_states=()
for p in $1/p*.pddl
do
    base_counter=$((base_counter+1))
    echo "Trying ${p}..."
    fd_command="../fast-downward.py --build=release64 $1/domain.pddl $p --search $2"
    echo $fd_command
    timeout $3 $fd_command > covtest-tmp-result
    if [ $? -eq 0 ]
    then
        counter=$((counter+1))
        results=(${results[@]} 1)
        n_state=`awk '/Generated [0-9]+ state/ {print $2}' covtest-tmp-result`
        n_states=(${n_states[@]} $n_state)
        echo "Solved ${p}."
    else
        echo "Failed ${p}."
        results=(${results[@]} 0)
        n_states=(${n_states[@]} "F")
    fi
done
echo ${results[@]}
echo ${n_states[@]}
echo "Solved ${counter} out of ${base_counter}."

rm covtest-tmp-result features.txt labels.txt states.txt output output.sas
