#!/bin/bash
# Usage: covtest.sh <problem directory> <search config> <timeout>
# Ex1: covtest.sh ../IPC/transport "--search eager_greedy(cea)" 60
# Ex2: covtest.sh ../IPC/transport "--heuristic h=ff() --search eager_greedy([h],preferred=[h])"

base_counter=0
counter=0
results=()
n_states=()
plan_costs=()
search_times=()
for p in $1/p*.pddl
do
    base_counter=$((base_counter+1))
    echo "Trying ${p}..."
    fd_command="../fast-downward.py --build=release64 $1/domain.pddl $p $2"
    # LAMA-first
    #fd_command="../fast-downward.py --build=release64 --alias lama-first $1/domain.pddl $p"
    echo $fd_command
    timeout $3 $fd_command > covtest-tmp-result
    if [ $? -eq 0 ]
    then
        counter=$((counter+1))
        results=(${results[@]} 1)
        n_state=`awk '/Generated [0-9]+ state/ {print $2}' covtest-tmp-result`
        n_states=(${n_states[@]} $n_state)
        plan_cost=`awk '/Plan cost: [0-9]+/ {print $3}' covtest-tmp-result`
        plan_costs=(${plan_costs[@]} $plan_cost)
        search_time=`awk '/Search time: [\.0-9]+s/ {print $3}' covtest-tmp-result`
        search_times=(${search_times[@]} $search_time)
        echo "Solved ${p}."
    else
        echo "Failed ${p}."
        results=(${results[@]} 0)
        n_states=(${n_states[@]} "F")
        plan_costs=(${plan_costs[@]} "F")
        search_times=(${search_times[@]} "F")
    fi
done
echo "RESULTS:"
echo ${results[@]}
echo "GENERATED STATES:"
echo ${n_states[@]}
echo "PLAN COSTS:"
echo ${plan_costs[@]}
echo "SEARCH TIMES:"
echo ${search_times[@]}
echo "Solved ${counter} out of ${base_counter}."

rm covtest-tmp-result features.txt labels.txt states.txt output output.sas
