#!/bin/bash
# Usage: condor_covtest.sh <problem directory> <search config> <timeout>
# Ex1: covtest.sh ../IPC/transport "--search eager_greedy(cea)" 60
# Ex2: covtest.sh ../IPC/transport "--heuristic h=ff() --search eager_greedy([h],preferred=[h])" 1800

OUT_OF_TIME_CHAR="T"
OUT_OF_MEMORY_CHAR="M"
COVTEST_DIR="condor_covtest"

problem_dir=$1
config=$2
time_limit=$3

mkdir -p ${COVTEST_DIR}

problem_counter=0
for problem in ${problem_dir}/p*.pddl
do
    mkdir -p ${COVTEST_DIR}/${problem_counter}
    printf "$problem" > ${COVTEST_DIR}/${problem_counter}/problem_name.txt
    ((problem_counter++)) 
done

CONDOR_SCRIPT="#!/bin/bash
#cd rsvm
#./build/server &
#server_pid=$!
#cd -
cd ${COVTEST_DIR}/\$1
fd_command=\"/usr/bin/python ../../../fast-downward.py --build release64 --overall-memory-limit 4G ${problem_dir}/domain.pddl \`cat problem_name.txt\` $config\"
echo \$fd_command
timeout $time_limit \$fd_command
printf \$? > return.txt
#kill ${server_pid}
#rm /tmp/fd-learn-socket"

CONDOR_SPEC="universe = vanilla
executable = ${COVTEST_DIR}/condor.sh
output = ${COVTEST_DIR}/\$(Process)/condor.out
error = ${COVTEST_DIR}/\$(Process)/condor.err
log = ${COVTEST_DIR}/condor.log
arguments = \$(Process)
requirements = regexp(\"^sprite[0-9][0-9]\", TARGET.Machine)
queue ${problem_counter}"


printf "$CONDOR_SCRIPT" > ${COVTEST_DIR}/condor.sh
chmod 775 ${COVTEST_DIR}/condor.sh
printf "$CONDOR_SPEC" > ${COVTEST_DIR}/condor.cmd

condor_submit ${COVTEST_DIR}/condor.cmd
condor_wait ${COVTEST_DIR}/condor.log
rm ${COVTEST_DIR}/condor.log # a workaround for the new version of Condor

results=()
n_states=()
plan_costs=()
search_times=()
solved=0
for (( counter=0; counter<problem_counter; counter++ ))
do
    return_code=`cat ${COVTEST_DIR}/${counter}/return.txt`
    if [ $return_code -eq 0 ]
    then
        results=(${results[@]} 1)
        n_state=`awk '/Generated [0-9]+ state/ {print $2}' ${COVTEST_DIR}/${counter}/condor.out`
        n_states=(${n_states[@]} $n_state)
        plan_cost=`awk '/Plan cost: [0-9]+/ {print $3}' ${COVTEST_DIR}/${counter}/condor.out`
        plan_costs=(${plan_costs[@]} $plan_cost)
        search_time=`awk '/Search time: [\.0-9]+s/ {print $3}' ${COVTEST_DIR}/${counter}/condor.out`
        search_times=(${search_times[@]} $search_time)
        ((solved++))
        # Retain data for future training
        #mv ${COVTEST_DIR}/${counter}/features.txt data-tmp/features${counter}.txt
        #mv ${COVTEST_DIR}/${counter}/labels.txt data-tmp/labels${counter}.txt
        #mv ${COVTEST_DIR}/${counter}/states.txt data-tmp/states${counter}.txt
        #mv ${COVTEST_DIR}/${counter}/sas_plan data-tmp/sas_plan${counter}
    else
        if [ $return_code -eq 124 ]
        then
            problem_char=$OUT_OF_TIME_CHAR
        else
            problem_char=$OUT_OF_MEMORY_CHAR
        fi
        results=(${results[@]} 0)
        n_states=(${n_states[@]} $problem_char)
        plan_costs=(${plan_costs[@]} $problem_char)
        search_times=(${search_times[@]} $problem_char)
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
echo "Solved ${solved} out of ${problem_counter}."
