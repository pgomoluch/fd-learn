#!/bin/bash

search_config=$1
time_limit=$2
output_dir=$3

while read p; do
    echo $p
    #echo $search_config
    ./covtest.sh $p "$search_config" $time_limit > $output_dir/`basename $p`.txt
done < domains.txt
