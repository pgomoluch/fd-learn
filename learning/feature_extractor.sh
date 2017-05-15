#!/bin/bash
# usage: feature_extractor.sh <data directory>

mkdir $1/features

for states in $1/*.txt
do
    filename=`basename $states`
    problem_id=`basename $filename .txt`
    ../builds/release64/bin/feature_extractor $1/output${problem_id} $states $1/features/$filename
done 
