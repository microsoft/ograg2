#!/bin/bash

# Define the list
crops=("soybean" "wheat")

test_size=10

distrs=( 
    '1.0,0.0,0.0'
    '0.0,1.0,0.0' 
    '0.0,0.0,1.0'
    '0.33,0.33,0.33'
    '0.5,0.25,0.25'
    '0.25,0.5,0.25' 
    '0.25,0.25,0.5')

# Loop over the list
for crop in "${crops[@]}"
do
    for distr in "${distrs[@]}"
    do
        python ragas_gen.py \
            -p data/ontology/farm_cropcultivation_schema_ontology_jsonld.json \
            -kg data/kg/${crop} \
            -i data/md/${crop} \
            --test_size ${test_size} \
            --distr ${distr} >> ragas_log.txt
    done
done