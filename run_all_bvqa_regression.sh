#!/bin/bash

MODELS=(
  'FAVER_Haar'
  'FAVER_Db2'
  'FAVER_Bior22'
   
)

DATASETS=(
   'LIVE_HFR'
)

for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do

  feature_file=features/${DS}_${m}_feats.mat
  mos_file=features/${DS}_metadata.csv
  out_file=result/${DS}_${m}_SVR_corr_bc_two.mat
  log_file=logs/${DS}_regression_bc_two.log

#   echo "$m" 
#   echo "${feature_file}"
#   echo "${out_file}"
#   echo "${log_file}"

  cmd="python src/evaluate_bvqa_features_by_content_regression_traintwo.py"
  cmd+=" --model_name $m"
  cmd+=" --dataset_name ${DS}"
  cmd+=" --feature_file ${feature_file}"
  cmd+=" --mos_file ${mos_file}"
  cmd+=" --out_file ${out_file}"
  cmd+=" --log_file ${log_file}"
  cmd+=" --num_cont 16"
  cmd+=" --num_dists 30"
  cmd+=" --use_parallel"
  cmd+=" --log_short"
  cmd+=" --num_iterations 100"

  echo "${cmd}"

  eval ${cmd}
done
done
