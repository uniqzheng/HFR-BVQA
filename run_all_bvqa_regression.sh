#!/bin/bash

MODELS=(
   #'ST_HIFRVIQ_Haar'
   #'ST_HIFRVIQ_Haar_ev16'
   #'ST_HIFRVIQ_Haar_ev8'
   #'ST_HIFRVIQ_Haar_ev4'
   #'ST_HIFRVIQ_Db2'
   #'ST_HIFRVIQ_Db2_ev16'
   #'ST_HIFRVIQ_Db2_ev8'
   #'ST_HIFRVIQ_Db2_ev4'
   #'ST_HIFRVIQ_Bior22'
   #'ST_HIFRVIQ_Bior22_ev16'
   #'ST_HIFRVIQ_Bior22_ev8'
   #'ST_HIFRVIQ_Bior22_ev4'
   )

DATASETS=(
  "LIVE_HFR"
)

for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do

  feature_file=features/${DS}_${m}_feats.mat
  mos_file=features/${DS}_metadata.csv
  out_file=result/${DS}_${m}_SVR_corr_bc.mat
  log_file=logs/${DS}_regression_bc.log

#   echo "$m" 
#   echo "${feature_file}"
#   echo "${out_file}"
#   echo "${log_file}"

  cmd="python src/evaluate_bvqa_features_regression.py"
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
