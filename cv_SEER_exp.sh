for i in {1..15}
do
  echo "CV benchmark on subject: $i, start ..."
  CUDA_VISIBLE_DEVICES=1 python experiment_5CV_benchmark_batch.py --subject $i --chns 4 --nn_choice 7
  echo "CV benchmark on subject: $i, finished"
done
