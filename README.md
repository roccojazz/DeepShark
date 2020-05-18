# DeepShark
Machine Learning and its Application to Bioinformatics: Lyndon factorizations

## Pre-requisites

Most of the methods are implemented in Python using `scikit-learn` (and utility
functions from `numpy` and `pandas`).
For ease of reproduction, the exact versions of the packages can be installed
through the **conda package manager** by installing
[`miniconda`](https://docs.conda.io/en/latest/miniconda.html) and replicating
the environment using the file `spec-file.txt` in this repository with the
command:

```
conda create --name lyndonenv --file spec-file.txt
```

## Data

Due to size constraints, data is available at [this
URL](https://drive.google.com/drive/folders/1_E-wKUA6PNSMqIa2jBGyFMiak4ilDsK3).
The dataset is composed of the following two files:

- `transcripts_genes.fa.gz`, containing the sequences of the transcripts used
  for training the model and that should be placed (and `gunzip`-ed) in the
  `training` directory.
- `sample_10M_genes.fastq.gz`, containing a sample of 10M 100-bp long reads
  simulated from Human chromosomes 1, 17, and 21 using [Flux
  Simulator](https://dx.doi.org/10.1093/nar/gks666). This file should be placed
  (and `gunzip`-ed) in the `testing` directory.


## Reproduction

To carry out an experiment, follows the steps:

1) COMPUTE FINGERPRINT ######################################################

    - METHOD   : experiment_fingerprint_1f_np_step in SCRIPT fingerprint.py

    - CMD_LINE : python fingerprint.py --type 1f_np --path training/ 
                 --fasta transcripts_genes.fa --fact create 
                 --type_factorization ICFL_COMB --shift shift 
                 --filter list -n 4

    - RETURN   : given a FASTA file (.fa or .gz) computes for each type of 
                 factorization a "fingerprint" file containing a row for each 
                 read, with the format "IDGENE FINGERPRINT", where 
                 "FINGERPRINT" is the fingerprint of the read

    N.B.
    * --fact create : to create a file containing the factors corresponding 
                      to the fingerprint fingerprint (--fact no_create, 
                      otherwise)
    * --shift shift : to generate the shifts of lengths 100 (--shift no_shift, 
                      otherwise)
    * --filter list : to consider only the reads for the genes contained in 
                      the file list_experiment.txt (--filter no_list, otherwise)


2) DATASETS #################################################################

    - METHOD   : experiment_dataset_step in SCRIPT training_mp.py

    - CMD_LINE : python training.py --step dataset --path training/ 
                 --type_factorization ICFL_COMB  --k_value 5
                 --enrich no_string --k_type extended -n 4

    - RETURN   : for each type of factorization it uses the corresponding 
                 "fingerprint" file to generate a dataset for each value of k. 
                 Such a dataset will be splitted in 2 pickle files: 
                 dataset_X_factorization which contain the samples, and 
                 dataset_y_factorization which contains the corresponding labels

    N.B.
    * --enrich string   : to associate the enriched string for each k-finger 
                          (--enrich no_string, otherwise)
    * --k_type extended : to apply the padding with -1 values to complete the 
                          k-fingers (--k_type valid, otherwise)


3) K-FINGERS CLASSIFIERS ####################################################

     - METHOD   : experiment_training_step in SCRIPT training.py

    - CMD_LINE : python training.py --step train --path training/ --k_value 5
                 --type_factorization ICFL_COMB  --model RF -n 4

    - RETURN   : for each trained classifier save a PICKLE file 
                 (ex. RF_ICFL_COMB_K5.pickle) and the report CSV containing 
                 the metrics for the performance in training 
                 (ex. RF_kfinger_clsf_report_ICFL_COMB_K5.csv)


4) READS CLASSIFICATION #####################################################

- PRE-SETTING:
    a) A k-finger trained classifier (ex. RF_ICFL_COMB_K5.pickle)
    b) The dataset for the k-finger trained classifier chosen 
       (ex. dataset_X_ICFL_COMB_K5.pickle, dataset_y_ICFL_COMB_K5.pickle)
    c) The fingerprint and fact_fingerprint corresponding to the type of 
       factorization for which the chosen classifier was trained 
       (ex. fingerprint_ICFL_COMB.txt e fact_fingerprint_ICFL_COMB.txt)

    a) RF FINGERPRINT CLASSIFIER: ##########################################

        i) TRAINING RF FINGERPRINT CLASSIFIER:

        - METHOD    : training_train_RF_fingerprint_step in SCRIPT training.py

        - CMD_LINE  : python training.py --step train_RF_fingerprint 
                      --path testing/ --model RF_FINGERPRINT
                      --type_factorization ICFL_COMB -n 4

        - RETURN    : save the PICKLE RF FINGERPRINT trained 
                      (ex. RF_fingerprint_classifier_ICFL_COMB.pickle)
                      and the corresponding CSV report 
                      (RF_fingerprint_clsf_report_ICFL_COMB.csv")

        ii) TESTING READS:

        - METHOD    : testing_reads_RF_fingerprint_step in SCRIPT testing.py

        - CMD_LINE  : python testing.py --step test_RF_fingerprint --path testing/ 
                      --fasta sample_10M_genes.fastq.gz --filter list
                      --rf_fingerprint_model RF_fingerprint_classifier_ICFL_COMB.pickle  
                      --type_factorization ICFL_COMB --random no_random -n 4

        - RETURN    : creates the file test_rf_fingerprint_result.txt containing 
                      a row for each read in the FASTA file. 
                      

    b) RULE-BASED READ CLASSIFIER: ###############################################

        - METHOD    : testing_reads_majority_step in SCRIPT testing.py

        - CMD_LINE  : python testing.py --step test_majority --path fingerprint/test/ 
                     --fasta sample_10M_genes.fastq.gz --best_model RF_ICFL_COMB_K5.pickle 
                     --fact create --criterion majority --random no_random
                     --filter list --type_factorization ICFL_COMB --k_value 5 
                     --k_type extended --n_for_genes 10 -n 4

        - RETURN    : creates a file test_majority_result.txt containing a row 
                      for each read in the FASTA file. 


5) COMPUTE METRIC ###########################################################

       - CMD_LINE   : python metrics.py --path fingerprint/test/ 
                      --file test_majority_result_no_thresholds_list.txt 
                      --problem classification
