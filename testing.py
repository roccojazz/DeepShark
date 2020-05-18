import argparse
import random
import pickle

from functools import partial
from multiprocessing.pool import ThreadPool as Pool
from fingerprint_utils import extract_reads, compute_fingerprint_by_list,cut_suffix_for_test
from machine_learning_utils import test_reads_majority,test_reads_rf_fingerprint


# Given a set of reads, performs classification by using the majority (or thresholds) criterion on best k-finger classification
# args.step = 'test_majority' ##########################################################################################
def testing_reads_majority_step(args):

    # Input fasta
    input_fasta = args.path + args.fasta

    # Extract of reads (Format = ID GENE read)
    read_lines = extract_reads(name_file=input_fasta, filter=args.filter, step='test', n_for_genes=args.n_for_genes)

    reads_no_suffix = []
    read_lines = [s.upper() for s in read_lines]

    if len(read_lines) == 0:
        print('No reads extracted!')
        exit(-1)

    if args.random == 'random':
        # Randomly extraction of 1000 reads from read_lines
        random_read_lines = []
        for i in range(10000):
            random_line = random.choice(read_lines)
            random_read_lines.append(random_line)

        read_lines = random_read_lines

    print('# READS: ', len(read_lines))

    # SPLIT for multiprocessing
    size = int(len(read_lines)/args.n)
    splitted_lines = [read_lines[i:i + size] for i in range(0, len(read_lines), size)]

    fingerprint_blocks = []
    with Pool(args.n) as pool:
        func = partial(compute_fingerprint_by_list, args.fact, args.shift, args.type_factorization)

        for res in pool.map(func, splitted_lines):

            fingerprint_blocks.append((res[0], res[1]))

        ################################################################################################################

    # SPLIT fingerprints blocks
    size = int(len(fingerprint_blocks) / args.n)
    splitted_blocks = [fingerprint_blocks[i:i + size] for i in range(0, len(fingerprint_blocks), size)]

    with Pool(args.n) as pool:

        # Best model
        best_model_path = args.path + args.best_model
        list_best_model = pickle.load(open(best_model_path, "rb"))

        func = partial(test_reads_majority, list_best_model, args.path, args.type_factorization, args.k_type, args.k_value, args.criterion, args.denoise)

        # Results txt file
        test_result_file = open(args.path + "test_majority_result_" + args.criterion + "_" + args.filter + ".txt", 'w')

        test_fingerprint_fact_lines = []
        for res in pool.map(func, splitted_blocks):
            test_fingerprint_fact_lines = test_fingerprint_fact_lines + res
        ################################################################################################################

        test_result_file.writelines(test_fingerprint_fact_lines)
        test_result_file.close()


# Given a set of reads, and a RF fingerprint classifier trained, performs classification
# args.step = 'test_RF_fingerprint' ####################################################################################
def testing_reads_RF_fingerprint_step(args):

    # Input fasta
    input_fasta = args.path + args.fasta

    # Extract of reads (Format = ID GENE read)
    read_lines = extract_reads(name_file=input_fasta, filter=args.filter, step='test')

    read_lines = [s.upper() for s in read_lines]

    if len(read_lines) == 0:
        print('No reads extracted!')
        exit(-1)

    if args.random == 'random':
        # Randomly extraction of 1000 reads from read_lines
        random_read_lines = []
        for i in range(1000):
            random_line = random.choice(read_lines)
            random_read_lines.append(random_line)

        read_lines = random_read_lines

    print('# READS: ', len(read_lines))

    # SPLIT for multiprocessing
    size = int(len(read_lines)/args.n)
    splitted_lines = [read_lines[i:i + size] for i in range(0, len(read_lines), size)]

    fingerprint_blocks = []
    with Pool(args.n) as pool:
        func = partial(compute_fingerprint_by_list, args.fact, args.shift, args.type_factorization)

        for res in pool.map(func, splitted_lines):

            fingerprint_blocks.append((res[0], res[1]))

        ################################################################################################################

    # SPLIT fingerprint blocks
    size = int(len(fingerprint_blocks) / args.n)
    splitted_blocks = [fingerprint_blocks[i:i + size] for i in range(0, len(fingerprint_blocks), size)]

    with Pool(args.n) as pool:

        # RF fingerprint model
        rf_fingerprint_model_path = args.path + args.rf_fingerprint_model
        list_rf_fingerprint_model = pickle.load(open(rf_fingerprint_model_path, "rb"))

        func = partial(test_reads_rf_fingerprint, list_rf_fingerprint_model)

        # Results txt file
        test_result_file = open(args.path + "test_rf_fingerprint_result_" + args.filter + ".txt", 'w')

        test_fingerprint_fact_lines = []
        for res in pool.map(func, splitted_blocks):
            test_fingerprint_fact_lines = test_fingerprint_fact_lines + res
        ################################################################################################################

        test_result_file.writelines(test_fingerprint_fact_lines)
        test_result_file.close()


########################################################################################################################
##################################################### MAIN #############################################################
########################################################################################################################
if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Gestione argomenti ###############################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', dest='step', action='store', default="1rf")
    parser.add_argument('--path', dest='path', action='store', default="fingerprint/test/")
    parser.add_argument('--fasta', dest='fasta', action='store', default="example_sample_10M_genes.fastq.gz")
    parser.add_argument('--fact', dest='fact', action='store', default='create')
    parser.add_argument('--shift', dest='shift', action='store', default='no_shift')
    parser.add_argument('--best_model', dest='best_model', action='store', default='RF_CFL_ICFL-20_K8.pickle')
    parser.add_argument('--rf_fingerprint_model', dest='rf_fingerprint_model', action='store', default='RF_fingerprint_classifier_ICFL_COMB.pickle')
    parser.add_argument('--k_type', dest='k_type', action='store', default='extended')
    parser.add_argument('--k_value', dest='k_value', action='store', default=3, type=int)
    parser.add_argument('--filter', dest='filter', action='store', default='no_list')
    parser.add_argument('--criterion', dest='criterion', action='store', default='majority')
    parser.add_argument('--random', dest='random', action='store', default='random')
    parser.add_argument('--n_for_genes', dest='n_for_genes', action='store', default=10, type=float)
    parser.add_argument('--type_factorization', dest='type_factorization', action='store', default="CFL")
    parser.add_argument('-n', dest='n', action='store', default=1, type=int)

    args = parser.parse_args()

    # TEST READS with RF FINGERPRINT Classifier (# class = # genes in the list)
    if args.step == 'test_RF_fingerprint':
        print('\nTesting Step: TEST READS with RF FINGERPRINT Classifier...\n')
        testing_reads_RF_fingerprint_step(args)

    # TEST set of READS with MAJORITY on Best k-finger classification (# class = # genes in the list)
    elif args.step == 'test_majority':
        print('\nTesting Step: TEST set of READS with MAJORITY on Best k-finger classification...\n')
        testing_reads_majority_step(args)