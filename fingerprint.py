import argparse

from fingerprint_utils import extract_reads,compute_fingerprint_by_list
from multiprocessing.pool import ThreadPool as Pool
from functools import partial


# Create fingerprint files (args.step = 'fingerprint') #################################################################
def experiment_fingerprint_1f_np_step(args):

    # Input FASTA file containing transcripts
    input_fasta = args.path + args.fasta

    # Extract of reads (Format = ID GENE read)
    read_lines = extract_reads(name_file=input_fasta, filter=args.filter,step='fingerprint')

    if len(read_lines) == 0:
        print('No reads extracted!')
        exit(-1)

    print('\nCompute fingerprint by list (%s, %s) - start...' % (args.type_factorization, args.fact))

    fingerprint_file = open("%s" % args.path + "fingerprint_" + args.type_factorization + ".txt", 'w')
    fact_fingerprint_file = None
    if args.fact == 'create':
        # Create file containing factorizations
        fact_fingerprint_file = open("%s" % args.path + "fact_fingerprint_" + args.type_factorization + ".txt", 'w')

    # SPLIT for multiprocessing
    size = int(len(read_lines)/args.n)
    splitted_lines = [read_lines[i:i + size] for i in range(0, len(read_lines), size)]

    with Pool(args.n) as pool:

        func = partial(compute_fingerprint_by_list, args.fact, args.shift, args.type_factorization)

        fingerprint_lines = []
        fingerprint_fact_lines = []
        for res in pool.map(func, splitted_lines):

            fingerprint_lines = fingerprint_lines + res[0]
            fingerprint_fact_lines = fingerprint_fact_lines + res[1]

        fingerprint_file.writelines(fingerprint_lines)
        if args.fact == 'create':
            fact_fingerprint_file.writelines(fingerprint_fact_lines)

        fingerprint_file.close()

        if args.fact == 'create':
            fact_fingerprint_file.close()

        print('\nCompute fingerprint by list (%s, %s) - stop!' % (args.type_factorization, args.fact))

########################################################################################################################


##################################################### MAIN #############################################################
########################################################################################################################
if __name__ == '__main__':

    # Gestione argomenti ###############################################################################################
    parser = argparse.ArgumentParser()

    # args.type =
    parser.add_argument('--type', dest='type', action='store', default="1f_np")
    parser.add_argument('--path', dest='path', action='store', default="training/")
    parser.add_argument('--type_factorization', dest='type_factorization', action='store', default="CFL")
    parser.add_argument('--fasta', dest='fasta', action='store', default="reads_150.fa")
    parser.add_argument('--fact', dest='fact', action='store', default='no_create')
    parser.add_argument('--shift', dest='shift', action='store', default='no_shift')
    parser.add_argument('--filter', dest='filter', action='store', default='list')
    parser.add_argument('-n', dest='n', action='store', default=1, type=int)

    args = parser.parse_args()

    if args.type == '1f_np':
        print('\nFingerprint Step: 1f_np...\n')
        experiment_fingerprint_1f_np_step(args)
