import pickle
import gzip
import random
import numpy as np

from factorizations import CFL
from factorizations import ICFL_recursive
from factorizations import CFL_icfl
from factorizations_comb import d_cfl
from factorizations_comb import d_icfl
from factorizations_comb import d_cfl_icfl
from factorizations_comb import reverse_complement

# Given a list of lengths computes the list of k-fingers
def computeWindow(lista,k,k_window='valid',facts_list=None):

    if len(lista) < k:
        if k_window == 'extended':
            len_lista = len(lista)
            for i in range(len_lista,k):
                #lista.append(-1)
                lista = np.append(lista, ['-1'])
                lista = lista.tolist()
                if facts_list != None:
                    facts_list.append('')

    toReturn = []
    for e in range(0,len(lista[:-(k-1)])):
            k_finger = lista[e:e+k]

            enrich_str = None
            if facts_list != None:
                enrich_str = get_enrich_str(facts_list[e:e+k])

            # Normalization of k_finger
            k_finger = normalize(k_finger)

            if facts_list != None:
                k_finger.append(enrich_str)

            toReturn.append(k_finger)

    return toReturn

# Given a k_finger, return the enriched string
def get_enrich_str(facts_list):

    enrich_str = None
    if len(facts_list) > 2:
        facts_list.pop(0)
        facts_list.pop()

        if len(facts_list) == 1:
            # Only 1 element
            if len(facts_list[0]) <= 20:
                enrich_str = reverse_complement(facts_list[0])
            else:
                enrich_str = (facts_list[0])[:10] + (facts_list[0])[(len(facts_list[0]) - 10):]
                enrich_str = reverse_complement(enrich_str)
        else:
            # More elements
            max = 0
            max_fact = ''
            for fact in facts_list[::-1]:
                if len(fact) > max:
                    max = len(fact)
                    max_fact = fact
            enrich_str = reverse_complement(max_fact)
            if len(enrich_str) <= 20:
                enrich_str = reverse_complement(enrich_str)
            else:
                enrich_str = (enrich_str)[:10] + (enrich_str)[(len(enrich_str) - 10):]
                enrich_str = reverse_complement(enrich_str)


    # Padding of length 20 for the enriched string
    if len(enrich_str) <= 20:
        for i in range(len(enrich_str), 20):
            enrich_str += 'N'
    return enrich_str



# Given a k_finger, returns the normalized version:
def normalize(k_finger):

    k_finger_inv = k_finger[::-1]

    for a, b in zip(k_finger, k_finger_inv):
        a = int(a)
        b = int(b)

        if a < b:
            return k_finger
        elif b < a:
            return k_finger_inv
        else:
            continue

    return k_finger



# Mapping pool for create datasets in multiprocessing
def mapping_pool_create_ML_dataset(path="fingerprint/ML/", k_window='valid', enrich='no_string', tuple_fact_k=('CFL',3)):
    create_ML_dataset(path, k_window, enrich, type_factorization= tuple_fact_k[0], k=tuple_fact_k[1])



# Given a file containing the fingerprints (IDGENE String), build the dataset for the training phase
def create_ML_dataset(path="fingerprint/ML/", k_window='valid', enrich='no_string',type_factorization='CFL', k=3):

    input_fingerprint = path + 'fingerprint_' + type_factorization + '.txt'
    input_fact = path + 'fact_fingerprint_' + type_factorization + '.txt'

    print('\nCreate ML dataset (%s, %s, %s, %s, %s) - start...' % (type_factorization,k,input_fact,k_window,enrich))

    fingerprint_file = open(input_fingerprint)
    fingerprints = fingerprint_file.readlines()

    fact_fingerprint_file = None
    fact_fingerprints = None

    if enrich == 'string':
        fact_fingerprint_file = open(input_fact)
        fact_fingerprints = fact_fingerprint_file.readlines()

    y = []
    X = []

    # Check unique sample ##############################################################################################
    file_experiment = open('list_experiment.txt')
    list_id_genes = file_experiment.readlines()
    list_id_genes = [s.replace('\n', '') for s in list_id_genes]
    dict_n_for_genes = {i: 0 for i in list_id_genes}
    ####################################################################################################################

    for i in range(0, len(fingerprints)):

        if len(fingerprints[i]) == 0:
            print('empty row')

        lengths = fingerprints[i]
        lengths_list = lengths.split()

        facts = None
        facts_list = None
        if enrich == 'string':
            facts = fact_fingerprints[i]
            facts_list = facts.split()

        class_gene = lengths_list[0]
        class_gene = str(class_gene).replace('\n', '')

        xi = None
        if enrich == 'string':
            xi = computeWindow(lengths_list[1:], k, k_window=k_window, facts_list=facts_list[1:])
        else:
            xi = computeWindow(lengths_list[1:], k, k_window=k_window)

        if xi != None:
            for j in xi:
                X.append(j)
            for j in range(len(xi)):
                y.append(class_gene)
                if class_gene in list_id_genes:
                    n_for_id_gene = dict_n_for_genes[class_gene]
                    n_for_id_gene += 1
                    dict_n_for_genes[class_gene] = n_for_id_gene

    # Chek unique sample ###############################################################################################
    for id_gene in list_id_genes:
        n_for_id_gene = dict_n_for_genes[id_gene]

        if n_for_id_gene == 1:
            # Bisogna raddoppiare il sample
            index = y.index(id_gene)
            x_sample = X[index]
            X = np.vstack((X, x_sample))
            y.append(id_gene)
    ####################################################################################################################

    print('# samples: ', len(y))

    # Dataset dump
    pickle.dump(X, open(path + "dataset_X_" + type_factorization + "_K" + str(k) + ".pickle",'wb'))
    pickle.dump(y, open(path + "dataset_y_" + type_factorization + "_K" + str(k) + ".pickle", 'wb'))

    print('\nCreate ML dataset (%s, %s, %s, %s, %s) - stop!' % (type_factorization, k, input_fact, k_window, enrich))



# Shift of size d of a string s
def shift_string(string='',size=100,shift='no_shift'):
    list_of_shift = []

    if shift == 'no_shift':
        list_of_shift.append(string)
    else:
        if len(string) < size:
            list_of_shift.append(string)
        else:
            for i in range(len(string)):
                fact = string[i:i+size]
                if i+size > len(string):
                    pref = string[:(i+size)-len(string)]
                    fact = fact + pref
                list_of_shift.append(fact)
    return list_of_shift



# Read file GZ containing reads
def read_gz(fasta_lines):

    lines = []
    read = ''
    step = ''

    i = 0
    while True:
        print(i)
        print(fasta_lines)
        # ID_GENE
        l_1 = str(fasta_lines[i])
        l_1= l_1.replace('b\'', '')
        s_l1 = l_1.split()
        if len(s_l1) == 2:
            id_gene = s_l1[1]
            id_gene = id_gene.replace('\\n', '')
            id_gene = id_gene.replace('\'', '')
            read = read + id_gene + ' '
        else:
            s_l1 = l_1.split(',')
            read = read + s_l1[1] + ' '

        # Read
        l_2 = str(fasta_lines[i+1])
        l_2 = l_2.replace('b\'', '')
        l_2 = l_2.replace('\\n', '')
        l_2 = l_2.replace('\'', '')
        read = read + l_2

        lines.append(read)
        read = ''

        i += 4
        if i == len(fasta_lines):
            break

    return lines


# Read file GZ containing reads
def read_gz_mp(fasta_lines):

    lines = []
    read = ''
    step = ''

    for fasta_line in fasta_lines:

        # ID_GENE
        l_1 = str(fasta_line[0])
        l_1= l_1.replace('b\'', '')
        s_l1 = l_1.split()
        if len(s_l1) == 2:
            id_gene = s_l1[1]
            id_gene = id_gene.replace('\\n', '')
            id_gene = id_gene.replace('\'', '')
            read = read + id_gene + ' '
        else:
            s_l1 = l_1.split(',')
            read = read + s_l1[1] + ' '

        # Read
        l_2 = str(fasta_line[1])
        l_2 = l_2.replace('b\'', '')
        l_2 = l_2.replace('\\n', '')
        l_2 = l_2.replace('\'', '')
        read = read + l_2

        lines.append(read)
        read = ''

    return lines


# Read file FASTA
def read_fasta(fasta_lines):

    lines = []
    read = ''
    step = ''
    for s in fasta_lines:
        if s[0] == '>':
            if read != '':
                lines.append(read)
                read = ''

            s = s.replace('\n', '')
            s_list = s.split()
            read = s_list[1] + ' '
        else:
            s = s.replace('\n', '')
            read += s

    return lines

# Read file FASTA
def read_fasta_mp(fasta_lines):

    lines = []
    read = ''
    step = ''
    fasta_lines = fasta_lines[0]
    for s in fasta_lines:
        if s[0] == '>':
            if read != '':
                lines.append(read)
                read = ''

            s = s.replace('\n', '')
            s_list = s.split()
            read = s_list[1] + ' '
        else:
            s = s.replace('\n', '')
            read += s

    return lines


# Given a FASTA file returns the list containing ONLY the reads, for each read, the corresponding fingerprint
def extract_reads(name_file='fingerprint/ML/reads_150.fa', filter='list', n_for_genes = None, step='fingerprint'):

    print('\nExtract reads - start...')

    # Creazione  dizionario per cpntare i geni trovati
    list_id_genes = None
    dict_n_for_genes = None
    if n_for_genes != None:
        file_experiment = open('list_experiment.txt')
        list_id_genes = file_experiment.readlines()
        list_id_genes = [s.replace('\n', '') for s in list_id_genes]
        dict_n_for_genes = {i: 0 for i in list_id_genes}
        file_experiment.close()

    file = None
    lines = []

    # Scrittura su file
    if name_file.endswith('.gz'):
        # GZ FILE
        file = gzip.open(name_file, 'rb')
        lines = read_gz(file.readlines())
    elif name_file.endswith('.fa') or name_file.endswith('.fasta') or name_file.endswith('.fastq'):
        # FASTA FILE
        file = open(name_file)
        lines = read_fasta(file.readlines())

    read_lines = []

    for s in lines:

        new_line = ''
        new_fact_line = ''

        str_line = s.split()
        id_gene = str_line[0]
        id_gene = id_gene.replace('\n', '')

        # Create lines
        file_experiment = open('list_experiment.txt')
        list_id_genes = file_experiment.readlines()
        list_id_genes = [s.replace('\n','') for s in list_id_genes]
        if filter == 'list':
            if id_gene in list_id_genes:

                ########################################################################################################
                if n_for_genes != None:
                    n_for_id_gene = dict_n_for_genes[id_gene]
                    if n_for_id_gene < n_for_genes:

                        lbl_id_gene = id_gene + ' '

                        # UPPER fingerprint
                        sequence = str_line[1]
                        if step == 'training':
                            sequence = sequence.upper()
                        new_line = lbl_id_gene + ' ' +  sequence
                        new_line += '\n'
                        read_lines.append(new_line)

                        n_for_id_gene += 1
                        dict_n_for_genes[id_gene] = n_for_id_gene
                else:
                ########################################################################################################
                    lbl_id_gene = id_gene + ' '

                    # UPPER
                    if step == 'training':
                        sequence = sequence.upper()

                    sequence = str_line[1]
                    new_line = lbl_id_gene + ' ' + sequence
                    new_line += '\n'
                    read_lines.append(new_line)
        else:
            lbl_id_gene = id_gene + ' '
            new_line = lbl_id_gene + str_line[1]
            read_lines.append(new_line)

        file_experiment.close()


    file.close()

    print('\nExtract reads - stop!')

    return read_lines


# Given a FASTA file returns the list containing ONLY the reads, for each read, the corresponding fingerprint
# Multi_processing version
def extract_reads_mp(name_file='fingerprint/ML/reads_150.fa', filter='list', step='fingerprint', n_for_genes = None, read_lines = []):

    print('\nExtract reads - start...')

    # Creazione  dizionario per cpntare i geni trovati
    list_id_genes = None
    dict_n_for_genes = None
    if n_for_genes != None:
        file_experiment = open('list_experiment.txt')
        list_id_genes = file_experiment.readlines()
        list_id_genes = [s.replace('\n', '') for s in list_id_genes]
        dict_n_for_genes = {i: 0 for i in list_id_genes}
        file_experiment.close()

    lines = []

    # Scrittura su file
    if name_file.endswith('.gz'):
        # GZ FILE
        lines = read_gz_mp(read_lines)
    elif name_file.endswith('.fa') or name_file.endswith('.fasta') or name_file.endswith('.fastq') or name_file.endswith('.txt'):
        # FASTA FILE
        lines = read_fasta_mp(read_lines)

    read_lines = []

    for s in lines:

        new_line = ''
        new_fact_line = ''

        str_line = s.split()
        id_gene = str_line[0]
        id_gene = id_gene.replace('\n', '')

        # Create lines
        file_experiment = open('list_experiment.txt')
        list_id_genes = file_experiment.readlines()
        list_id_genes = [s.replace('\n','') for s in list_id_genes]
        if filter == 'list':
            if id_gene in list_id_genes:

                ########################################################################################################
                if n_for_genes != None:
                    n_for_id_gene = dict_n_for_genes[id_gene]
                    if n_for_id_gene < n_for_genes:

                        lbl_id_gene = id_gene + ' '

                        # UPPER fingerprint
                        sequence = str_line[1]
                        if step == 'training':
                            sequence = sequence.upper()
                        new_line = lbl_id_gene + ' ' +  sequence
                        new_line += '\n'
                        read_lines.append(new_line)

                        n_for_id_gene += 1
                        dict_n_for_genes[id_gene] = n_for_id_gene
                else:
                ########################################################################################################
                    lbl_id_gene = id_gene + ' '

                    # UPPER
                    if step == 'training':
                        sequence = sequence.upper()

                    sequence = str_line[1]
                    new_line = lbl_id_gene + ' ' + sequence
                    new_line += '\n'
                    read_lines.append(new_line)
        else:
            lbl_id_gene = id_gene + ' '
            new_line = lbl_id_gene + str_line[1]
            read_lines.append(new_line)

        file_experiment.close()

    print('\nExtract reads - stop!')

    return read_lines




# Given a list of reads and a factorization technique, compute the list containing, for each read, the corresponding fingerprint
def compute_fingerprint_by_list(fact_file='no_create', shift='no_shift', factorization=CFL,T=None,list_reads=[]):
    
    print('Compute fingerprint start...')
    fingerprint_lines = []
    fingerprint_fact_lines = []

    id_gene = ''
    step = ''
    for s in list_reads:

        str_line = s.split()

        id_gene = str_line[0]
        read = str_line[1]

        list_of_shifts = shift_string(read, 100, shift)
        for sft in list_of_shifts:
            list_fact = factorization(sft,T)

            # Remove special characters
            if '>>' in list_fact:
                list_fact[:] = (value for value in list_fact if value != '>>')
            if '<<' in list_fact:
                list_fact[:] = (value for value in list_fact if value != '<<')

            # Create lines
            lbl_id_gene = id_gene + ' '
            new_line = lbl_id_gene + ' '.join(str(len(fact)) for fact in list_fact)
            new_line += '\n'
            fingerprint_lines.append(new_line)
            if fact_file == 'create':
                new_fact_line = lbl_id_gene + ' '.join(fact for fact in list_fact)
                new_fact_line += '\n'
                fingerprint_fact_lines.append(new_fact_line)
    
    print('... Compute fingerprint stop!')
    
    return fingerprint_lines,fingerprint_fact_lines

def cut_suffix_for_test(read):
    for i in range(len(read)):

        suffix = read[i:]
        # print(suffix)

        bool_3_upper = False
        # For each factor of len 3
        for j in range(len(suffix) - 2):
            factor_3 = suffix[j:j + 3]
            # print(factor_3)
            if factor_3[0].isupper() and factor_3[1].isupper() and factor_3[2].isupper():
                bool_3_upper = True
                break

        if bool_3_upper == False:
            read = read.replace(suffix, '')
            break
    return read
