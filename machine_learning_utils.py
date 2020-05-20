import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from fingerprint_utils import computeWindow


# Mapping pool for training in multiprocessing
def mapping_pool_train(path="fingerprint/ML/", tuple_fact_k=('CFL',3, 'RF')):
    train(path, tuple_fact_k[0], tuple_fact_k[1], tuple_fact_k[2])


# Training of classifiers
def train(path="fingerprint/ML/", type_factorization='CFL', k=3, type_model = 'RF'):

    if type_model == 'RF':
        random_forest_kfinger(path=path, type_factorization=type_factorization, k=k)
    elif type_model == 'NB':
        multinomial_NB_model(path=path, type_factorization=type_factorization, k=k)
    elif type_model == 'Logistic':
        logistic_regression(path=path, type_factorization=type_factorization, k=k)
    elif type_model == 'RF_FINGERPRINT':
        random_forest_fingerprint(path=path, type_factorization=type_factorization)


# Split of the dataset
def train_test_generator(dataset_name):
    
    X = pickle.load(open(dataset_name,"rb"))
    y_dataset = dataset_name.replace("X", "y")
    y = pickle.load(open(y_dataset,"rb"))

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),)

    training_set_data,test_set_data,training_set_labels,test_set_labels = train_test_split(X,integer_encoded,test_size=0.7,stratify=integer_encoded)

    scaler = MinMaxScaler()
    train_scaled_D = scaler.fit_transform(training_set_data)
    test_scaled_D = scaler.transform(test_set_data)

    return (train_scaled_D,test_scaled_D,training_set_labels,test_set_labels, label_encoder,scaler)
    

# Compute the classification thresholds for each class gene
def compute_classification_thresholds(model=None, test_set=None, labels=None,clsf=None):

    # Probability predictions (matrix x_samples X n_genes)
    prediction_proba = model.predict_proba(test_set)
    prediction_proba = np.array(prediction_proba)

    # Real prediction (array 1 X n_genes)
    prediction = model.predict(test_set)
    prediction = prediction.tolist()

    thresholds = []

    # For each class
    for i in range(len(labels)):
        samples_predicted_for_i = []
        for j in range(len(prediction)):
            sample = prediction[j]
            if sample == i:
                samples_predicted_for_i.append((prediction_proba[j])[i])

        threshold = np.amin(samples_predicted_for_i, axis=0)
        thresholds = np.append(thresholds, [threshold])

    for lbl, threshold in zip(labels,thresholds):
        dict_item = clsf[lbl]
        new_dict={'precision':dict_item['precision'],'recall':dict_item['recall'],'f1-score':dict_item['f1-score'],'support':dict_item['support'],'threshold':threshold}
        clsf[lbl] = new_dict

    return clsf


# Multinomial Naive Bayesian (MNB) model
def multinomial_NB_model(path="training/", type_factorization='CFL', k=3):

    print('\nTrain MNB (%s, %s) - start...' % (type_factorization, k))

    # Dataset
    dataset_name = path + "dataset_X_" + type_factorization +"_K" + str(k) + ".pickle"
    train_scaled_D,test_scaled_D, training_set_labels, test_set_labels, label_encoder, min_max_scaler = train_test_generator(dataset_name)

    n_genes = len(set(training_set_labels))
    classificatore=MultinomialNB()
    classificatore.fit(train_scaled_D, training_set_labels)
    
    labels_originarie = label_encoder.inverse_transform(np.arange(n_genes))
    y_pred = classificatore.predict(test_scaled_D)

    clsf = classification_report(test_set_labels, y_pred, target_names=labels_originarie, output_dict=True)

    # Compute thresholds
    clsf = compute_classification_thresholds(model=classificatore, test_set=test_scaled_D, labels=labels_originarie, clsf=clsf)
    clsf_report = pd.DataFrame(data=clsf).transpose()
    csv_name = path + "NB_clsf_report_" + type_factorization + "_K" + str(k) + ".csv"
    clsf_report.to_csv(csv_name, index= True)

    acc = accuracy_score(test_set_labels,classificatore.predict(test_scaled_D))
    print(" MultinomialNB accuracy: ", accuracy_score(test_set_labels,classificatore.predict(test_scaled_D)))

    print('\nTrain MNB (%s, %s) - stop!' % (type_factorization, k))


# Logistic Regression (LR) model
def logistic_regression(path="training/", type_factorization='CFL', k=3):

    print('\nTrain Logistic regression (%s, %s) - start...' % (type_factorization, k))

    # Create name dataset
    dataset_name = path + "dataset_X_" + type_factorization + "_K" + str(k) + ".pickle"
    train_scaled_D,test_scaled_D, training_set_labels, test_set_labels, label_encoder, min_max_scaler = train_test_generator(dataset_name)
    n_geni = len(set(training_set_labels))
    param_grid = [{'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}]

    classificatore = GridSearchCV(LogisticRegression(random_state=0, max_iter=6000), param_grid, scoring='accuracy', cv=2)
    classificatore.fit(train_scaled_D, training_set_labels)

    labels_originarie = label_encoder.inverse_transform(np.arange(n_geni))
    y_pred = classificatore.predict(test_scaled_D)

    clsf = classification_report(test_set_labels, y_pred, target_names=labels_originarie, output_dict=True)
    clsf = compute_classification_thresholds(model=classificatore, test_set=test_scaled_D, labels=labels_originarie, clsf=clsf)
    clsf_report = pd.DataFrame(data=clsf).transpose()
    csv_name = path + "Logistic_clsf_report_" + type_factorization + "_K" + str(k) + ".csv"
    clsf_report.to_csv(csv_name, index= True)

    acc = accuracy_score(test_set_labels,classificatore.predict(test_scaled_D))
    print("Logistic regression accuracy: ", acc)

    print('\nTrain Logistic regresssion (%s, %s) - stop!' % (type_factorization, k))


# Random forest k_finger classifier
def random_forest_kfinger(path="training/", type_factorization='CFL', k=8):

    print('\nTrain RF k_finger classifier (%s, %s) - start...' % (type_factorization, k))

    # Create name dataset
    dataset_name = path + "dataset_X_" + type_factorization + "_K" + str(k) + ".pickle"
    train_scaled_D,test_scaled_D, training_set_labels, test_set_labels, label_encoder, min_max_scaler = train_test_generator(dataset_name)

    n_genes = len(set(training_set_labels))
    classificatore = RandomForestClassifier(n_estimators=8, min_samples_leaf=1, n_jobs=-1)
    classificatore.fit(train_scaled_D, training_set_labels)
    
    labels_originarie = label_encoder.inverse_transform(np.arange(n_genes))
    y_pred = classificatore.predict(test_scaled_D)

    clsf = classification_report(test_set_labels, y_pred, target_names =labels_originarie, output_dict=True)
    clsf = compute_classification_thresholds(model=classificatore, test_set=test_scaled_D, labels=labels_originarie, clsf=clsf)
    clsf_report = pd.DataFrame(data=clsf).transpose()
    csv_name = path + "RF_kfinger_clsf_report_" + type_factorization + "_K" + str(k) + ".csv"
    clsf_report.to_csv(csv_name, index= True)

    #print("Random Forest accuracy: ", accuracy_score(test_set_labels,classificatore.predict(test_scaled_D)))

    # Pickle [RF model, labels_originarie,
    pickle.dump([classificatore,label_encoder, min_max_scaler], open(path + "RF_" + type_factorization + "_K" + str(k) + ".pickle", 'wb'))

    print('\nTrain RF k_finger classifier (%s, %s) - stop!' % (type_factorization, k))


# Given a dataset, train a RF fingerprint classifier
def random_forest_fingerprint(path="training/", type_factorization='CFL'):

    print('\nTrain RF fingerprint classifier (%s) - start...' % (type_factorization))

    input_fingerprint = path + 'fingerprint_' + type_factorization + '.txt'
    fingerprint_file = open(input_fingerprint)
    fingerprints = fingerprint_file.readlines()

    # Build dataset
    y = []
    X = []
    count_lenghts_fingerprint = 0
    for fingerprint in fingerprints:
        lengths = fingerprint
        lengths_list = lengths.split()

        y_i = lengths_list[0]
        x_i = lengths_list[1:]

        count_lenghts_fingerprint += len(x_i)

        X.append(x_i)
        y.append(y_i)

    mean_length = int(count_lenghts_fingerprint / len(fingerprints))

    # Padding
    X_padded = []
    for x_i in X:

        if len(x_i) < mean_length:
            for i in range(len(x_i), mean_length):
                x_i = np.append(x_i, ['-1'])

            x_i = x_i.tolist()

        elif len(x_i) >= mean_length:
            x_i = x_i[:mean_length]

        X_padded.append(x_i)

    # Split dataset

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), )
    training_set_data, test_set_data, training_set_labels, test_set_labels = train_test_split(X_padded, integer_encoded,
                                                                                              stratify=integer_encoded)
    scaler = MinMaxScaler()
    train_scaled_D = scaler.fit_transform(training_set_data)
    test_scaled_D = scaler.transform(test_set_data)

    ####################################################################################################################
    n_geni = len(set(training_set_labels))
    classificatore=RandomForestClassifier(n_estimators=8,min_samples_leaf=1,n_jobs=-1)
    classificatore.fit(train_scaled_D, training_set_labels)

    labels_originarie = label_encoder.inverse_transform(np.arange(n_geni))
    y_pred = classificatore.predict(test_scaled_D)

    clsf = classification_report(test_set_labels, y_pred, target_names=labels_originarie, output_dict=True)
    clsf = compute_classification_thresholds(model=classificatore, test_set=test_scaled_D, labels=labels_originarie,clsf=clsf)
    clsf_report = pd.DataFrame(data=clsf).transpose()
    csv_name = path + "RF_fingerprint_clsf_report_" + type_factorization + ".csv"
    clsf_report.to_csv(csv_name, index=True)

    acc = accuracy_score(test_set_labels, classificatore.predict(test_scaled_D))
    #print('RF fingerprint classifier accuracy: ', acc)

    # Dump pickle
    pickle.dump([classificatore, label_encoder, scaler, mean_length], open(path + "RF_fingerprint_classifier_" + type_factorization + ".pickle", 'wb'))

    print('\nTrain RF fingerprint classifier (%s) - stop!' % (type_factorization))


# RULE-BASED READ CLASSIFIER
# Given a set of reads, performs classification by using the majority (or thresholds) criterion on  Best k-finger classification
def test_reads_majority(list_best_model=None, path='testing/', type_factorization='CFL_ICFL-20', k_window='extended',k_value=8,criterion='majority', fingerprint_block = []):

    print('\nRule-based read classifier - start...')
    test_lines = []

    # best model
    best_model = list_best_model[0]
    best_label_encoder = list_best_model[1]
    best_min_max_scaler = list_best_model[2]

    # Load thresholds
    data = pd.read_csv(path + "RF_kfinger_clsf_report_" + type_factorization + "_K" + str(k_value) + ".csv")
    df_threshold = data['threshold']
    df_threshold = df_threshold[:len(df_threshold) - 3]
    np_threshold = np.array(df_threshold)
    min_threshold = np.min(np_threshold)

    for fingerprint in fingerprint_block[0]:
        lengths_list = fingerprint.split()

        k_fingers = computeWindow(lengths_list[1:], k_value, k_window=k_window)

        # Scaler tests set
        test_scaled_D = best_min_max_scaler.transform(k_fingers)
        y_pred = best_model.predict(test_scaled_D)

        id_pred = -1
        if criterion == 'majority':
            ############################################################################################################

            counts = np.bincount(y_pred)
            max_element = np.argmax(counts)
            count_max_element = np.count_nonzero(y_pred == max_element)

            unique_max = True
            for a in y_pred:
                if a != max_element:
                    count_a_element = np.count_nonzero(y_pred == a)
                    if count_a_element == count_max_element:
                        unique_max = False
                        break

            if (unique_max == True) and (count_max_element >= (len(y_pred.tolist())/2)):
                # Absolute majority
                id_pred = max_element
                fingerprint = fingerprint.replace('\n', '')
                test_lines.append('ID_PREDICTED: ' + str((best_label_encoder.inverse_transform([id_pred]))[0]) + ' - ID_LABEL: ' + fingerprint + ' - PREDICTION: ' + str(y_pred) + '\n')
            else:

                y_pred_proba = best_model.predict_proba(test_scaled_D)

                max_index = []
                max_diff = []
                for sample in y_pred_proba:
                    sample_diff_threshold = [(float(i) - min_threshold) for i in sample]

                    max_ind = np.argmax(np.array(sample_diff_threshold))
                    max_index.append(max_ind)

                    max_d = np.amax(np.array(sample_diff_threshold))
                    max_diff.append(max_d)

                max_th_index = np.argmax(np.array(max_diff))
                id_pred = max_index[max_th_index]

                fingerprint = fingerprint.replace('\n', '')
                test_lines.append('ID_PREDICTED: ' + str((best_label_encoder.inverse_transform([id_pred]))[0]) + ' - ID_LABEL: ' + fingerprint + ' - PREDICTION: ' + str(y_pred) + ' - MAX_DIFF: ' + str(max_diff) + '\n')

            ############################################################################################################
        else:

            y_pred_proba = best_model.predict_proba(test_scaled_D)

            max_index = []
            max_diff = []
            for sample in y_pred_proba:
                sample_diff_threshold = [(float(i) - min_threshold) for i in sample]

                max_ind = np.argmax(np.array(sample_diff_threshold))
                max_index.append(max_ind)

                max_d = np.amax(np.array(sample_diff_threshold))
                max_diff.append(max_d)

            max_th_index = np.argmax(np.array(max_diff))
            id_pred = max_index[max_th_index]

            fingerprint = fingerprint.replace('\n','')
            test_lines.append('ID_PREDICTED: ' + str((best_label_encoder.inverse_transform([id_pred]))[0]) + ' - ID_LABEL: ' + fingerprint + ' - PREDICTION: ' + str(y_pred) + ' - MAX_DIFF: ' + str(max_diff) + '\n')

    print('\nRule-based read classifier - stop!')

    return test_lines


# Given a set of reads, and a RF fingerprint classifier trained, performs classification
def test_reads_rf_fingerprint(list_rf_fingerprint_model=None, fingerprint_block = []):

    print('\nTest reads RF Fingerprint - start...')
    test_lines = []

    # best model
    rf_fingerprint_model = list_rf_fingerprint_model[0]
    rf_fingerprint_encoder = list_rf_fingerprint_model[1]
    rf_fingerprint_scaler = list_rf_fingerprint_model[2]
    rf_fingerprint_mean_length = list_rf_fingerprint_model[3]

    for fingerprint in fingerprint_block[0]:

        lengths_list = fingerprint.split()

        fing = lengths_list[1:]

        # Padding with -1 values
        if len(fing) < rf_fingerprint_mean_length:
            for i in range(len(fing), rf_fingerprint_mean_length):
                fing = np.append(fing, ['-1'])

            fing = fing.tolist()
        elif len(fing) >= rf_fingerprint_mean_length:
            fing = fing[:rf_fingerprint_mean_length]

        # Read_rf classification
        test_scaled_D = rf_fingerprint_scaler.transform([fing])
        read_y_pred = rf_fingerprint_model.predict(test_scaled_D)

        fingerprint = fingerprint.replace('\n', '')
        label = str((rf_fingerprint_encoder.inverse_transform(read_y_pred))[0])
        test_lines.append('ID_PREDICTED: ' + label + ' - ID_LABEL: ' + fingerprint + ' - PREDICTION: ' + str(read_y_pred) + '\n')

    print('\nTest reads  RF Fingerprint - stop!')

    return test_lines

