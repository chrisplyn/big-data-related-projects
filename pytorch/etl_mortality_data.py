import os
import pickle
import pandas as pd
import numpy as np

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
    icd9_str = str(icd9_object)
    if icd9_str[0] == 'E':
        converted = icd9_str[0:4]
    else:
        converted = icd9_str[0:3]

    return converted


def build_codemap():
    df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
    df_digits = df_icd9['ICD9_CODE'].apply(convert_icd9)
    keys = df_digits.unique()
    codemap = dict(zip(keys, np.arange(len(keys))))
    return codemap


def create_dataset(path, codemap):
    df_mortality = pd.read_csv(os.path.join(PATH_TRAIN, "MORTALITY.csv"))
    df_admission = pd.read_csv(os.path.join(PATH_TRAIN, "ADMISSIONS.csv"))
    df_diagnoses = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"))

    df_diagnoses['ICD9_CODE'] = df_diagnoses['ICD9_CODE'].apply(convert_icd9)
    df_diagnoses['featureID'] = df_diagnoses['ICD9_CODE'].map(codemap)
    df_diagnoses = df_diagnoses[pd.notnull(df_diagnoses['ICD9_CODE'])]
    
    aggregatedByVisit = pd.DataFrame(df_diagnoses.groupby(['HADM_ID'])['featureID'].apply(list))
    aggregatedByVisit['HADM_ID'] = aggregatedByVisit.index

    df_admission = df_admission[["SUBJECT_ID", "HADM_ID", "ADMITTIME"]]
    aggregatedByVisit = aggregatedByVisit.join(df_admission.set_index("HADM_ID"), how = "left")
    
    aggregatedByPatient = pd.DataFrame(aggregatedByVisit.sort_values(['SUBJECT_ID','ADMITTIME'],ascending=True).groupby('SUBJECT_ID')['featureID'].apply(list))
    aggregatedByPatient['SUBJECT_ID'] = aggregatedByPatient.index
    
    aggregatedByPatient = aggregatedByPatient.set_index("SUBJECT_ID").join(df_mortality.set_index("SUBJECT_ID"), how = "left")
    
    patient_ids = list(aggregatedByPatient.index.values)
    labels = list(aggregatedByPatient['MORTALITY'].values)
    seq_data = list(aggregatedByPatient['featureID'].values)
    return patient_ids, labels, seq_data


def main():
	# Build a code map from the train set
	print("Build feature id map")
	codemap = build_codemap()
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
