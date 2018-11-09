import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset


def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.
	input = pd.read_csv(path)

	if model_type == 'MLP':
		target = torch.tensor((input['y']-1).values)
		data = torch.tensor(input.drop('y', axis=1).values.astype(np.float32))
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = input.loc[:, 'X1':'X178'].as_matrix()
		target = torch.tensor((input['y']-1).values)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(1),target)
	elif model_type == 'RNN':
		data = input.loc[:, 'X1':'X178'].as_matrix()
		target = torch.tensor((input['y']-1).values)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2), target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		def constructSparseFeatureMatrix(singlePatientSeq):
			numVisits = len(singlePatientSeq)
			rowInd = []
			colInd = []
			for i in range(numVisits):
				rowInd.extend([i]*len(singlePatientSeq[i]))
				colInd.extend(singlePatientSeq[i])
				
			data = [1]*len(colInd)
			return sparse.coo_matrix((data, (rowInd, colInd)), shape=(numVisits, num_features))
		
		self.seqs = [constructSparseFeatureMatrix(seq) for seq in seqs]  # replace this with your implementation.

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence

    def collateSequence(sequences):
        seq_len = [seq.size()[0] for seq in sequences]
		num_features = sequences[0].size()[1]
        collated_seqs = torch.zeros([len(sequences), max(seq_len), num_features],dtype=torch.float32)
		
        for i, seq in enumerate(sequences):
            end = seq_len[i]
            collated_seqs[i, :end, num_features] = seq[i, :end, num_features]
        return collated_seqs, torch.LongTensor(seq_len)
		
	batch.sort(key=lambda x: x[0].size()[0], reverse=True)

    # seperate source and target sequences
    seqs_tensor, labels_tensor = zip(*batch)	
	seqs_tensor, lengths_tensor = collateSequence(seqs_tensor)

	return (seqs_tensor, lengths_tensor), torch.LongTensor([x.item() for x in labels_tensor])
