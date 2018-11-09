import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		# self.fc1 = nn.Linear(178, 16)
		# self.fc2 = nn.Linear(16, 5)
		self.linear1 = nn.Linear(178, 512)
		self.linear2 = nn.Linear(512, 256)
		self.linear3 = nn.Linear(256, 5)
		self.dropout1 = nn.Dropout(p=0.25)
		self.dropout2 = nn.Dropout(p=0.25)
		self.bn1 = nn.BatchNorm1d(178)
		self.bn2 = nn.BatchNorm1d(512)
		
	def forward(self, x):
		# x = self.fc1(x)
		# x = F.sigmoid(x)
		# x = self.fc2(x)
		# return x
		x = F.relu(self.dropout1(self.linear1(self.bn1(x))))
		x = F.relu(self.dropout2(self.linear2(self.bn2(x))))
		out = self.linear3(x)
		return out


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.dropout = nn.Dropout(p=0.3)
		self.conv2 = nn.Conv1d(6, 16, 5)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 5)
		
	def forward(self, x):
		x = self.pool(F.relu(self.dropout(self.conv1(x))))
		x = self.pool(F.relu(self.dropout(self.conv2(x))))
		x = x.view(-1, 16 * 41)
		x = F.relu(self.dropout(self.fc1(x)))
		x = F.relu(self.dropout(self.fc2(x)))
		x = self.fc3(x)
		return x

class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=3, dropout=0.4, batch_first=True)
		self.fc = nn.Linear(in_features=32, out_features=5)

	def forward(self, x):
		x, _ = self.rnn(x)
		x = F.tanh(x[:, -1, :])
		x = self.fc(x)
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc1 = nn.Linear(in_features=dim_input, out_features=32)
		self.rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
		self.fc2 = nn.Linear(in_features=16, out_features=2)
		
	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		seqs, lengths = input_tuple
		seqs = F.tanh(self.fc1(seqs))
		packed_input = pack_padded_sequence(seqs, lengths, batch_first=True)
		rnn_output, _ = self.rnn(packed_input)
		padded_output, _ = pad_packed_sequence(rnn_output)
		padded_output = padded_output[np.arange(len(padded_output)),lengths-1]
		out = self.fc2(F.tanh(padded_output))
		return out