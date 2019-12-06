# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import configparser
import matplotlib.pyplot as plt
import numpy as np
import logging
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-F", "--force_training", action="store_true", help="force to run training even if save file exists")
args = arg_parser.parse_args()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def show_img(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def print_result(source_img, source_vec, watermarked_img, recon_vec):
	images = torch.cat((source_img, watermarked_img))
	show_img(torchvision.utils.make_grid(images))
	print('Original Vector and Reconstructed Vector:\n{}'.format(torch.cat((source_vec, recon_vec), dim=1)))

class _Dataset(object):
	preprocess = [transforms.ToTensor()]
	iter = None

	def __init__(self, vec_size):
		self.vec_size = vec_size
		
	def __iter__(self):
		return self
	
	def __next__(self):
		img, _ = self.iter.next()
		if img.size()[0] != 32:
			raise StopIteration
		return img, torch.randint(2, (32, self.vec_size), dtype=torch.float)


class TrainDataset(_Dataset):
	augment = []
	
	def __init__(self, vec_size, batch_size):
		super(TrainDataset, self).__init__(vec_size)
		self.train = True
		transform = transforms.Compose(self.augment + self.preprocess)
		self.dataset = torchvision.datasets.CIFAR10(root='./data', train=self.train,
                                        download=True, transform=transform)
		self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
											shuffle=True, num_workers=0)
		self.iter = iter(self.loader)

class TestDataset(_Dataset):
	def __init__(self, vec_size, batch_size):
		super(TestDataset, self).__init__(vec_size)
		self.train = False
		transform = transforms.Compose(self.preprocess)
		self.dataset = torchvision.datasets.CIFAR10(root='./data', train=self.train,
											download=True, transform=transform)
		self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
											shuffle=False, num_workers=0)
		self.iter = iter(self.loader)

class Marker(nn.Module):
	'''Net input : Image, vector
	Net output : Image includes vector'''
	def __init__(self, vec_size):
		super(Marker, self).__init__()
		# Contracting
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5 + vec_size, 16 * 5 * 5)
		# Expanding
		self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
		self.upsample = nn.Upsample(scale_factor=2)
		self.deconv2 = nn.ConvTranspose2d(6, 3, 5)
	
	def forward(self, img, vecs):
		new_img = self.pool(F.relu(self.conv1(img)))
		new_img = self.pool(F.relu(self.conv2(new_img)))
		new_img = new_img.view(-1, 16 * 5 * 5)
		new_img = torch.cat((vecs, new_img), dim=1)
		new_img = F.relu(self.fc1(new_img))
		new_img = torch.reshape(new_img, (-1, 16, 5, 5))
		new_img = F.relu(self.deconv1(self.upsample(new_img)))
		new_img = self.deconv2(self.upsample(new_img))
		#new_img = torch.add(F.sigmoid(new_img), img)
		new_img = F.sigmoid(new_img)
		return new_img
	
class Unmarker(nn.Module):
	'''Net input : Image includes vector
	Net output : Image, vector'''
	def __init__(self, vec_size):
		super(Unmarker, self).__init__()
		self.vec_size = vec_size
		# Contracting
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, vec_size + 16 * 5 * 5)
		# Expanding
		self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
		self.upsample = nn.Upsample(scale_factor=2)
		self.deconv2 = nn.ConvTranspose2d(6, 3, 5)

	def forward(self, img):
		new_img = self.pool(F.relu(self.conv1(img)))
		new_img = self.pool(F.relu(self.conv2(new_img)))
		new_img = new_img.view(-1, 16 * 5 * 5)
		new_img = self.fc1(new_img)
		vecs, new_img = tuple(torch.split(new_img, [self.vec_size, 16 * 5 * 5], dim=1))
		vecs = F.sigmoid(vecs)
		new_img = F.relu(new_img)
		new_img = torch.reshape(new_img, (-1, 16, 5, 5))
		new_img = F.relu(self.deconv1(self.upsample(new_img)))
		new_img = self.deconv2(self.upsample(new_img))
		new_img = torch.add(F.sigmoid(new_img), img)
		return new_img, vecs

class Net(nn.Module):
	def __init__(self, vec_size):
		super(Net, self).__init__()
		self.marker = Marker(vec_size)
		self.unmarker = Unmarker(vec_size)

	def forward(self, img, vec):
		watermarked_img = self.marker.forward(img, vec)
		recon_img, recon_vecs = self.unmarker.forward(watermarked_img)
		return recon_img, recon_vecs, watermarked_img


def main():
	logging.info('Session start.')
	session_config = configparser.ConfigParser()
	session_config.read('session.ini')

	# Prepare dataset
	VEC_SIZE = int(session_config['dimensions']['vec_size'])
	train_dataset = TrainDataset(vec_size=VEC_SIZE, batch_size=32)
	logging.info('Loaded training dataset.')
	#device = torch.device("cuda:0" if torch.cuda_is_available() else "cpu")
	net = Net(VEC_SIZE)

	# Define loss
	img_loss = nn.MSELoss()
	vec_loss = nn.MSELoss()
	vec_weight = float(session_config['hyperparameters']['vec_weight'])
	learning_rate = 1E-03
	total_loss = (lambda input, output: (img_loss(input[0], output[0]) + vec_weight * vec_loss(input[1], output[1])))
	from itertools import chain
	optimizer = optim.Adam(chain(net.marker.parameters(), net.unmarker.parameters()), lr=learning_rate)
	
	save_file_path = session_config['save_files']['state_dict']
	if os.path.isfile(save_file_path) and not args.force_training:
		logging.info('Found saved session. Loading...')
		net.load_state_dict(torch.load(save_file_path))
	else:
		for epoch in range(1):
			loss_count = 0
			train_loss_sum = 0.0
			for i, data in enumerate(train_dataset):
				input_img, input_vecs = data

				optimizer.zero_grad()

				recon_img, recon_vecs, watermarked_img = net(input_img, input_vecs)
				loss = total_loss((input_img, input_vecs), (watermarked_img, recon_vecs))

				loss.backward()
				optimizer.step()

				# print statistics
				train_loss_sum += loss.item()
				loss_count += 1
				if (i+1) % 100 == 0:
					logging.debug('(Epoch {}, step {})\ninput_vecs[0] : {}\nreconv_vecs[0] : {}'.format(epoch, i+1, torch.narrow(input_vecs, 0, 0, 1), torch.narrow(recon_vecs, 0, 0, 1)))
					logging.info('(Epoch {}, step {}) loss : {}'.format(epoch, i+1, train_loss_sum/loss_count))
					train_loss_sum = 0.0
					loss_count = 0
		torch.save(net.state_dict(), save_file_path)
	logging.info('Training ended.')
	test_dataset = TestDataset(vec_size=VEC_SIZE, batch_size=32)
	logging.info('Loaded test dataset.')
	input_img, input_vecs = next(test_dataset)
	recon_img, recon_vecs, watermarked_img = net(input_img, input_vecs)

	print_result(input_img, input_vecs, watermarked_img, recon_vecs)

if __name__ == '__main__':
	main()