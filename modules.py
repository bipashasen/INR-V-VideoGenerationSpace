import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

import clip

class BatchLinear(nn.Linear, MetaModule):
	'''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
	hypernetwork.'''
	__doc__ = nn.Linear.__doc__

	def forward(self, input, params=None):
		if params is None:
			params = OrderedDict(self.named_parameters())

		bias = params.get('bias', None)
		weight = params['weight']

		# print(weight.shape, bias.shape, input.shape, weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2).shape)
		# print(input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2)).shape)

		output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
		output += bias.unsqueeze(-2)

		return output


class Sine(nn.Module):
	def __init(self):
		super().__init__()

	def forward(self, input):
		# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
		return torch.sin(30 * input)

class FCBlock(MetaModule):
	'''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
	Can be used just as a normal neural network though, as well.
	'''

	def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
				 outermost_linear=False, nonlinearity='relu', weight_init=None):
		super().__init__()

		# nonlinearity = 'sine'

		self.first_layer_init = None

		# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
		# special first-layer initialization scheme
		nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
						 'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
						 'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
						 'tanh':(nn.Tanh(), init_weights_xavier, None),
						 'selu':(nn.SELU(inplace=True), init_weights_selu, None),
						 'softplus':(nn.Softplus(), init_weights_normal, None),
						 'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

		nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

		if weight_init is not None:  # Overwrite weight init if passed
			self.weight_init = weight_init
		else:
			self.weight_init = nl_weight_init

		self.net = []
		self.net.append(MetaSequential(
			BatchLinear(in_features, hidden_features), nl
		))

		for i in range(num_hidden_layers):
			self.net.append(MetaSequential(
				BatchLinear(hidden_features, hidden_features), nl
			))

		if outermost_linear:
			self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
		else:
			self.net.append(MetaSequential(
				BatchLinear(hidden_features, out_features), nl
			))

		self.net = MetaSequential(*self.net)
		if self.weight_init is not None:
			self.net.apply(self.weight_init)

		if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
			self.net[0].apply(first_layer_init)

	def forward(self, coords, params=None, **kwargs):
		if params is None:
			params = OrderedDict(self.named_parameters())

		# print('passing on with siren ', siren, get_subdict(params, 'net').keys())
		output = self.net(coords, params=get_subdict(params, 'net'))
		return output

	def forward_with_activations(self, coords, params=None, retain_grad=False):
		'''Returns not only model output, but also intermediate activations.'''
		if params is None:
			params = OrderedDict(self.named_parameters())

		activations = OrderedDict()

		x = coords.clone().detach().requires_grad_(True)
		activations['input'] = x
		for i, layer in enumerate(self.net):
			subdict = get_subdict(params, 'net.%d' % i)
			for j, sublayer in enumerate(layer):
				if isinstance(sublayer, BatchLinear):
					x = sublayer(x, params=get_subdict(subdict, '%d' % j))
				else:
					x = sublayer(x)

				if retain_grad:
					x.retain_grad()
				activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
		return activations

class PosEncodingNeRF(nn.Module):
	'''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
	def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
		super().__init__()

		self.in_features = in_features

		if self.in_features == 3:
			self.num_frequencies = 10
		elif self.in_features == 2:
			assert sidelength is not None
			if isinstance(sidelength, int):
				sidelength = (sidelength, sidelength)
			self.num_frequencies = 4
			if use_nyquist:
				self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
		elif self.in_features == 1:
			assert fn_samples is not None
			self.num_frequencies = 4
			if use_nyquist:
				self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

		self.out_dim = in_features + 2 * in_features * self.num_frequencies

	def get_num_frequencies_nyquist(self, samples):
		nyquist_rate = 1 / (2 * (2 * 1 / samples))
		return int(math.floor(math.log(nyquist_rate, 2)))

	def forward(self, coords):
		coords = coords.view(coords.shape[0], -1, self.in_features)

		coords_pos_enc = coords
		for i in range(self.num_frequencies):
			for j in range(self.in_features):
				c = coords[..., j]

				sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
				cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

				coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

		return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class CLIP(nn.Module):
	def __init__(self):
		super().__init__()

		dim = 512

		self.gru = nn.GRU(
			input_size=dim, 
			hidden_size=dim//2, 
			num_layers=3, 
			batch_first=True, 
			bidirectional=True, 
			dropout=0.1)

		self.model, self.preprocess = clip.load("ViT-B/32")

	def forward(self, video):
		video = video.permute(1, 0, 4, 2, 3)

		with torch.no_grad():
			video_features = [self.model.encode_image(x).unsqueeze(0) for x in video]

			video_features = torch.vstack(video_features).permute(1,0,2).float()

		video_features, _ = self.gru(video_features) # N x T x (D * Hout) (D = T)
		video_features = video_features.mean(1)

		return video_features

class SingleINR(MetaModule):
	'''A canonical representation network for a BVP.'''

	def __init__(self, num_hidden_layers, hidden_features, out_features=1, type='sine', in_features=2, **kwargs):
		super().__init__()

		self.positional_encoding = PosEncodingNeRF(in_features=in_features,
												   sidelength=kwargs.get('sidelength', None),
												   fn_samples=kwargs.get('fn_samples', None),
												   use_nyquist=kwargs.get('use_nyquist', True))
		in_features = self.positional_encoding.out_dim

		self.net = FCBlock(in_features=in_features, out_features=out_features, 
						num_hidden_layers=num_hidden_layers,
						hidden_features=hidden_features, outermost_linear=True, 
						nonlinearity=type)

	def forward(self, model_input, params=None):
		if params is None:
			params = OrderedDict(self.named_parameters())

		# Enables us to compute gradients w.r.t. coordinates
		coords_org = model_input['coords'].clone().detach().requires_grad_(True)
		coords = coords_org
		coords = self.positional_encoding(coords)

		output = self.net(coords, get_subdict(params, 'net'))

		return { 'model_in': coords_org, 'model_out': output }

class VideoGen(MetaModule):
	'''A Video Generation Network.'''
	def __init__(self, in_features=3, out_features=1, num_instances=1, mode='nerf', type='relu',
				hn_hidden_features=256, hn_hidden_layers=1, hn_in=512, hidden_features=256, 
				num_hidden_layers=3, std=0.01, useCLIP=False, **kwargs):
		super().__init__()

		self.mode = mode
		self.useCLIP = useCLIP
		self.num_instances = num_instances

		clip_dim = 512

		if self.useCLIP:
			self.clip = CLIP()

			self.mergeclipinstance = nn.Sequential(
				nn.Linear(clip_dim*2, clip_dim),
				nn.ReLU(True),
				nn.Linear(clip_dim, hn_in))

			self.latent_codes = nn.Embedding(self.num_instances, clip_dim)
			print('Using CLIP Embeddings!')
		else:
			self.latent_codes = nn.Embedding(self.num_instances, hn_in)
			
		# initializing the codebook from normal dsitribution
		nn.init.normal_(self.latent_codes.weight, mean=0, std=std)

		self.positional_encoding = PosEncodingNeRF(in_features=in_features,
						   sidelength=kwargs.get('sidelength', None),
						   fn_samples=kwargs.get('fn_samples', None),
						   use_nyquist=kwargs.get('use_nyquist', True))		
		
		in_features = self.positional_encoding.out_dim

		self.net = FCBlock(in_features=in_features, out_features=out_features, 
						num_hidden_layers=num_hidden_layers,
						hidden_features=hidden_features, outermost_linear=True, 
						nonlinearity=type)

		self.hyper_net = HyperNetwork(hyper_in_features=hn_in,
				  hyper_hidden_layers=hn_hidden_layers,
				  hyper_hidden_features=hn_hidden_features,
				  hypo_module=self.net)

	def init_from_learned_latents(self, latent_codes):
		with torch.no_grad():
			for i, _ in enumerate(latent_codes):
				self.latent_codes.weight[i] = latent_codes[i]

			print('Loaded latent codes from the previous checkpoint!')

	def run_hyper_net(self, z, coords):
		params = self.hyper_net(z)
		return self.net(coords, params)

	def forward(self, model_input):
		# Enables us to compute gradients w.r.t. coordinates
		coords = model_input['coords'].clone().detach().requires_grad_(True)
		coords = self.positional_encoding(coords)

		try:
			idx = model_input['idx']
		except:
			idx = torch.tensor([0], device=coords.device)

		if 'z' in model_input:
			z = model_input['z']
		elif self.useCLIP:
			z_clip = self.clip(model_input['clip_img'])

			z_normal = self.latent_codes(idx) # 1 x 256
			z_cat = torch.cat((z_clip, z_normal), 1)
			
			z = self.mergeclipinstance(z_cat)
		else:
			z = self.latent_codes(idx) # 1 x 256

		output = self.run_hyper_net(z, coords)

		return {
			'idx': idx, 
			'model_in': coords, 
			'model_out': output,
			'z': z
		}

########################
# HyperNetwork modules
class HyperNetwork(nn.Module):
	def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
		'''

		Args:
			hyper_in_features: In features of hypernetwork
			hyper_hidden_layers: Number of hidden layers in hypernetwork
			hyper_hidden_features: Number of hidden units in hypernetwork
			hypo_module: MetaModule. The module whose parameters are predicted.
		'''
		super().__init__()

		hypo_parameters = hypo_module.meta_named_parameters()

		self.names = []
		self.nets = nn.ModuleList()
		self.param_shapes = []
		for name, param in hypo_parameters:
			self.names.append(name)
			self.param_shapes.append(param.size())

			hn = FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
					   num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
					   outermost_linear=True)
			if 'weight' in name:
				hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
			elif 'bias' in name:
				hn.net[-1].apply(lambda m: hyper_bias_init(m))

			self.nets.append(hn)


	def forward(self, z):
		'''
		Args:-
			z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

		Returns:
			params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
		'''
		params = OrderedDict()
		for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
			batch_param_shape = (-1,) + param_shape
			params[name] = net(z).reshape(batch_param_shape)

			# print(f'name: {name}, param_shape: {param_shape}, params[name].shape: {params[name].shape}')
		
		return params

############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net, siren=False):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
		m.weight.data = m.weight.data / 1e1

	# if hasattr(m, 'bias') and siren:
	#     with torch.no_grad():
	#         m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m, siren=False):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
		m.weight.data = m.weight.data / 1.e1

	# if hasattr(m, 'bias') and siren:
	#     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
	#     with torch.no_grad():
	#         m.bias.uniform_(-1/fan_in, 1/fan_in)


########################
# Encoder modules
class SetEncoder(nn.Module):
	def __init__(self, in_features, out_features,
				 num_hidden_layers, hidden_features, nonlinearity='relu'):
		super().__init__()

		assert nonlinearity in ['relu', 'sine'], 'Unknown nonlinearity type'

		if nonlinearity == 'relu':
			nl = nn.ReLU(inplace=True)
			weight_init = init_weights_normal
		elif nonlinearity == 'sine':
			nl = Sine()
			weight_init = sine_init

		self.net = [nn.Linear(in_features, hidden_features), nl]
		self.net.extend([nn.Sequential(nn.Linear(hidden_features, hidden_features), nl)
						 for _ in range(num_hidden_layers)])
		self.net.extend([nn.Linear(hidden_features, out_features), nl])
		self.net = nn.Sequential(*self.net)

		self.net.apply(weight_init)

	def forward(self, context_x, context_y, ctxt_mask=None, **kwargs):
		input = torch.cat((context_x, context_y), dim=-1)
		embeddings = self.net(input)

		if ctxt_mask is not None:
			embeddings = embeddings * ctxt_mask
			embedding = embeddings.mean(dim=-2) * (embeddings.shape[-2] / torch.sum(ctxt_mask, dim=-2))
			return embedding
		return embeddings.mean(dim=-2)


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	# grab from upstream pytorch branch and paste here for now
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly fill tensor with values from [l, u], then translate to
		# [2l-1, 2u-1].
		tensor.uniform_(2 * l - 1, 2 * u - 1)

		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		tensor.erfinv_()

		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)

		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)
		return tensor


def init_weights_trunc_normal(m):
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			fan_in = m.weight.size(1)
			fan_out = m.weight.size(0)
			std = math.sqrt(2.0 / float(fan_in + fan_out))
			mean = 0.
			# initialize with the same behavior as tf.truncated_normal
			# "The generated values follow a normal distribution with specified mean and
			# standard deviation, except that values whose magnitude is more than 2
			# standard deviations from the mean are dropped and re-picked."
			_no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			nn.init.xavier_normal_(m.weight)


def sine_init(m):
	with torch.no_grad():
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			# See supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
	with torch.no_grad():
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
	y = x.clone()
	y[..., 1::2] = -1 * y[..., 1::2]
	return y


def compl_div(x, y):
	''' x / y '''
	a = x[..., ::2]
	b = x[..., 1::2]
	c = y[..., ::2]
	d = y[..., 1::2]

	outr = (a * c + b * d) / (c ** 2 + d ** 2)
	outi = (b * c - a * d) / (c ** 2 + d ** 2)
	out = torch.zeros_like(x)
	out[..., ::2] = outr
	out[..., 1::2] = outi
	return out


def compl_mul(x, y):
	'''  x * y '''
	a = x[..., ::2]
	b = x[..., 1::2]
	c = y[..., ::2]
	d = y[..., 1::2]

	outr = a * c - b * d
	outi = (a + b) * (c + d) - a * c - b * d
	out = torch.zeros_like(x)
	out[..., ::2] = outr
	out[..., 1::2] = outi
	return out
