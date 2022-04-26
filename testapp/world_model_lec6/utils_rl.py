import time
import math
import torch
import torch.nn as nn
#import pdb
import os
#import collections
import pickle
import re
import numpy as np
#import sys
import glob
#import json
import gym
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

#------------------------------------------------------------
# utils.timer
#------------------------------------------------------------
class Timer:

	def __init__(self):
		self._start = time.time()

	def __call__(self, reset=True):
		now = time.time()
		diff = now - self._start
		if reset:
			self._start = now
		return diff


#------------------------------------------------------------
# models.ein
#------------------------------------------------------------
class EinLinear(nn.Module):

    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """
            input : [ B x n_models x input_dim ]
        """
        ## [ B x n_models x output_dim ]
        output = torch.einsum('eoi,bei->beo', self.weight, input)
        if self.bias is not None:
            raise RuntimeError()
        return output

    def extra_repr(self):
        return 'n_models={}, in_features={}, out_features={}, bias={}'.format(
            self.n_models, self.in_features, self.out_features, self.bias is not None
        )

'''
#------------------------------------------------------------
# utils.config
#------------------------------------------------------------
class Config(collections.Mapping):

    def __init__(self, _class, verbose=True, savepath=None, **kwargs):
        self._class = _class
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
            pickle.dump(self, open(savepath, 'wb'))
            print(f'Saved config to: {savepath}\n')

    def __repr__(self):
        string = f'\nConfig: {self._class}\n'
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f'    {key}: {val}\n'
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __call__(self):
        return self.make()

    def __getattr__(self, attr):
        if attr == '_dict' and '_dict' not in vars(self):
            self._dict = {}
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def make(self):
        if 'GPT' in str(self._class) or 'Trainer' in str(self._class) or 'TrajectoryTransformer' in str(self._class):
            ## GPT class expects the config as the sole input
            return self._class(self)
        else:
            return self._class(**self._dict)
'''

#------------------------------------------------------------
# utils.progress
#------------------------------------------------------------
class Progress:

	def __init__(self, total, name = 'Progress', ncol=3, max_length=30, indent=8, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()


	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		print('\n', end='')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
					(key, val)
					for key, val in params.items()
				])

		############
		# Position #
		############
		self._clear()

		###########
		# Percent #
		###########
		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		#########
		# Speed #
		#########
		speed = self._format_speed(self._step)

		##########
		# Params #
		##########
		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines


		description = '{} | {}{}'.format(percent, speed, params_string)
		print(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		print(position, end='')
		print(empty)
		print(position, end='')

	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent*100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i+n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0,'')
		padding = '\n' + ' '*self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param, str_length=8):
		k, v = param
		k = k.rjust(str_length)
		if type(v) == float or hasattr(v, 'item'):
			string = '{}: {:12.4f}'
		else:
			string = '{}: {}'
			v = str(v).rjust(12)
		return string.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			string = re.sub(r'\s+', ' ', string)
			self._clear()
			print(string, end='\n')
			# sys.stdout.write(string)
			# sys.stdout.flush()
			self._skip_lines = 1 # 0
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()

class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None


#------------------------------------------------------------
# search.sampling
#------------------------------------------------------------

#-------------------------------- helper functions --------------------------------#

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def filter_cdf(logits, threshold):
    batch_inds = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    bins_inds = torch.arange(logits.shape[-1], device=logits.device)
    probs = logits.softmax(dim=-1)
    probs_sorted, _ = torch.sort(probs, dim=-1)
    probs_cum = torch.cumsum(probs_sorted, dim=-1)
    ## get minimum probability p such that the cdf up to p is at least `threshold`
    mask = probs_cum < threshold
    masked_inds = torch.argmax(mask * bins_inds, dim=-1)
    probs_threshold = probs_sorted[batch_inds, masked_inds]
    ## filter
    out = logits.clone()
    logits_mask = probs <= probs_threshold.unsqueeze(dim=-1)
    out[logits_mask] = -1000
    return out

def round_to_multiple(x, N):
    '''
        Rounds `x` up to nearest multiple of `N`.

        x : int
        N : int
    '''
    pad = (N - x % N) % N
    return x + pad

def sort_2d(x):
    '''
        x : [ M x N ]
    '''
    M, N = x.shape
    x = x.view(-1)
    x_sort, inds = torch.sort(x, descending=True)

    rows = inds // N
    cols = inds % N

    return x_sort, rows, cols

#-------------------------------- forward pass --------------------------------#
def crop_x(x, model, max_block=None, allow_crop=True, crop_increment=None, **kwargs):
    '''
        Crops the input if the sequence is too long.

        x : tensor[ batch_size x sequence_length ]
    '''
    block_size = min(model.get_block_size(), max_block or np.inf)

    if x.shape[1] > block_size:
        assert allow_crop, (
            f'[ search/sampling ] input size is {x.shape} and block size is {block_size}, '
            'but cropping not allowed')

        ## crop out entire transition at a time so that the first token is always s_t^0
        n_crop = round_to_multiple(x.shape[1] - block_size, crop_increment)
        assert n_crop % crop_increment == 0
        x = x[:, n_crop:]

    return x


def forward(model, x, max_block=None, allow_crop=True, crop_increment=None, **kwargs):
    '''
        A wrapper around a single forward pass of the transformer.
        Crops the input if the sequence is too long.

        x : tensor[ batch_size x sequence_length ]
    '''
    model.eval()

    block_size = min(model.get_block_size(), max_block or np.inf)

    if x.shape[1] > block_size:
        assert allow_crop, (
            f'[ search/sampling ] input size is {x.shape} and block size is {block_size}, '
            'but cropping not allowed')

        ## crop out entire transition at a time so that the first token is always s_t^0
        n_crop = round_to_multiple(x.shape[1] - block_size, crop_increment)
        assert n_crop % crop_increment == 0
        x = x[:, n_crop:]

    logits, _ = model(x, **kwargs)
    print("forward/logits:",logits)
    print("forword/_:",_)

    return logits

def get_logp(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    '''
        x : tensor[ batch_size x sequence_length ]
    '''
    ## [ batch_size x sequence_length x vocab_size ]
    logits = forward(model, x, **forward_kwargs)

    ## pluck the logits at the final step and scale by temperature
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## optionally crop logits to only the top `1 - cdf` percentile
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## optionally crop logits to only the most likely `k` options
    if topk is not None:
        logits = top_k_logits(logits, topk)

    ## apply softmax to convert to probabilities
    logp = logits.log_softmax(dim=-1)

    return logp

#-------------------------------- sampling --------------------------------#

def sample(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    '''
        Samples from the distribution parameterized by `model(x)`.

        x : tensor[ batch_size x sequence_length ]
    '''
    ## [ batch_size x sequence_length x vocab_size ]
    logits = forward(model, x, **forward_kwargs)

    ## pluck the logits at the final step and scale by temperature
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## keep track of probabilities before modifying logits
    raw_probs = logits.softmax(dim=-1)

    ## optionally crop logits to only the top `1 - cdf` percentile
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## optionally crop logits to only the most likely `k` options
    if topk is not None:
        logits = top_k_logits(logits, topk)

    ## apply softmax to convert to probabilities
    probs = logits.softmax(dim=-1)

    ## sample from the distribution
    ## [ batch_size x 1 ]
    indices = torch.multinomial(probs, num_samples=1)

    return indices, raw_probs

@torch.no_grad()
def sample_n(model, x, N, **sample_kwargs):
    batch_size = len(x)

    ## keep track of probabilities from each step;
    ## `vocab_size + 1` accounts for termination token
    probs = torch.zeros(batch_size, N, model.vocab_size + 1, device=x.device)

    for n in range(N):
        indices, p = sample(model, x, **sample_kwargs)

        ## append to the sequence and continue
        ## [ batch_size x (sequence_length + n) ]
        x = torch.cat((x, indices), dim=1)

        probs[:, n] = p

    return x, probs


#------------------------------------------------------------
# utils.serialization
#------------------------------------------------------------
def get_latest_epoch(loadpath):
    states = glob.glob1(loadpath, 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_model(*loadpath, epoch=None, device='cuda:0'):
    loadpath = os.path.join(*loadpath)
    config_path = os.path.join(loadpath, 'model_config.pkl')

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'[ utils/serialization ] Loading model epoch: {epoch}')
    state_path = os.path.join(loadpath, f'state_{epoch}.pt')

    config = pickle.load(open(config_path, 'rb'))
    state = torch.load(state_path)

    model = config()
    model.to(device)
    model.load_state_dict(state, strict=True)

    print(f'\n[ utils/serialization ] Loaded config from {config_path}\n')
    print(config)

    return model, epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_from_config(*loadpath):
    config = load_config(*loadpath)
    return config.make()


#------------------------------------------------------------
# utils.arrays
#------------------------------------------------------------
DTYPE = torch.float
DEVICE = 'cuda:0'

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	return torch.tensor(x, dtype=dtype, device=device)


#------------------------------------------------------------
# utils.discretization
#------------------------------------------------------------
class discretization:

    class QuantileDiscretizer:

    	def __init__(self, data, N):
    		self.data = data
    		self.N = N

    		n_points_per_bin = int(np.ceil(len(data) / N))
    		obs_sorted = np.sort(data, axis=0)
    		thresholds = obs_sorted[::n_points_per_bin, :]
    		maxs = data.max(axis=0, keepdims=True)

    		## [ (N + 1) x dim ]
    		self.thresholds = np.concatenate([thresholds, maxs], axis=0)

    		# threshold_inds = np.linspace(0, len(data) - 1, N + 1, dtype=int)
    		# obs_sorted = np.sort(data, axis=0)

    		# ## [ (N + 1) x dim ]
    		# self.thresholds = obs_sorted[threshold_inds, :]

    		## [ N x dim ]
    		self.diffs = self.thresholds[1:] - self.thresholds[:-1]

    		## for sparse reward tasks
    		# if (self.diffs[:,-1] == 0).any():
    		# 	raise RuntimeError('rebin for sparse reward tasks')

    		self._test()

    	def __call__(self, x):
    		indices = self.discretize(x)
    		recon = self.reconstruct(indices)
    		error = np.abs(recon - x).max(0)
    		return indices, recon, error

    	def _test(self):
    		print('[ utils/discretization ] Testing...', end=' ', flush=True)
    		inds = np.random.randint(0, len(self.data), size=1000)
    		X = self.data[inds]
    		indices = self.discretize(X)
    		recon = self.reconstruct(indices)
    		## make sure reconstruction error is less than the max allowed per dimension
    		error = np.abs(X - recon).max(0)
    		assert (error <= self.diffs.max(axis=0)).all()
    		## re-discretize reconstruction and make sure it is the same as original indices
    		indices_2 = self.discretize(recon)
    		assert (indices == indices_2).all()
    		## reconstruct random indices
    		## @TODO: remove duplicate thresholds
    		# randint = np.random.randint(0, self.N, indices.shape)
    		# randint_2 = self.discretize(self.reconstruct(randint))
    		# assert (randint == randint_2).all()
    		print('done.')

    	def largest_nonzero_index(self, x, dim):
    	    N = x.shape[dim]
    	    arange = np.arange(N) + 1

    	    for i in range(dim):
    	        arange = np.expand_dims(arange, axis=0)
    	    for i in range(dim+1, x.ndim):
    	        arange = np.expand_dims(arange, axis=-1)

    	    inds = np.argmax(x * arange, axis=0)
    	    ## masks for all `False` or all `True`
    	    lt_mask = (~x).all(axis=0)
    	    gt_mask = (x).all(axis=0)

    	    inds[lt_mask] = 0
    	    inds[gt_mask] = N

    	    return inds

    	def discretize(self, x, subslice=(None, None)):
    	    '''x : [ B x observation_dim ]'''

    	    if torch.is_tensor(x):
    		    x = to_np(x)

    	    ## enforce batch mode
    	    if x.ndim == 1:
    		    x = x[None]

    	    ## [ N x B x observation_dim ]
    	    start, end = subslice
    	    thresholds = self.thresholds[:, start:end]

    	    gt = x[None] >= thresholds[:,None]
    	    indices = self.largest_nonzero_index(gt, dim=0)

    	    if indices.min() < 0 or indices.max() >= self.N:
    	        indices = np.clip(indices, 0, self.N - 1)

    	    return indices

    	def reconstruct(self, indices, subslice=(None, None)):

    		if torch.is_tensor(indices):
    			indices = to_np(indices)

    		## enforce batch mode
    		if indices.ndim == 1:
    			indices = indices[None]

    		if indices.min() < 0 or indices.max() >= self.N:
    			print(f'[ utils/discretization ] indices out of range: ({indices.min()}, {indices.max()}) | N: {self.N}')
    			indices = np.clip(indices, 0, self.N - 1)

    		start, end = subslice
    		thresholds = self.thresholds[:, start:end]

    		left = np.take_along_axis(thresholds, indices, axis=0)
    		right = np.take_along_axis(thresholds, indices + 1, axis=0)
    		recon = (left + right) / 2.
    		return recon

    	#---------------------------- wrappers for planning ----------------------------#

    	def expectation(self, probs, subslice):
    		'''
    			probs : [ B x N ]
    		'''

    		if torch.is_tensor(probs):
    			probs = to_np(probs)

    		## [ N ]
    		thresholds = self.thresholds[:, subslice]
    		## [ B ]
    		left  = probs @ thresholds[:-1]
    		right = probs @ thresholds[1:]

    		avg = (left + right) / 2.
    		return avg

    	def percentile(self, probs, percentile, subslice):
    		'''
    			percentile `p` :
    				returns least value `v` s.t. cdf up to `v` is >= `p`
    				e.g., p=0.8 and v=100 indicates that
    					  100 is in the 80% percentile of values
    		'''
    		## [ N ]
    		thresholds = self.thresholds[:, subslice]
    		## [ B x N ]
    		cumulative = np.cumsum(probs, axis=-1)
    		valid = cumulative > percentile
    		## [ B ]
    		inds = np.argmax(np.arange(self.N, 0, -1) * valid, axis=-1)
    		left = thresholds[inds-1]
    		right = thresholds[inds]
    		avg = (left + right) / 2.
    		return avg

    	#---------------------------- wrappers for planning ----------------------------#

    	def value_expectation(self, probs):
    		'''
    			probs : [ B x 2 x ( N + 1 ) ]
    				extra token comes from termination
    		'''

    		if torch.is_tensor(probs):
    			probs = to_np(probs)
    			return_torch = True
    		else:
    			return_torch = False

    		probs = probs[:, :, :-1]
    		assert probs.shape[-1] == self.N

    		rewards = self.expectation(probs[:, 0], subslice=-2)
    		next_values = self.expectation(probs[:, 1], subslice=-1)

    		if return_torch:
    			rewards = to_torch(rewards)
    			next_values = to_torch(next_values)

    		return rewards, next_values

    	def value_fn(self, probs, percentile):
    		if percentile == 'mean':
    			return self.value_expectation(probs)
    		else:
    			## percentile should be interpretable as float,
    			## even if passed in as str because of command-line parser
    			percentile = float(percentile)

    		if torch.is_tensor(probs):
    			probs = to_np(probs)
    			return_torch = True
    		else:
    			return_torch = False

    		probs = probs[:, :, :-1]
    		assert probs.shape[-1] == self.N

    		rewards = self.percentile(probs[:, 0], percentile, subslice=-2)
    		next_values = self.percentile(probs[:, 1], percentile, subslice=-1)

    		if return_torch:
    			rewards = to_torch(rewards)
    			next_values = to_torch(next_values)

    		return rewards, next_values



#------------------------------------------------------------
# datasets.d4rl
#------------------------------------------------------------
@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

#with suppress_output():
    ## d4rl prints out a variety of warnings
#    import d4rl_pybullet # import d4rl

# def construct_dataloader(dataset, **kwargs):
#     dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, pin_memory=True, **kwargs)
#     return dataloader

def qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []

    episode_step = 0
    maximum_episode_step = 1000
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        # final_timestep = dataset['timeouts'][i]
        final_timestep = (episode_step == maximum_episode_step-1)

        if i < N - 1:
            # done_bool += dataset['timeouts'][i] #+1]
            done_bool += (episode_step == maximum_episode_step-1)

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:,None],
        'terminals': np.array(done_)[:,None],
        'realterminals': np.array(realdone_)[:,None],
    }


def load_environment(name):
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


#------------------------------------------------------------
# datasets.sequence
#------------------------------------------------------------
def segment(observations, terminals, max_path_length):
    """
        segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        if term.squeeze():
            trajectories.append([])

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros((n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype)
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=np.bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i,:path_length] = traj
        early_termination[i,path_length:] = 1

    return trajectories_pad, early_termination, path_lengths

'''
class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env, sequence_length=250, step=10, discount=0.99, max_path_length=1000, penalty=None, device='cuda:0'):
        print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.env = env = load_environment(env) if type(env) is str else env
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        self.device = device

        print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
        dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=True)
        print('done.')

        """
        preprocess_fn = dataset_preprocess_functions.get(env.name)
        if preprocess_fn:
            print(f'[ datasets/sequence ] Modifying environment')
            dataset = preprocess_fn(dataset)
        """

        observations = dataset['observations']
        actions = dataset['actions']
        next_observations = dataset['next_observations']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        realterminals = dataset['realterminals']

        self.observations_raw = observations
        self.actions_raw = actions
        self.next_observations_raw = next_observations
        self.joined_raw = np.concatenate([observations, actions], axis=-1)
        self.rewards_raw = rewards
        self.terminals_raw = terminals

        ## terminal penalty
        if penalty is not None:
            terminal_mask = realterminals.squeeze()
            self.rewards_raw[terminal_mask] = penalty

        ## segment
        print(f'[ datasets/sequence ] Segmenting...', end=' ', flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(self.joined_raw, terminals, max_path_length)
        self.rewards_segmented, *_ = segment(self.rewards_raw, terminals, max_path_length)
        print('done.')

        self.discount = discount
        self.discounts = (discount ** np.arange(self.max_path_length))[:,None]

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:,t+1:] * self.discounts[:-t-1]).sum(axis=1)
            self.values_segmented[:,t] = V

        ## add (r, V) to `joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]
        self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
        self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)

        ## get valid indices
        indices = []
        for path_ind, length in enumerate(self.path_lengths):
            end = length - 1
            for i in range(end):
                indices.append((path_ind, i, i+sequence_length))

        self.indices = np.array(indices)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.joined_segmented = np.concatenate([
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length-1, joined_dim)),
        ], axis=1)
        self.termination_flags = np.concatenate([
            self.termination_flags,
            np.ones((n_trajectories, sequence_length-1), dtype=np.bool),
        ], axis=1)

    def __len__(self):
        return len(self.indices)


class DiscretizedDataset(SequenceDataset):

    def __init__(self, *args, N=50, discretizer='QuantileDiscretizer', **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        discretizer_class = getattr(discretization, discretizer)
        self.discretizer = discretizer_class(self.joined_raw, N)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step]

        joined_discrete = self.discretizer.discretize(joined)

        ## replace with termination token if the sequence has ended
        assert (joined[terminations] == 0).all(), \
                f'Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}'
        joined_discrete[terminations] = self.N

        ## [ (sequence_length / skip) x observation_dim]
        joined_discrete = to_torch(joined_discrete, device='cpu', dtype=torch.long).contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0

        ## flatten everything
        joined_discrete = joined_discrete.view(-1)
        mask = mask.view(-1)

        X = joined_discrete[:-1]
        Y = joined_discrete[1:]
        mask = mask[:-1]

        return X, Y, mask
'''

#------------------------------------------------------------
# search.utils
#------------------------------------------------------------
def make_prefix(discretizer, context, obs, prefix_context=True):
    observation_dim = obs.size
    obs_discrete = discretizer.discretize(obs, subslice=[0, observation_dim])
    obs_discrete = to_torch(obs_discrete, dtype=torch.long)

    if prefix_context:
        prefix = torch.cat(context + [obs_discrete], dim=-1)
    else:
        prefix = obs_discrete

    return prefix

def extract_actions(x, observation_dim, action_dim, t=None):
    assert x.shape[1] == observation_dim + action_dim + 2
    actions = x[:, observation_dim:observation_dim+action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions

VALUE_PLACEHOLDER = 1e6

def update_context(context, discretizer, observation, action, reward, max_context_transitions):
    '''
        context : list of transitions
            [ tensor( transition_dim ), ... ]
    '''
    ## use a placeholder for value because input values are masked out by model
    rew_val = np.array([reward, VALUE_PLACEHOLDER])
    transition = np.concatenate([observation, action, rew_val])

    ## discretize transition and convert to torch tensor
    transition_discrete = discretizer.discretize(transition)
    transition_discrete = to_torch(transition_discrete, dtype=torch.long)

    ## add new transition to context
    context.append(transition_discrete)

    ## crop context if necessary
    context = context[-max_context_transitions:]

    return context
