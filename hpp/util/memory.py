from collections import deque
from random import Random
import numpy as np
import pickle
import os

from hpp.core import Transition


class ReplayBuffer:
    """
    Implementation of the replay experience buffer. Creates a buffer which uses
    the deque data structure. Here you can store experience transitions (i.e.: state,
    action, next state, reward) and sample mini-batches for training.
    You can  retrieve a transition like this:
    Example of use:
    .. code-block:: python
        replay_buffer = ReplayBuffer(10)
        replay_buffer.store()
        replay_buffer.store([0, 2, 1], [1, 2], -12.9, [2, 2, 1], 0)
        # ... more storing
        transition = replay_buffer(2)
    Parameters
    ----------
    buffer_size : int
        The buffer size
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.random = Random()
        self.count = 0

    def __call__(self, index):
        """
        Returns a transition from the buffer.

        Parameters
        ----------
        index : int
            The index number of the desired transition

        Returns
        -------
        tuple
            The transition
        """
        return self.buffer[index]

    def store(self, transition):
        """
        Stores a new transition on the buffer.

        Parameters
        ----------
        transition: Transition
            A transition of {state, action, next_state, reward, terminal}
        """
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample_batch(self, given_batch_size):
        """
        Samples a mini-batch from the buffer.

        Parameters
        ----------
        given_batch_size : int
            The size of the mini-batch.

        Returns
        -------
        numpy.array
            The state batch
        numpy.array
            The action batch
        numpy.array
            The reward batch
        numpy.array
            The next state batch
        numpy.array
            The terminal batch
        """

        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch = self.random.sample(self.buffer, batch_size)

        state_batch = np.array([_.state for _ in batch])
        action_batch = np.array([_.action for _ in batch])
        reward_batch = np.array([_.reward for _ in batch])
        next_state_batch = np.array([_.next_state for _ in batch])
        terminal_batch = np.array([_.terminal for _ in batch])

        return Transition(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

    def clear(self):
        """
        Clears the buffer my removing all elements.
        """
        self.buffer.clear()
        self.count = 0

    def size(self):
        """
        Returns the current size of the buffer.
        Returns
        -------
        int
            The number of existing transitions.
        """
        return self.count

    def seed(self, random_seed):
        self.random.seed(random_seed)

    def save(self, file_path):
        b = {'buffer': self.buffer, 'buffer_size': self.buffer_size, 'count': self.count}
        pickle.dump(b, open(file_path, 'wb'))

    @classmethod
    def load(cls, file_path):
        b = pickle.load(open(file_path, 'rb'))
        self = cls(b['buffer_size'])
        self.buffer = b['buffer']
        self.count = b['count']
        return self

    def remove(self, index):
        del self.buffer[index]
        self.count -= 1


class ReplayBufferDisk:
    def __init__(self, buffer_size, log_dir):
        self.buffer_size = buffer_size
        self.random = Random()
        self.count = 0
        self.log_dir = log_dir
        self.buffer_ids = []

        if not os.path.exists(os.path.join(log_dir, 'replay_buffer')):
            os.mkdir(os.path.join(log_dir, 'replay_buffer'))

        self.z_fill = 6

    def __call__(self, index):
        return self.buffer[index]

    def store(self, transition):
        pickle.dump(transition, open(os.path.join(self.log_dir, 'replay_buffer',
                                                  'transition_' + str(self.count).zfill(self.z_fill)), 'wb'))
        self.buffer_ids.append(self.count)
        if self.count < self.buffer_size:
            self.count += 1

    def sample_batch(self, given_batch_size):
        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch_ids = self.random.sample(self.buffer_ids, batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        for batch_id in batch_ids:
            transition = pickle.load(open(os.path.join(self.log_dir, 'replay_buffer',
                                          'transition_' + str(batch_id).zfill(self.z_fill)), 'rb'))
            state_batch.append(transition.state)
            action_batch.append(transition.action)
            reward_batch.append(transition.reward)
            next_state_batch.append(transition.next_state)
            terminal_batch.append(transition.terminal)

        transition = Transition(np.asarray(state_batch), np.asarray(action_batch), np.asarray(reward_batch),
                                np.asarray(next_state_batch), np.asarray(terminal_batch))
        return transition

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def size(self):
        return self.count

    def seed(self, random_seed):
        self.random.seed(random_seed)

    def state_dict(self):
        return {'buffer_size': self.buffer_size,
                'count': self.count,
                'random': self.random.getstate(),
                'log_dir': self.log_dir,
                'buffer_ids': self.buffer_ids,
                }

    def save(self, path):
        pickle.dump(self.state_dict(), open(path, 'wb'))

    @classmethod
    def load(cls, file_path):
        b = pickle.load(open(file_path, 'rb'))
        self = cls(b['buffer_size'], b['log_dir'])
        self.count = b['count']
        self.random.setstate(b['random'])
        self.buffer_ids = b['buffer_ids']
        return self

    def remove(self, index):
        del self.buffer[index]
        self.count -= 1


class TaskPrioritizedReplayBuffer(ReplayBufferDisk):
    def __init__(self, buffer_size, log_dir):
        super(TaskPrioritizedReplayBuffer, self).__init__(buffer_size, log_dir)
        self.predicted_value_log = []
        self.reward_value_log = []

    def store(self, transition, predicted_value_log):
        self.predicted_value_log.append(predicted_value_log)
        self.reward_value_log.append(transition.reward)
        super(TaskPrioritizedReplayBuffer, self).store(transition)

    def sample_proportional(self, sample_reward_value):
        sample_ind = np.argwhere(np.asarray(self.reward_value_log) == sample_reward_value)
        if len(sample_ind) > 0:
            sample_surprise_values = np.abs(np.asarray(self.predicted_value_log)[sample_ind[:, 0]] - np.asarray(self.reward_value_log)[sample_ind[:, 0]])
            sorted_surprise_ind = np.argsort(sample_surprise_values)
            sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
            pow_law_exp = 2
            rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
            sample_iteration = sorted_sample_ind[rand_sample_ind]
            print('Experience replay: iteration %d (surprise value: %f) reward %f' % (
            sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]], sample_reward_value))
            return sample_iteration
        else:
            return None

    def sample(self, sample_reward_value):
        sample_id = self.sample_proportional(sample_reward_value)
        if sample_id is None:
            return None, None
        else:
            transition = pickle.load(open(os.path.join(self.log_dir, 'replay_buffer',
                                                       'transition_' + str(sample_id).zfill(self.z_fill)), 'rb'))
            return transition, sample_id

    def update_priorities(self, idx, predicted_value_log):
        self.predicted_value_log[idx] = predicted_value_log

    def state_dict(self):
        state_dict = super(TaskPrioritizedReplayBuffer, self).state_dict()
        state_dict['predicted_value_log'] = self.predicted_value_log
        state_dict['reward_value_log'] = self.reward_value_log
        return state_dict
    
    @classmethod
    def load(cls, file_path):
        self = super(TaskPrioritizedReplayBuffer, cls).load(file_path)
        b = pickle.load(open(file_path, 'rb'))
        self.predicted_value_log = b['predicted_value_log']
        self.reward_value_log = b['reward_value_log']
        return self

