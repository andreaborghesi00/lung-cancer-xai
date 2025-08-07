import torch
from torch.utils.data import Sampler
import numpy as np

class CurriculumSampler(Sampler):
    def __init__(self, indices, total_epochs, current_epoch, min_prob=0.1):
        self.indices = indices
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        self.min_prob = min_prob  # even hard samples get sampled sometimes

        # Linearly increase preference for harder samples
        self.weights = self._compute_weights()

    def _compute_weights(self):
        n = len(self.indices)
        progress = self.current_epoch / self.total_epochs
        # Weight = mix between easy-to-hard (sorted)
        linear = np.linspace(1, 0, n)
        weights = self.min_prob + (1 - self.min_prob) * ((1 - progress) * (1 - linear) + progress * linear)
        return torch.DoubleTensor(weights)

    def __iter__(self):
        sampled = torch.multinomial(self.weights, num_samples=len(self.indices), replacement=False)
        return iter([self.indices[i] for i in sampled])

    def __len__(self):
        return len(self.indices)

class CurriculumBalancedSampler(Sampler):
    def __init__(self, labels, difficulties, total_epochs, samples_per_epoch):
        self.labels = np.array(labels)
        self.difficulties = np.array(difficulties) # we don't really need to sort the df by difficulty, we just need the values, we'll filter them later
        self.total_epochs = total_epochs
        self.samples_per_epoch = samples_per_epoch

        self.benign_idx = np.where(self.labels == 0)[0]
        self.malignant_idx = np.where(self.labels == 1)[0]
        
        self.progress = 0.1

        # Normalize difficulties separately per class
        self.benign_diff = self._normalize(self.difficulties[self.benign_idx])
        self.malignant_diff = self._normalize(self.difficulties[self.malignant_idx])

    def _normalize(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def set_epoch(self, epoch):
        self.progress = (epoch) / (self.total_epochs) # we'll use this as a moving threshold for the allowed difficulty range. the +1

    def _get_class_sample(self, class_indices, class_diff, n):
        # mask = class_diff <= self.progress # allow only samples with difficulty <= progress (we can do this as we assume to be both between 0 and 1).
        mask = class_diff <= 0.35 + self.progress
        # mask = class_diff <= 9999
        if mask.sum() == 0:
            mask = class_diff <= self.progress  # avoid empty
        eligible = class_indices[mask]
        weights = np.ones(len(eligible))
        return np.random.choice(eligible, size=n, replace=True, p=weights / weights.sum())
    
    def __iter__(self):
        half = self.samples_per_epoch // 2
        benign_sample = self._get_class_sample(self.benign_idx, self.benign_diff, half)
        malignant_sample = self._get_class_sample(self.malignant_idx, self.malignant_diff, half)
        all_indices = np.concatenate([benign_sample, malignant_sample])
        np.random.shuffle(all_indices)
        return iter(all_indices.tolist())

    def __len__(self):
        return self.samples_per_epoch
    
class CurriculumSampler(Sampler):
    def __init__(self, labels, difficulties, total_epochs, samples_per_epoch):
        self.difficulties = np.array(difficulties) # we don't really need to sort the df by difficulty, we just need the values, we'll filter them later
        self.total_epochs = total_epochs
        self.samples_per_epoch = samples_per_epoch

        self.progress = 0.0

        # Normalize difficulties separately per class
        self.diff = self._normalize(self.difficulties)

    def _normalize(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def set_epoch(self, epoch):
        self.progress = (epoch) / (self.total_epochs) # we'll use this as a moving threshold for the allowed difficulty range. the +1

    def _get_sample(self, class_diff, n):
        # mask = class_diff <= self.progress # allow only samples with difficulty <= progress (we can do this as we assume to be both between 0 and 1).
        mask = class_diff <= 0.4 + self.progress
        # mask = class_diff <= 9999
        if mask.sum() == 0:
            mask = class_diff <= self.progress  # avoid empty
        # one-class eligibility
        eligible = np.where(mask)[0]        
        weights = np.ones(len(eligible))
        return np.random.choice(eligible, size=n, replace=True, p=weights / weights.sum())
    
    def __iter__(self):
        sample = self._get_sample(self.diff, self.samples_per_epoch)
        return iter(sample.tolist())

    def __len__(self):
        return self.samples_per_epoch
    


