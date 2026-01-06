import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
from utils.logger import *

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from typing import List, Iterator, Tuple, Optional 
import math 


@DATASETS.register_module()
class AsEP(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset 
        if self.subset != 'test':
            self.logger = get_logger(config.log_name)
        else:
            self.logger = None
        print_log(f'[DATASET] Reading data from: {self.pc_path}', logger = self.logger)
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        self.use_PLMs = config.get('use_PLMs', False) 
        self.only_PLMs = config.get('only_PLMs', False) 
        self.data_augmentation = config.get('data_augmentation', True) 

        if not os.path.exists(self.data_list_file):
             raise FileNotFoundError(f"Data list file not found: {self.data_list_file}")
        val_data_list_file = os.path.join(self.data_root, 'val.txt')

        self.whole = config.get('whole', False)
        print_log(f'[DATASET] Reading data list from: {self.data_list_file}', logger = self.logger)
        
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        print_log(f'[DATASET] data samples: {lines[:5]}', logger = self.logger)
        if self.whole:
            if os.path.exists(val_data_list_file):
                print_log(f'[DATASET] Reading additional data list from: {val_data_list_file}', logger = self.logger)
                with open(val_data_list_file, 'r') as f:
                    val_lines = f.readlines()
                lines.extend(val_lines) 
            else:
                 print_log(f'[DATASET] Warning: val.txt not found for whole=True mode.', logger=self.logger)

        self.file_list = [line.strip() for line in lines if line.strip()]
        if not self.file_list:
             raise ValueError("No valid samples found in the data list file(s).")
        print_log(f'[DATASET] {len(self.file_list)} instances loaded for subset: {self.subset}', logger = self.logger)
        self._point_counts = None 
        self.augmentation_rng = np.random.RandomState(123)  

    def standardize_PSSM(self, PSSM):
        mean = np.mean(PSSM, axis=1, keepdims=True)
        std = np.std(PSSM, axis=1, keepdims=True) + 1e-8
        return (PSSM - mean) / std

    def normalize_ASA(self, ASA, global_max_ASA):
        return ASA / global_max_ASA

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        if pc.shape[0] == 0: 
             return pc
        centroid = np.mean(pc, axis=0) 
        pc = pc - centroid 
        m = np.max(np.sqrt(np.sum(pc**2, axis=1))) 
        if m < 1e-6:
             return pc
        pc = pc / m 
        return pc
    
    def normalize_embedding(self, embedding):
        mean = embedding.mean(axis=0, keepdims=True)
        std = embedding.std(axis=0, keepdims=True) + 1e-6
        return (embedding - mean) / std
    def add_gaussian_noise(self, tensor, std, clamp=None, renormalize=False, eps=1e-6):
        """
        tensor: input tensor [*, D]
            std: standard deviation of the noise
            clamp: (min, max) for clipping after noise
            renormalize: whether to normalize the tensor along the last dimension
        """
        noise = np.random.normal(0.0, std, tensor.shape).astype(tensor.dtype)
        noisy = tensor + noise

        if clamp is not None:
            noisy = np.clip(noisy, clamp[0], clamp[1])
        
        if renormalize:
            norm = np.linalg.norm(noisy, axis=-1, keepdims=True) + eps
            noisy = noisy / norm
    
        return noisy

    def random_rotation_matrix(self):
        """Generate a random 3D rotation matrix"""
        # Use separate random state for augmentation
        angles = self.augmentation_rng.uniform(0, 2*np.pi, 3)
        
        # Rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1]])
        
        # Combined rotation matrix
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R.astype(np.float32)

    def apply_random_rotation(self, xyz, normals=None):
        """Apply random rotation to coordinates and optionally to normals"""

        R = self.random_rotation_matrix()
        xyz_rotated = np.dot(xyz, R.T)
        if normals is not None:
            normals_rotated = np.dot(normals, R.T)
            return xyz_rotated, normals_rotated,R
        
        return xyz_rotated,None,R
        
    def __getitem__(self, idx):
        sample_filename = self.file_list[idx]
        file_path = os.path.join(self.pc_path, sample_filename + '.npz')
        data = np.load(file_path)

        xyz_A = data['xyz_A'].astype(np.float32) 
        atom_types_one_hot_A = data['atom_types_one_hot_A'].astype(np.float32) 
        atom_normals_A = data['atom_normals_A'].astype(np.float32)  

        if self.subset != 'test' and self.data_augmentation: 
            xyz_A = self.pc_norm(xyz_A) 
            xyz_A = self.add_gaussian_noise(xyz_A, std=0.029)            
            atom_normals_A = self.add_gaussian_noise(atom_normals_A, std=0.036, renormalize=True)

        if self.subset != 'test':
            xyz_A_rotated, atom_normals_A_rotated, _ = self.apply_random_rotation(xyz_A, atom_normals_A)
            atom_features_A = np.concatenate([
                atom_types_one_hot_A,
                atom_normals_A_rotated,
            ], axis=1)
        else: 
            atom_features_A = np.concatenate([
                atom_types_one_hot_A,
                atom_normals_A, 
            ], axis=1)
            xyz_A_rotated = xyz_A
        
        residue_starts_A = data['residue_starts_A'].astype(np.int64)
        one_hot_aa_A = data['one_hot_aa_A'].astype(np.float32)
        neigh_freqs_A = data['neigh_freqs_A'].astype(np.float32)
        ASA_A = data['ASA_A'].astype(np.float32)
        RSA_A = data['RSA_A'].astype(np.float32)
        pssm_A = data['pssm_A'].astype(np.float32)
        esm2_A = data['esm2_A'].astype(np.float32) 

        global_max_ASA = np.max(np.concatenate([data['ASA_A'], data['ASA_H'], data['ASA_L']])) + 1e-8
        if not self.only_PLMs:
            if not self.use_PLMs:
                aa_features_A = np.concatenate([one_hot_aa_A, neigh_freqs_A, self.normalize_ASA(ASA_A, global_max_ASA).reshape(-1, 1), RSA_A.reshape(-1, 1), self.standardize_PSSM(pssm_A)], axis=1) # 20+20+1+1+20=62
            else:
                aa_features_A = np.concatenate([one_hot_aa_A, neigh_freqs_A, self.normalize_ASA(ASA_A, global_max_ASA).reshape(-1, 1), RSA_A.reshape(-1, 1), self.standardize_PSSM(pssm_A) , self.normalize_embedding(esm2_A)], axis=1)
        else:
            aa_features_A =  self.normalize_embedding(esm2_A)

        xyz_H = data['xyz_H'].astype(np.float32)
        atom_types_one_hot_H = data['atom_types_one_hot_H'].astype(np.float32)
        atom_normals_H = data['atom_normals_H'].astype(np.float32)
        xyz_L = data['xyz_L'].astype(np.float32)
        atom_types_one_hot_L = data['atom_types_one_hot_L'].astype(np.float32)
        atom_normals_L = data['atom_normals_L'].astype(np.float32)

        xyz_antibody = np.concatenate([xyz_H, xyz_L], axis=0) 
        atom_normals_antibody = np.concatenate([atom_normals_H, atom_normals_L], axis=0)

        if self.subset != 'test' and self.data_augmentation:
            xyz_antibody = self.pc_norm(xyz_antibody)
            xyz_antibody = self.add_gaussian_noise(xyz_antibody, std=0.029)
            atom_normals_antibody = self.add_gaussian_noise(atom_normals_antibody, std=0.036, renormalize=True)

        if self.subset != 'test':
            xyz_antibody_rotated, atom_normals_antibody_rotated, rotation_matrix = self.apply_random_rotation(xyz_antibody, atom_normals_antibody) 

            atom_features_H = np.concatenate([
                atom_types_one_hot_H, 
                atom_normals_antibody_rotated[:len(xyz_H)],  
            ], axis=1)
            atom_features_L = np.concatenate([
                atom_types_one_hot_L, 
                atom_normals_antibody_rotated[len(xyz_H):], 
            ], axis=1)
        else:
            atom_features_H = np.concatenate([
                atom_types_one_hot_H, 
                atom_normals_antibody[:len(xyz_H)]
            ], axis=1)
            atom_features_L = np.concatenate([
                atom_types_one_hot_L, 
                atom_normals_antibody[len(xyz_H):]
            ], axis=1)
            xyz_antibody_rotated = xyz_antibody

        atom_features_antibody = np.concatenate([atom_features_H, atom_features_L], axis=0)


        residue_starts_H = data['residue_starts_H'].astype(np.int64)
        one_hot_aa_H = data['one_hot_aa_H'].astype(np.float32)
        neigh_freqs_H = data['neigh_freqs_H'].astype(np.float32)
        ASA_H = data['ASA_H'].astype(np.float32)
        RSA_H = data['RSA_H'].astype(np.float32)
        pssm_H = data['pssm_H'].astype(np.float32)
        antiberty_H = data['antiberty_H'].astype(np.float32) 
        if not self.only_PLMs:
            if not self.use_PLMs:
                aa_features_H = np.concatenate([one_hot_aa_H, neigh_freqs_H,  self.normalize_ASA(ASA_H, global_max_ASA).reshape(-1, 1), RSA_H.reshape(-1, 1), self.standardize_PSSM(pssm_H)], axis=1)
            else:
                aa_features_H = np.concatenate([one_hot_aa_H, neigh_freqs_H, self.normalize_ASA(ASA_H, global_max_ASA).reshape(-1, 1), RSA_H.reshape(-1, 1), self.standardize_PSSM(pssm_H), self.normalize_embedding(antiberty_H)], axis=1) # 20+20+1+1+20+128=62
        else:
            aa_features_H = self.normalize_embedding(antiberty_H)
        
        residue_starts_L = data['residue_starts_L'].astype(np.int64)
        one_hot_aa_L = data['one_hot_aa_L'].astype(np.float32)
        neigh_freqs_L = data['neigh_freqs_L'].astype(np.float32)
        ASA_L = data['ASA_L'].astype(np.float32)
        RSA_L = data['RSA_L'].astype(np.float32)
        pssm_L = data['pssm_L'].astype(np.float32)
        antiberty_L = data['antiberty_L'].astype(np.float32)  
        
        if not self.only_PLMs:
            if not self.use_PLMs:
                aa_features_L = np.concatenate([one_hot_aa_L, neigh_freqs_L,self.normalize_ASA(ASA_L, global_max_ASA).reshape(-1, 1), RSA_L.reshape(-1, 1), self.standardize_PSSM(pssm_L)], axis=1)
            else:
                aa_features_L = np.concatenate([one_hot_aa_L, neigh_freqs_L,self.normalize_ASA(ASA_L, global_max_ASA).reshape(-1, 1), RSA_L.reshape(-1, 1), self.standardize_PSSM(pssm_L),self.normalize_embedding(antiberty_L)], axis=1)
        else:
            aa_features_L = self.normalize_embedding(antiberty_L)

        aa_features_antibody = np.concatenate([aa_features_H, aa_features_L], axis=0)
        
        epitope = data['epitope'].astype(np.float32) 

        residue_starts_antibody = np.concatenate([
            residue_starts_H,
            residue_starts_L + len(xyz_H)
        ])


        xyz_A_normalized = self.pc_norm(xyz_A_rotated)
        xyz_antibody_normalized = self.pc_norm(xyz_antibody_rotated)

        if self.subset == 'test':
            return (
                torch.from_numpy(xyz_A_normalized).float(),
                torch.from_numpy(atom_features_A).float(),
                torch.from_numpy(xyz_antibody_normalized).float(),
                torch.from_numpy(atom_features_antibody).float(),
                torch.from_numpy(residue_starts_A).long(),
                torch.from_numpy(residue_starts_antibody).long(),
                torch.from_numpy(epitope).float(),
                torch.from_numpy(aa_features_A).float(),
                torch.from_numpy(aa_features_antibody).float(),
                sample_filename
            )
        else:
            return (
                torch.from_numpy(xyz_A_normalized).float(),
                torch.from_numpy(atom_features_A).float(),
                torch.from_numpy(xyz_antibody_normalized).float(),
                torch.from_numpy(atom_features_antibody).float(),
                torch.from_numpy(residue_starts_A).long(),
                torch.from_numpy(residue_starts_antibody).long(),
                torch.from_numpy(epitope).float(),
                torch.from_numpy(aa_features_A).float(),
                torch.from_numpy(aa_features_antibody).float(),
            )


    def get_point_counts(self):
        """Returns a list of (antigen_point_count, antibody_point_count) for each sample."""
        if self._point_counts is not None:
            return self._point_counts
        print_log(f'[DATASET] Calculating point counts for {len(self.file_list)} samples...', logger=self.logger)

        point_counts = []
        for idx in range(len(self)):
            sample = self.file_list[idx]
            data = np.load(os.path.join(self.pc_path, sample + '.npz'))
            antigen_size = data['xyz_A'].shape[0]
            antibody_size = data['xyz_H'].shape[0] + data['xyz_L'].shape[0]
            point_counts.append((antigen_size, antibody_size))
        self._point_counts = point_counts # Cache the result
        return point_counts
    
    def __len__(self):
        return len(self.file_list)
    
def custom_collate_fn(batch):
    has_sample_ids = len(batch[0]) > 9
    xyz_A_batch = [item[0] for item in batch]
    atom_features_A_batch = [item[1] for item in batch]
    xyz_antibody_batch = [item[2] for item in batch]
    atom_features_antibody_batch = [item[3] for item in batch]
    residue_starts_A_batch = [item[4] for item in batch]
    residue_starts_antibody_batch = [item[5] for item in batch]

    epitope_batch = [item[6] for item in batch]

    aa_features_A_batch = [item[7] for item in batch]
    aa_features_antibody_batch = [item[8] for item in batch]

    max_antigen_atoms = max(x.shape[0] for x in xyz_A_batch)
    max_antibody_atoms = max(x.shape[0] for x in xyz_antibody_batch)
    max_antigen_residues = max(x.shape[0] for x in aa_features_A_batch)
    max_antibody_residues = max(x.shape[0] for x in aa_features_antibody_batch)
    max_epitope_length = max(e.shape[0] for e in epitope_batch)

    padded_xyz_A = torch.stack([
        torch.cat([x, torch.zeros(max_antigen_atoms - x.shape[0], x.shape[1])], dim=0)
        for x in xyz_A_batch
    ])
    padded_atom_features_A = torch.stack([
        torch.cat([x, torch.zeros(max_antigen_atoms - x.shape[0], x.shape[1])], dim=0)
        for x in atom_features_A_batch
    ])
    padded_xyz_antibody = torch.stack([
        torch.cat([x, torch.zeros(max_antibody_atoms - x.shape[0], x.shape[1])], dim=0)
        for x in xyz_antibody_batch
    ])
    padded_atom_features_antibody = torch.stack([
        torch.cat([x, torch.zeros(max_antibody_atoms - x.shape[0], x.shape[1])], dim=0)
        for x in atom_features_antibody_batch
    ])

    atom_A_mask = torch.stack([
        torch.cat([torch.ones(x.shape[0]), torch.zeros(max_antigen_atoms - x.shape[0])])
        for x in xyz_A_batch
    ])
    atom_ab_mask = torch.stack([
        torch.cat([torch.ones(x.shape[0]), torch.zeros(max_antibody_atoms - x.shape[0])])
        for x in xyz_antibody_batch
    ])

    padded_aa_features_A = torch.stack([
        torch.cat([x, torch.zeros(max_antigen_residues - x.shape[0], x.shape[1])], dim=0)
        for x in aa_features_A_batch
    ])
    padded_aa_features_antibody = torch.stack([
        torch.cat([x, torch.zeros(max_antibody_residues - x.shape[0], x.shape[1])], dim=0)
        for x in aa_features_antibody_batch
    ])

    aa_A_mask = torch.stack([
        torch.cat([torch.ones(x.shape[0]), torch.zeros(max_antigen_residues - x.shape[0])])
        for x in aa_features_A_batch
    ])
    aa_ab_mask = torch.stack([
        torch.cat([torch.ones(x.shape[0]), torch.zeros(max_antibody_residues - x.shape[0])])
        for x in aa_features_antibody_batch
    ])

    padded_epitope = torch.stack([
        torch.cat([e, torch.full((max_epitope_length - e.shape[0],), -1.0)])
        for e in epitope_batch
    ])

    sample_ids = None
    if has_sample_ids:
        sample_ids = [item[9] for item in batch]

    result = {
        "xyz_A": padded_xyz_A,
        "atom_features_A": padded_atom_features_A,
        "xyz_antibody": padded_xyz_antibody,
        "atom_features_antibody": padded_atom_features_antibody,
        "atom_A_mask": atom_A_mask,
        "atom_ab_mask": atom_ab_mask,
        "residue_starts_A": residue_starts_A_batch,
        "residue_starts_antibody": residue_starts_antibody_batch,
        "epitope": padded_epitope,
        "aa_features_A": padded_aa_features_A,
        "aa_features_antibody": padded_aa_features_antibody,
        "aa_A_mask": aa_A_mask,
        "aa_ab_mask": aa_ab_mask,
    }
  
    if sample_ids:
        result["sample_id"] = sample_ids
        
    return result

class PointCloudBatchSampler(Sampler):
    
    def __init__(self, data_source: Dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sample_sizes = data_source.get_point_counts() 
        self.sort_indices = np.argsort([size[0] for size in self.sample_sizes])
        self.num_batches = len(self.data_source) // self.batch_size
        if not self.drop_last and len(self.data_source) % self.batch_size > 0:
            self.num_batches += 1
    
    def __iter__(self):
        indices = self.sort_indices.copy()
        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        if self.shuffle:
            np.random.shuffle(batches)
        
        return iter(batches)
    
    def __len__(self):
        return self.num_batches

class DistributedPointCloudBatchSampler(Sampler[List[int]]):
    """
    Distributed sampler that batches samples with similar sizes together within each process.

    Sorts the dataset indices based on point cloud size, partitions the sorted
    indices among processes, batches locally within each process, and shuffles
    the batches each epoch.
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False,
                 sort_key_index: int = 0): 
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                 num_replicas = 1
            else:
                 num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                 rank = 0
            else:
                 rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.sort_key_index = sort_key_index 
        try:
            self.sample_sizes = dataset.get_point_counts()
            if not self.sample_sizes or len(self.sample_sizes) != len(self.dataset):
                 raise ValueError("Inconsistent sample sizes returned by dataset.")
        except AttributeError:
            raise AttributeError("Dataset must have a 'get_point_counts' method returning List[Tuple[int, int]]")
        except Exception as e:
             raise RuntimeError(f"Error getting point counts from dataset: {e}")

        self.global_sorted_indices = np.argsort([size[self.sort_key_index] for size in self.sample_sizes])

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples_total = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples_total = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples_total * self.num_replicas

        self.num_samples_per_rank = self.num_samples_total 
        num_batches_per_rank = self.num_samples_per_rank / self.batch_size 
        self.num_batches = math.floor(num_batches_per_rank) if self.drop_last else math.ceil(num_batches_per_rank)


    def __iter__(self) -> Iterator[List[int]]:
        indices = self.global_sorted_indices

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices = np.concatenate((indices, indices[:padding_size]))
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices_for_rank = indices[self.rank::self.num_replicas] 
        assert len(indices_for_rank) == self.num_samples_per_rank

        batches = [
            indices_for_rank[i : i + self.batch_size].tolist()
            for i in range(0, self.num_samples_per_rank, self.batch_size)
        ]

        if self.drop_last and len(indices_for_rank) % self.batch_size != 0:
             batches = batches[:-1]
       
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_order] 

        assert len(batches) == self.num_batches or (len(batches) == self.num_batches -1 and self.drop_last and len(indices_for_rank) % self.batch_size != 0)


        return iter(batches)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

def create_dataloader(dataset: Dataset,
                      batch_size: int,
                      shuffle: bool = True,
                      drop_last: bool = False,
                      num_workers: int = 0,
                      worker_init_fn = None,
                      distributed: bool = False,
                      seed: int = 0,
                      split=None) -> Tuple[Optional[Sampler], DataLoader]:
    """Creates a DataLoader with appropriate sampler for single or distributed training."""
    batch_sampler = None
    if split is None:
        if distributed:
            batch_sampler = DistributedPointCloudBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle, 
                seed=seed,
                drop_last=drop_last
            )
            shuffle = None
        else:
            batch_sampler = PointCloudBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last
            )
    if split != "test":
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=custom_collate_fn,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True 
        )
    else:
        dataloader = DataLoader(
            dataset,
            collate_fn=custom_collate_fn,
            num_workers=num_workers,
            batch_size=1,
            worker_init_fn=worker_init_fn,
            pin_memory=True 
        )

    if distributed:
        return batch_sampler, dataloader 
    else:
        return None, dataloader