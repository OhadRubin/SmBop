from contextlib import contextmanager
import glob
import io
import os
import logging
import tempfile
import json
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import timedelta
from fnmatch import fnmatch
from os import PathLike
from urllib.parse import urlparse
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    Union,
    IO,
    Callable,
    Set,
    List,
    Iterator,
    Iterable,
    Dict,
    NamedTuple,
    MutableMapping,
)
from hashlib import sha256
from functools import wraps
from weakref import WeakValueDictionary
from zipfile import ZipFile, is_zipfile
import tarfile
import shutil
import pickle
import time
import warnings

import boto3
import botocore
import torch
from botocore.exceptions import ClientError, EndpointConnectionError
from filelock import FileLock as _FileLock
import numpy as np
from overrides import overrides
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
import lmdb
from torch import Tensor
import dill
from allennlp.common.tqdm import Tqdm
def _serialize(data):
    buffer = pickle.dumps(data, protocol=-1)
    return np.frombuffer(buffer, dtype=np.uint8)

class TensorCache:
    """
    This is a key-value store, mapping strings to tensors. The data is kept on disk,
    making this class useful as a cache for storing tensors.
    `TensorCache` is also safe to access from multiple processes at the same time, so
    you can use it in distributed training situations, or from multiple training
    runs at the same time.
    """

    def __init__(
        self,
        filename: Union[str, PathLike],
        *,
        map_size: int = 1024 * 1024 * 1024 * 1024,
        read_only: bool = False,
    ) -> None:
        """
        Creates a `TensorCache` by either opening an existing one on disk, or creating
        a new one. Its interface is almost exactly like a Python dictionary, where the
        keys are strings and the values are `torch.Tensor`.
        Parameters
        ----------
        filename: `str`
            Path to the location of the cache
        map_size: `int`, optional, defaults to 1TB
            This is the maximum size the cache will ever grow to. On reasonable operating
            systems, there is no penalty to making this a large value.
            `TensorCache` uses a memory-mapped file to store the data. When the file is
            first opened, we have to give the maximum size it can ever grow to. This is
            that number. Reasonable operating systems don't actually allocate that space
            until it is really needed.
        """
        filename = str(filename)

        cpu_count = os.cpu_count() or 1
        if os.path.exists(filename):
            if os.path.isfile(filename):
                # If the file is not writable, set read_only to True, but issue a warning.
                if not os.access(filename, os.W_OK):
                    if not read_only:
                        warnings.warn(
                            f"File '{filename}' is read-only, so cache will be read-only",
                            UserWarning,
                        )
                    read_only = True
            else:
                # If it's not a file, raise an error.
                raise ValueError("Expect a file, found a directory instead")

        use_lock = True
        if read_only:
            # Check if the lock file is writable. If it's not, then we won't be able to use the lock.

            # This is always how lmdb names the lock file.
            lock_filename = filename + "-lock"
            if os.path.isfile(lock_filename):
                use_lock = os.access(lock_filename, os.W_OK)
            else:
                # If the lock file doesn't exist yet, then the directory needs to be writable in
                # order to create and use the lock file.
                use_lock = os.access(os.path.dirname(lock_filename), os.W_OK)

        if not use_lock:
            warnings.warn(
                f"Lacking permissions to use lock file on cache '{filename}'.\nUse at your own risk!",
                UserWarning,
            )

        self.lmdb_env = lmdb.open(
            str(filename),
            subdir=False,
            map_size=map_size,
            max_readers=cpu_count * 2,
            max_spare_txns=cpu_count * 2,
            metasync=False,
            sync=True,
            readahead=False,
            meminit=False,
            readonly=read_only,
            lock=use_lock,
        )

        # We have another cache here that makes sure we return the same object for the same key. Without it,
        # you would get a different tensor, using different memory, every time you call __getitem__(), even
        # if you call it with the same key.
        # The downside is that we can't keep self.cache_cache up to date when multiple processes modify the
        # cache at the same time. We can guarantee though that it is up to date as long as processes either
        # write new values, or read existing ones.



    def __contains__(self, key: object):
        encoded_key = str(key).encode()
        with self.lmdb_env.begin(write=False) as txn:
            result = txn.get(encoded_key)
            return result is not None

    # def __getitem__(self, key):

    #     encoded_key = str(key).encode()
    #     with self.lmdb_env.begin(write=False) as txn:
    #         buffer = txn.get(encoded_key)
    #         if buffer is None:
    #             raise KeyError()
                
    #         tensor = dill.load(io.BytesIO(buffer))
    #     return tensor
    
    # def get_all(self, max_instances = 1000000):

    #     with self.lmdb_env.begin() as txn:
    #         big_dict = {k: io.BytesIO(v)
    #                     for  i,(k, v) in enumerate(txn.cursor().iternext()) if i<max_instances}
    #     big_dict = {k: dill.load(v) for (k, v) in big_dict.items()}
    #     return big_dict
                
            
    def write(self, instance_list):
        with self.lmdb_env.begin() as txn:
            length = txn.stat()['entries']
        with self.lmdb_env.begin(write=True) as txn:
            encoded_key = str(length).encode()
            buffer = io.BytesIO()
            dill.dump(instance_list, buffer)
            txn.put(encoded_key, buffer.getbuffer())

    def __iter__(self):
        # It is not hard to implement this, but we have not needed it so far.
        with self.lmdb_env.begin() as txn:
            for k, instance_list in txn.cursor().iternext():
                yield from dill.load(io.BytesIO(instance_list))


