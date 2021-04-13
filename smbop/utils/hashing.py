from itertools import *
import torch
from hashlib import blake2b
import struct
from anytree.search import *
import smbop.utils.node_util as node_util

import re


def transform(x):
    x = str(x).lower().replace("%", "").replace(" ,", "").replace(",", "").strip()
    if len(x) > 1 and x[-1] == "s":
        x = x[:-1]
    return x


def dethash(key):
    my_hash = blake2b(key="hi".encode(), digest_size=4)
    my_hash.update(transform(key).encode())
    digest = my_hash.digest()
    return struct.unpack("I", digest)[0]


class Hasher:
    def __init__(self, device):
        self.device = device
        self.tensor1 = torch.LongTensor([402653189]).to(device)
        self.tensor2 = torch.LongTensor([3644798167]).to(device)
        self.tensor3 = torch.LongTensor([28]).to(device)
        self.tensor4 = torch.LongTensor([1]).to(device)
        self.tensor5 = torch.LongTensor([56]).to(device)

    def set_hash(self, h_list, _h=None, _hash=None):
        flag = False
        if _hash is None or _h is None:
            flag = True
            _hash = torch.tensor([1], dtype=torch.long).to(self.device)
            _h = torch.tensor([1], dtype=torch.long).to(self.device)
            h_list = torch.tensor(h_list, dtype=torch.long).to(self.device)

        if len(h_list) == 3:
            parent, a, b = h_list
        else:
            parent, a = h_list
            b = a
        _hash.copy_(a)
        _h.copy_(b)
        _hash <<= 28
        _h >>= 1
        _hash = _hash.add_(_h)
        parent <<= 56
        _hash = _hash.add_(parent)
        _hash *= self.tensor2
        _hash = _hash.fmod(self.tensor1)
        if flag:
            return int(_hash[0])
        return _hash

    def add_hash_att(self, node, type_dict):
        try:
            if not node.children:
                if isinstance(node.val, dict):
                    node.hash = self.set_hash([type_dict[node.name], dethash("value")])
                else:
                    node.hash = self.set_hash(
                        [type_dict[node.name], dethash(str(node.val))]
                    )
            else:
                node.children = [
                    self.add_hash_att(child, type_dict) for child in node.children
                ]
                if node.name == "keep":
                    node.hash = node.children[0].hash
                else:
                    node.hash = self.set_hash(
                        [type_dict[node.name]] + [child.hash for child in node.children]
                    )
        except Exception as e:
            print(node_util.print_tree(node))
            raise Exception
        assert hasattr(node, "hash")
        return node
