import numpy as np
import torch
from pygments.lexers.asn1 import word_sequences
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt


if __name__ == '__main__':
    batch_size = 2 # mini_batch size
    embedding_size = 2 # embedding size

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",\
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]
    word_sequences = " ".join(sentences).split()