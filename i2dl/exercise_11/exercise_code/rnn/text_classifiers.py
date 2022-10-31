import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }
        # if you do not inherit from lightning module use the following line
        self.hparams = hparams
        self.use_lstm = hparams['use_lstm']
        # if you inherit from lightning module, comment out the previous line and use the following line
        # self.hparams.update(hparams)
        
        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
        
        num_layers = 1
        class_no = 2

        self.embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        if self.use_lstm == True:
          self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers)
        else:
          self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(num_layers * hidden_size, 1)
        self.output = nn.Sigmoid()


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################
      
        seq_length, batch_size = sequence.shape
        x = sequence
        x = self.embedding(x)
        if lengths != None:
          x = pack_padded_sequence(x,lengths)

        if self.use_lstm==True:
          _, (h, _) = self.rnn(x)
        else: 
          _, h= self.rnn(x)
        h = h.transpose(0, 1).contiguous().view(batch_size, -1)
        
        x = self.fc(h) 
        output = self.output(x)
        output = output.reshape((batch_size,))


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output
