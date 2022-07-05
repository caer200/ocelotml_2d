import torch
import torch.nn as nn
from dgl.nn.pytorch import Set2Set
from dgllife.model.model_zoo.mpnn_predictor import MPNNGNN


class MPNNPredictor_evidential(nn.Module):
    """MPNN for regression and classification on graphs.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 dropout=0,
                 num_layer_set2set=3, descriptor_feats=0):
        super(MPNNPredictor_evidential, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * node_out_feats + descriptor_feats, node_out_feats),
            nn.ReLU(),
            nn.BatchNorm1d(node_out_feats),
            nn.Linear(node_out_feats, 4 * n_tasks)
        )

    def forward(self, g, node_feats, edge_feats,concat_feats=None):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        if concat_feats != None:
            final_feats = torch.cat((graph_feats, concat_feats), dim=1)
        else:
            final_feats = graph_feats
        output = self.predict(final_feats)

        min_val = 1e-6
        # Split the outputs into the four distribution parameters
        means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1] // 4, dim=1)
        lambdas = nn.Softplus()(loglambdas) + min_val
        alphas = nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = nn.Softplus()(logbetas) + min_val

        # Return these parameters as the output of the model
        output = torch.stack((means, lambdas, alphas, betas),
                             dim=2).view(output.size())
        return output
