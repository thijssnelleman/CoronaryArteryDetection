from collections import namedtuple
from re import S

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import softmax

def calculate_components(n_nodes: int, edges: torch.tensor):
        #From https://stackoverflow.com/questions/10301000/python-connected-components
        def get_all_connected_groups(graph):
            already_seen = set()
            result = []

            def get_connected_group(node, seen):
                result = []
                nodes = set([node])
                while nodes:
                    node = nodes.pop()
                    seen.add(node)
                    #nodes = nodes or graph[node] - seen #Selects nodes unless its empty
                    nodes.update(graph[node] - already_seen)
                    result.append(node)
                return result, seen
            
            for node in graph:
                if node not in already_seen:
                    connected_group, already_seen = get_connected_group(node, already_seen)
                    result.append(connected_group)
            return result

        adj_list = {x: set() for x in range(n_nodes)} #Create an empty adjacency list for all nodes
        for edge in edges.T.tolist(): #Put values into the adjacency list  #30.9% of time
            adj_list[edge[0]].add(edge[1]) 
            adj_list[edge[1]].add(edge[0])
        
        return get_all_connected_groups(adj_list)

class ClusterPooling(torch.nn.Module):
    r""" REWRITE THIS
    
    The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.

    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["x", "edge_index", "cluster", "batch", "new_edge_score", "old_edge_score", "selected_edges", "cluster_map", "edge_mask"])

    def __init__(self, in_channels, edge_score_method=None, dropout=0.0,
                 add_to_edge_score=0.5):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        #First we drop the self edges as those cannot be clustered
        msk = edge_index[0] != edge_index[1]
        edge_index = edge_index[:,msk]

        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1) #Concatenates the source feature with the target features
        e = self.lin(e).view(-1) #Apply linear NN on the edge "features", view(-1) to reshape to 1 dimension
        e = F.dropout(e, p=self.dropout, training=self.training)
        
        
        e = self.compute_edge_score(e, edge_index, x.size(0))        
        e = e + self.add_to_edge_score
        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info
    
    """ New merge function for combining the nodes """    
    def __merge_edges__(self, x, edge_index, batch, edge_score):
        cluster = torch.empty_like(batch, device=torch.device('cpu'))

        #We don't deal with double edged node pairs e.g. [a,b] and [b,a] in edge_index
        
        if edge_index.size(1) > x.size(0): #More edges than nodes, calculate quantile node based
            quantile = 1- (int(x.size(0) / 2) / edge_index.size(1)) #Calculate the top quantile
            edge_mask = (edge_score > (torch.quantile(edge_score, quantile)))
        else: #More nodes than edges, select half of the edges
            edge_mask = (edge_score >= torch.median(edge_score))
        
        sel_edge = edge_mask.nonzero().flatten()        
        new_edge = torch.index_select(edge_index, dim=1, index=sel_edge).to(cluster.device)
        
        components = calculate_components(x.size(0), new_edge) #47.3% of time
        
        i = 0
        for c in components: #15% of time
            cluster[c] = i
            i += 1
        
        cluster = cluster.to(x.device)
        new_edge = new_edge.to(x.device)

        #We compute the new features as the average of the cluster's nodes' features
        #We can do something with the edge weights here: for each node compute scalar + mean(ni + nj), then mean all of these new features --> This has changed
        new_edge_score = edge_score[sel_edge] #Get the scores that come into play
        node_reps = (x[new_edge[0]] + x[new_edge[1]]) #/2 (used to dived by two)
        node_reps = node_reps * new_edge_score.view(-1,1)
        new_x = torch.clone(x)
        
        trans_factor = torch.bincount(new_edge.flatten())
        trans_mask = (trans_factor > 0).nonzero().flatten()
        new_x[trans_mask] = 0
        trans_factor = trans_factor[trans_mask]
        
        new_x = torch.index_add(new_x, dim=0, index=new_edge[0], source=node_reps)
        new_x = torch.index_add(new_x, dim=0, index=new_edge[1], source=node_reps)
        new_x[trans_mask] = new_x[trans_mask] / trans_factor.view(-1,1) #Some nodes get index_added more than once, so divide by that number
        #new_x = scatter_mean(new_x, cluster, dim=0, dim_size=i)
        new_x = scatter_add(new_x, cluster, dim=0, dim_size=i) #This seems to work much better in terms of backprop

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N) #Remap the edges based on cluster, and coalesce removes all the doubles

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(x=x,
                                           edge_index=edge_index,
                                           cluster=cluster,
                                           batch=batch,
                                           new_edge_score=new_edge_score,
                                           old_edge_score=edge_score,
                                           selected_edges=new_edge,
                                           cluster_map=components,
                                           edge_mask=edge_mask)

        return new_x.to(x.device), new_edge_index.to(x.device), new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.

        REWRITE

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """
        
        #We just copy the cluster feature into every node and do nothing...
        node_maps = unpool_info.cluster_map
        import numpy as np
        n_nodes = np.sum([len(c) for c in node_maps])
        repack = np.array([-1 for _ in range(n_nodes)])
        for i,c in enumerate(node_maps):
            repack[c] = i
        new_x = x[repack]
        
        """"node_factors = torch.ones(n_nodes, device=new_x.device)
        node_factors = torch.index_add(node_factors, dim=0, index=unpool_info.edge_index[0][unpool_info.edge_mask], source=unpool_info.old_edge_score[unpool_info.edge_mask])
        node_factors = torch.index_add(node_factors, dim=0, index=unpool_info.edge_index[1][unpool_info.edge_mask], source=unpool_info.old_edge_score[unpool_info.edge_mask])

        #print(node_factors)
        new_x = new_x / node_factors.view(-1,1)

        if torch.any(torch.isnan(new_x)):
            nnn = torch.isnan(new_x)
            print("nan in new_x")
            print(new_x)
            print("Percentage of nan: ", torch.sum(nnn) / (nnn.size(0) * nnn.size(1)))

            print(node_factors)
            if torch.any(torch.isnan(node_factors)):
                nnn = torch.isnan(node_factors)
                print("Percentage nf of nan: ", torch.sum(nnn) / nnn.size(0))

            input()"""
        

        #unpool_info.old_edge_score --> how can we use the old edgescores to modify the nodes coming out of the cluster? What if two edges of one node were selected?
        #print(new_x.size())
        #Now maybe do something with the edge score on the new_x?
        """edges = unpool_info.selected_edges.to(new_x.device)
        scores = unpool_info.new_edge_score.view(-1,1).to(new_x.device)
        
        reduce = torch.bincount(edges.flatten()) #How should we divide the adjusted cluster features?
        selected = reduce.nonzero().flatten()
        adapted_left = new_x[edges[0]] / scores
        adapted_right = new_x[edges[1]] / scores
        
        #edges = unpool_info.selected_edges.to(new_x.device)
        
        new_x[selected] = 0
        #print(adapted_left.size())
        #print(new_x[edges[0]].size())
        #print(edges[0].size())
        new_x[edges[0]] += adapted_left
        new_x[edges[1]] += adapted_right
        #print(new_x.size())

        
        #print("unfolded size:", new_x.size())
        #print("node count ", reduce.size())
        #print("Nodes that occur at least once", selected.size())
        #print(new_x[selected].size())
        #print(reduce[selected].size())
        new_x[selected] = new_x[selected] / reduce[selected].view(-1,1)"""
        #print(new_x.size())
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
