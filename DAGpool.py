import profile
import torch
#import torch.nn.functional as F
from torch_sparse import coalesce
from torch_scatter import scatter_mean

class DAGPool(torch.nn.Module):
    r"""Pooling layer for Directed Acyclic Graphs. Based on the desired clustersize,
        the layer will traverse from roots to leafs, creating clusters along the way.
        When a node has multiple children (branch), the node is included in the current
        segment, and the children are treated as new roots. The new nodes (clusters)
        are a concatenated representation of their features. If a cluster does not
        conform to the required cluster size, it is padded with zeros.

        In the case that your DAG is actually bidirectional, sort the edges in the manner that the "duplicates"
        are on the un-even indices and set filter_bi to True.
    """

    def __init__(self, cluster_size=2, filter_bi=True, pool_splits=False, cat_scatter=False):
        super().__init__()
        self.clusMax = cluster_size
        self.bidirec = filter_bi
        self.concat = cat_scatter

    def forward(self, x: torch.tensor, edges:torch.tensor):
        cluster = torch.zeros(x.size(0), device=torch.device('cpu'), dtype=torch.long)
        edge_cop = torch.clone(edges)

        self_edge_mask = (edges[0] != edges[1]).nonzero().flatten() #Remove self loops
        if self_edge_mask.size(0) < edges.size(1): #If we have anything to remove
            edges = torch.index_select(edges, dim=1, index=self_edge_mask)        
        
        if self.bidirec:
            mask = torch.tensor([x for x in range(edges.size(1)) if x & 1 == 1], device=edges.device)
            edges = torch.index_select(edges, dim=1, index=mask)

        #We define a root as a node with zero incoming edges
        #And a bifurcation as a node with more than two outgoing edges
        nodes = set(range(x.size(0)))
        roots = nodes - set(edges[1].tolist())
        bifurcations = (torch.bincount(edges[0]) > 1).nonzero().flatten()
        bif_mask = torch.sum(torch.stack([edges[0] == x for x in bifurcations]), dim=0, dtype=torch.bool)        

        extra_roots = edges[1][bif_mask].flatten() #Get the bifurcation recipients
        roots = list(roots)
        roots.extend(extra_roots.tolist())

        adj_list = {x: set() for x in range(x.size(0))} #Create an empty adjacency list for all nodes        

        for edge in edges.T.tolist(): #Put values into the adjacency list  #30.9% of time
            adj_list[edge[0]].add(edge[1]) 

        #Turn bifurcations into leafs
        for bif in bifurcations.tolist():
            adj_list[bif] = set()

        i = 0
        clus_count = 0
        
        if self.concat:
            x_mod = torch.cat((x, torch.zeros(1, x.size(1), device=x.device))) #Appends a zero row as a fake node
            empty_vec = x.size(0)
            clusters = [[empty_vec for _ in range(self.clusMax)]]
        
        for root in roots:
            segment = DAGPool.__get_segment__(root, adj_list)
            for node in segment:
                if i >= self.clusMax:
                    if self.concat:
                        clusters.append([empty_vec for _ in range(self.clusMax)])
                    
                    i = 0
                    clus_count +=1
                if self.concat:
                    clusters[-1][i] = node
                cluster[node] = clus_count
                i += 1
            
            i = 0
            clus_count +=1
            if self.concat:
                clusters.append([empty_vec for _ in range(self.clusMax)])

        if self.concat: #Produce the results as a concatenation of nodes
            clusters = torch.tensor(clusters, dtype=torch.long)
            new_nodes = x_mod[clusters].flatten(start_dim=1) #Select the nodes that are in clusters together, then reduce the dimension by one
        else: #Produce the results as a mean of nodes
            new_nodes = scatter_mean(x, cluster.to(x.device), dim=0, dim_size=clus_count)

        #Reroute the edges               
        N = new_nodes.size(0)
        new_edge_index, _ = coalesce(cluster[edge_cop.to(cluster.device)], None, N, N) #Remap the edges based on cluster, and coalesce removes all the doubles

        return new_nodes.to(x.device), new_edge_index.to(x.device), [x, edge_cop]

    def unpool(self, unpool):
        return unpool[0], unpool[1]
    
    def __get_segment__(node, adj_list):
        seg = [node]
        nb = adj_list[node]
        while nb != set():
            node = nb.pop()
            seg.append(node)
            nb = adj_list[node]

        return seg
        
        