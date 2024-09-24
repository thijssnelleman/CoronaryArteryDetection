""" #For Nils' Histograms: Calculate the distribution of cluster sizes"""
        """if self.show_cluster_plots:
        #if True or not self.training:
            clusters_per_layer = []
            for i,p in enumerate(unpool_infos): #These are ordered in sequence
                clusters = p.cluster_map
                if i > 0: #We have to fix this dynamically

                    #cluster_size = [0 for range(len(cluster))]
                    #prev_map = unpool_infos[-1].cluster_map #Get how many nodes were in the previous clusters
                    unfolded_clusters = []
                    tot_nodes = []
                    for i, clus in enumerate(clusters): #for every cluster we have
                        cur_clus = []
                        for node in clus:
                            cur_clus.extend(clusters_per_layer[-1][node])
                        unfolded_clusters.append(cur_clus)
                        tot_nodes.extend(cur_clus)
                    clusters = unfolded_clusters

                clusters_per_layer.append(clusters)
            
            cluster_sizes = [] #for each layer
            
            for idx, layer in enumerate(clusters_per_layer):
                lngths = []
                labs = torch.tensor([0 for _ in range(data.y.size(0))]).to(data.y.device)
                for cluster in layer:
                    lngths.append(len(cluster))
                    clus_dist = torch.sum(data.y[cluster]) / len(cluster)

                    if clus_dist > 0.5: 
                        labs[cluster] = 1

                cluster_sizes.append(lngths)
                
                self.cf1[idx].append(metrics.f1_score(data.y.int().cpu().detach().numpy(), labs.int().cpu().detach().numpy(), zero_division=0))
                #print(torch.sum(labs == data.y) / data.y.size(0))


            #We can create a histogram from cluster sizes
            
            #for i,l in enumerate(cluster_sizes):
            #    plt.hist(l)
            #    plt.ylabel("Number of clusters")
            #    plt.xlabel("# Nodes in cluster")
            #    plt.title("Cluster sizes " + str(i+1) + " (" + str(data.y.size(0))  +")")
            #    plt.show()
            #And a box plot from cluster impurities?
            #print(cluster_f1)
            if not self.training and not self.shown:
                
                #plt.plot([x+1 for x in range(len(self.cf1))], self.cf1)#, labels=["Layer " +str(i+1) for i in range(len(unpool_infos))])
                plt.boxplot(self.cf1)
                plt.xlabel("Layers in model")
                plt.ylabel("Best Cluster Based F1 score")
                plt.title("Cluster F1 over " +str(len(unpool_infos)) + " " + str(self.poolingType.__name__) + " Layers")
                plt.show()

                for i in range(self.depth):
                    self.cf1[i] = []
                self.shown = True
            elif self.training:
                self.shown = False"""