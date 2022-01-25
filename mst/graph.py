import numpy as np
import heapq
from typing import Union

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """ Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. 
        Note that because we assume our input graph is undirected, `self.adj_mat` is symmetric. 
        Row i and column j represents the edge weight between vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        """

        n_vertices = self.adj_mat.shape[0] # Given the symmetry of our adjacency matrix, we can count the number of entries in the first dimension and get the number of vertices in the graph
        visited_vertices = set() # Initialize a set to keep track of the vertices we have already visited
        self.mst = np.zeros_like(self.adj_mat) # Create a matrix for our MST. It'll be initialized with 0s and have the same dimensions as our adjacency matrix
        minheap = [] # Initialize our heap for loading in the vertices and weights for their respective edges

        for i in range(n_vertices): # Add in the first set of vertices for the starting node to our heap, along with the corresponding weight
            if self.adj_mat[0][i] != 0:
                heapq.heappush(minheap, (self.adj_mat[0][i], (0, i)))

        while len(visited_vertices) <  n_vertices: # This loop continues until we have visited all vertices in the graph

            weight, vertex = heapq.heappop(minheap) # Pop the queue to get the neighboring vertex to our current vertex, and the weight of the respective edge. The weight is an integer while the vertices are a tuple
            if vertex[1] not in visited_vertices: # If we have not visited the neighboring vertex (second element in the tuple), proceed with the loop
                
                self.mst[vertex[0]][vertex[1]] = weight # Add the weight of the edge to our MST. Do it for "both sides" to maintain symmetry
                self.mst[vertex[1]][vertex[0]] = weight
                
                visited_vertices.add(vertex[1]) # Add the neighboring vertex to our set of visited vertices
                for i in range(n_vertices): # Push the next neighboring vertices to our heap
                    heapq.heappush(minheap, (self.adj_mat[vertex[1]][i], (vertex[1], i)))
            
        return self.mst # Return the MST for our graph