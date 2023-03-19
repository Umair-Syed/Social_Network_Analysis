import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

"""# Question 1"""

# Load CSV data into a Pandas dataframe
df = pd.read_csv('edges_list.csv')

# Create a dictionary that maps node names to indices in the adjacency matrix
node_dict = {node: i for i, node in enumerate(set(df['src']) | set(df['dst']))}

# Create an empty adjacency matrix
adj_matrix = np.zeros((len(node_dict), len(node_dict)))

# Fill the adjacency matrix with edge weights
for _, row in df.iterrows():
    src_idx = node_dict[row['src']]
    dst_idx = node_dict[row['dst']]
    adj_matrix[src_idx, dst_idx] = row['weight']

# Create a graph using the adjacency matrix
UG = nx.Graph(adj_matrix)

# Set the node labels to be the node names
node_labels = {i: node for node, i in node_dict.items()}
nx.set_node_attributes(UG, node_labels, 'label')

# Draw the graph using NetworkX and Matplotlib
pos = nx.spring_layout(UG)
nx.draw(UG, pos, labels=node_labels, with_labels=True, node_size=500, font_color ='w')
edge_labels = nx.get_edge_attributes(UG, 'weight')
nx.draw_networkx_edge_labels(UG, pos, edge_labels=edge_labels, font_color='red')
print(adj_matrix)
plt.show()

# Create a graph using the adjacency matrix
DG = nx.DiGraph(adj_matrix)

# Set the node labels to be the node names
node_labels = {i: node for node, i in node_dict.items()}
nx.set_node_attributes(DG, node_labels, 'label')

# Draw the graph using NetworkX and Matplotlib
pos = nx.spring_layout(DG)
nx.draw(DG, pos, labels=node_labels, with_labels=True, node_size=500, font_color ='w')
edge_labels = nx.get_edge_attributes(DG, 'weight')
nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, font_color='red')
plt.show()

"""# Question 2

## For undirected
"""

# Number of nodes
print("Number of nodes: " + str(UG.number_of_nodes()))

# Nnumber of edges
print("Number of nodes: " + str(UG.number_of_edges()))

# Node with max degree
max_degree = max(UG.degree(), key=lambda x: x[1])[1] # (index, degree)
print(f"Nodes with maximum degree of {max_degree} are {[UG.nodes[idx]['label'] for idx, deg in UG.degree() if deg == max_degree]}")

# Node with min degree
min_degree = min(UG.degree(), key=lambda x: x[1])[1]
print(f"Nodes with maximum degree of {min_degree} are {[UG.nodes[idx]['label'] for idx, deg in UG.degree() if deg == min_degree]}")

"""## For directed"""

# Number of nodes
print("Number of nodes: " + str(DG.number_of_nodes()))

# Nnumber of edges
print("Number of nodes: " + str(DG.number_of_edges()))

in_degrees = DG.in_degree()
out_degrees = DG.out_degree()

# Node with the maximum in-degree
max_indegree = max(in_degrees, key=lambda x: x[1])[1] # (index, degree)
print(f"Nodes with maximum in-degree of {max_indegree} are {[DG.nodes[idx]['label'] for idx, deg in in_degrees if deg == max_indegree]}")

# Node with the minimum in-degree
min_indegree = min(in_degrees, key=lambda x: x[1])[1] # (index, degree)
print(f"Nodes with minimum in-degree of {min_indegree} are {[DG.nodes[idx]['label'] for idx, deg in in_degrees if deg == min_indegree]}")

# Node with the maximum out-degree
max_outdegree = max(out_degrees, key=lambda x: x[1])[1] # (index, degree)
print(f"Nodes with maximum out-degree of {max_outdegree} are {[DG.nodes[idx]['label'] for idx, deg in out_degrees if deg == max_outdegree]}")

# Print the node with the minimum out-degree
min_outdegree = min(in_degrees, key=lambda x: x[1])[1] # (index, degree)
print(f"Nodes with minimum out-degree of {min_outdegree} are {[DG.nodes[idx]['label'] for idx, deg in out_degrees if deg == min_outdegree]}")


"""# Question 3"""

print(UG.edges(2, data=True))

# Get the sum of weights of all outgoing edges for each vertex
outgoing_weights = {}
for node in UG.nodes():
    outgoing_weights[node] = sum([edge[2]['weight'] for edge in UG.edges(node, data=True)])

# Print the sum of weights of outgoing edges for each vertex
for node, weight in outgoing_weights.items():
    print(f"Node {UG.nodes[node]['label']}: Sum of weights edges = {round(weight, 2)}")

# Get the sum of weights of all outgoing edges for each vertex
outgoing_weights = {}
for node in DG.nodes():
    outgoing_weights[node] = sum([edge[2]['weight'] for edge in DG.out_edges(node, data=True)])

# Get the sum of weights of all incoming edges for each vertex
incoming_weights = {}
for node in DG.nodes():
    incoming_weights[node] = sum([edge[2]['weight'] for edge in DG.in_edges(node, data=True)])

# Print the sum of weights of incoming and outgoing edges for each vertex
for node, outgoing_weight in outgoing_weights.items():
    incoming_weight = incoming_weights[node]
    print(f"Node {DG.nodes[node]['label']}: Sum of weights of outgoing edges = {round(outgoing_weight, 2)}, Sum of weights of incoming edges = {round(incoming_weight, 2)}")

"""# Question 4"""

# Degree Centrality
degree_centrality = nx.degree_centrality(UG)

# Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(UG)

# Closeness Centrality
closeness_centrality = nx.closeness_centrality(UG)

# PageRank
page_rank = nx.pagerank(UG)

# Eigenvector Centrality
eigenvector_centrality = nx.eigenvector_centrality(UG)

# Create a dictionary to hold the centrality measures for each node
centrality = {'Node': [], 'Degree Centrality': [], 'Betweenness Centrality': [], 'Closeness Centrality': [], 'PageRank': [], 'Eigenvector Centrality': []}

# Populate the dictionary with the centrality measures for each node
for node in UG.nodes():
    centrality['Node'].append(UG.nodes[node]['label'])
    centrality['Degree Centrality'].append(round(degree_centrality[node], 2))
    centrality['Betweenness Centrality'].append(round(betweenness_centrality[node], 2))
    centrality['Closeness Centrality'].append(round(closeness_centrality[node], 2))
    centrality['PageRank'].append(round(page_rank[node], 2))
    centrality['Eigenvector Centrality'].append(round(eigenvector_centrality[node], 2))

# Create a Pandas dataframe from the centrality dictionary
df2 = pd.DataFrame(centrality)
df2 = df2.sort_values(by=['Node'])

# Print the dataframe
print(df2.to_string(index=False))

# Create a list of dictionaries containing centrality measures and corresponding nodes
centrality_list = [
    {
        'centrality': 'degree centrality',
        'min node': df2['Node'].iloc[df2['Degree Centrality'].idxmin()],
        'min score': df2['Degree Centrality'].min(),
        'max node': df2['Node'].iloc[df2['Degree Centrality'].idxmax()],
        'max score': df2['Degree Centrality'].max()        
    },
    {
        'centrality': 'betweenness centrality',
        'min node': df2['Node'].iloc[df2['Betweenness Centrality'].idxmin()],
        'min score': df2['Betweenness Centrality'].min(),
        'max node': df2['Node'].iloc[df2['Betweenness Centrality'].idxmax()],
        'max score': df2['Betweenness Centrality'].max()        
    },
    {
        'centrality': 'closeness centrality',
        'min node': df2['Node'].iloc[df2['Closeness Centrality'].idxmin()],
        'min score': df2['Closeness Centrality'].min(),
        'max node': df2['Node'].iloc[df2['Closeness Centrality'].idxmax()],
        'max score': df2['Closeness Centrality'].max()                
    },
    {
        'centrality': 'page rank',
        'min node': df2['Node'].iloc[df2['PageRank'].idxmin()],
        'min score': df2['PageRank'].min(),
        'max node': df2['Node'].iloc[df2['PageRank'].idxmax()],
        'max score': df2['PageRank'].max()        
    },
    {
        'centrality': 'eigenvector centrality',
        'min node': df2['Node'].iloc[df2['Eigenvector Centrality'].idxmin()],
        'min score': df2['Eigenvector Centrality'].min(),
        'max node': df2['Node'].iloc[df2['Eigenvector Centrality'].idxmax()],
        'max score': df2['Eigenvector Centrality'].max()        
    },
]

# Create a new dataframe with the list of dictionaries
df3 = pd.DataFrame(centrality_list)

# Set the index to be the centrality measure
df3 = df3.set_index('centrality')

# Display the dataframe with formatted values
print("Centrality Measures and Corresponding Nodes:\n")
print(df3.to_string(formatters={
    'min score': '{:.2f}'.format,
    'max score': '{:.2f}'.format
}))

