# spatial-graph-analysis

calculate_vessel_network_properties:
- reads in spatial graph files (_.am) of segmented and skeletonized 3D blood vessel networks
- calculates the segment radii, lengths, length-to-diameter ratios (LDR), vessel densities, tortuosity, and global branching angles in the network
- print statistics of the 6 measures in the command window
- exports .csv files containing the measures

plot_vessel_network_properties:
- reads in one of the .csv file outputs of calculate_vessel_network_properties.py (e.g. branching_angles.csv)
- exports a .png plot (a combination violin-box-scatter plot)
