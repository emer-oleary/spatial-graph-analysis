import numpy as np
import tifffile as tiff
from scipy.ndimage import label, distance_transform_edt
from skimage.measure import regionprops
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import csv
import statistics
import numpy as np
from collections import defaultdict, deque


def read_am(file_path):
    """
    Read metadata and data from Amira (.am) ASCII file as a single string.
    
    """
    with open(file_path, 'r') as file:
        # Read the entire file as a single string
        content = file.read()
    
    # Extract number of vertices, edges, and points from the header
    lines = content.splitlines()
    num_vertices = None
    num_edges = None
    num_points = None
    
    for line in lines:
        if line.startswith("define VERTEX"):
            num_vertices = int(line.split()[2])
        elif line.startswith("define EDGE"):
            num_edges = int(line.split()[2])
        elif line.startswith("define POINT"):
            num_points = int(line.split()[2])
        
    # Locate the first occurrence of "@1"
    first_idx_1 = content.find("@1")
    
    # Locate the second occurrence of "@1"
    second_idx_1 = content.find("@1", first_idx_1 + len("@1"))
    
    # Trim the string after the second occurrence of "@1"
    first_points_string = content[second_idx_1 + len("@1"):].strip()
    
    # Find first_points
    first_points = []
    for index, line in enumerate(first_points_string.splitlines()):
        values = list(map(float, line.split()))
        if len(values) != 3:
            break
        else:
            # Index, x, y, z with 2nd and 4th columns swapped
            first_points.append([index, values[2], values[1], values[0]])
    
    # Find segment_node_pairs
    segment_node_pairs_string = first_points_string.split("@2", 1)[1].strip()
    segment_node_pairs = []
    for index, line in enumerate(segment_node_pairs_string.splitlines(), 1):
        values = list(map(int, line.split()))
        if len(values) != 2:
            break
        else:
            segment_node_pairs.append([values[0], values[1]])
    
    # Find pts_per_segment
    pts_per_segment_string = segment_node_pairs_string.split("@3", 1)[1].strip()
    pts_per_segment = []
    for index, line in enumerate(pts_per_segment_string.splitlines(), 1):
        if "@4" in line:
            break
        values = list(map(int, line.split()))
        if len(values) != 1:
            break
        else:
            pts_per_segment.append([values[0]])
            
    # Find all_points
    all_points_string = pts_per_segment_string.split("@4", 1)[1].strip()
    all_points = []
    for index, line in enumerate(all_points_string.splitlines(), 1):
        if "@5" in line:
            break
        values = list(map(float, line.split()))
        if len(values) != 3:
            break
        else:
            # Index, x, y, z with 2nd and 4th columns swapped
            all_points.append([index, values[2], values[1], values[0]])
    
    # Find thickness
    thickness_string = all_points_string.split("@5", 1)[1].strip()
    thickness = []
    for index, line in enumerate(thickness_string.splitlines(), 1):
        values = list(map(float, line.split()))
        if len(values) != 1:
            break
        else:
            thickness.append([values[0]])
    
    return (num_vertices, num_edges, num_points, 
            first_points, segment_node_pairs, 
            pts_per_segment, all_points, thickness)

def calculate_segment_mean_radii(pts_per_segment, radii):
    """
    Calculates the mean radius for each segment.
    Each segment's radius is computed from a consecutive block of radii,
    where the block length is given by the corresponding value in pts_per_segment.

    Parameters:
        pts_per_segment (list of [int]): Number of points in each segment (e.g., [[2], [3]])
        radii (list of [float]): Radii values for each point in each segment (e.g., [[2], [2], [3], [4], [2]])

    Returns:
        list of float: Mean radius for each segment.
    """
    mean_radii = []
    flat_radii = [r[0] for r in radii]  # Flatten [[2], [2], ...] -> [2, 2, ...]
    segment_lengths = [p[0] for p in pts_per_segment]  # Flatten [[2], [3]] -> [2, 3]

    idx = 0
    for seg_len in segment_lengths:
        segment_radii = flat_radii[idx:idx + seg_len]
        mean_radii.append(sum(segment_radii) / seg_len)
        idx += seg_len

    return mean_radii


def calculate_subgraphs(segments):
    # Build undirected adjacency list
    adjacency = defaultdict(set)
    for seg in segments:
        a, b = seg
        adjacency[a].add(b)
        adjacency[b].add(a)

    visited = set()
    subgraph_count = 0

    def bfs(start_node):
        queue = deque([start_node])
        visited.add(start_node)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    # Loop through all nodes
    all_nodes = set(adjacency.keys())
    for node in all_nodes:
        if node not in visited:
            bfs(node)
            subgraph_count += 1

    return subgraph_count


def calculate_segment_lengths_LDR_tortuosity(segment_radii, pts_per_segment, points):
    segment_lengths = []
    length_diameter_ratios = []
    tortuosities = []
    
    point_idx = 0  # Running index into the points list
    
    for i, seg_pts in enumerate(pts_per_segment):
        num_pts = seg_pts[0]  # e.g., [72]
        
        # Extract relevant points
        segment_points = points[point_idx:point_idx + num_pts]
        coords = np.array([p[1:] for p in segment_points])  # [x, y, z]
        
        # Total path length
        path_dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        segment_length = np.sum(path_dists)
        segment_lengths.append(segment_length)
        
        # Length-diameter ratio (LDR)
        radius = segment_radii[i]
        ldr = segment_length / (2 * radius) if radius > 0 else np.nan
        length_diameter_ratios.append(ldr)

        # Tortuosity = straight line distance / path length
        if len(coords) >= 2:
            straight_line = np.linalg.norm(coords[-1] - coords[0])
            tortuosity = segment_length / straight_line if straight_line > 0 else np.nan
        else:
            tortuosity = np.nan
        tortuosities.append(tortuosity)

        point_idx += num_pts

    return segment_lengths, length_diameter_ratios, tortuosities

''' # Local branching - evaluates 1st point along segment in spatial graph - this can vary between different algorithms so I think the global (segment end point) method is better for now
def calculate_branching_angles_local(nodes, segments, segment_radii, segment_lengths, pts_per_segment, points):
    # Convert nodes to dictionary for faster access
    node_coords = {int(n[0]): np.array([n[1], n[2], n[3]]) for n in nodes}  # index: [z, y, x]
    
    # Step 1: Build degree map for each node
    from collections import defaultdict
    node_to_segments = defaultdict(list)  # node index -> list of segment indices
    for idx, (n1, n2) in enumerate(segments):
        node_to_segments[n1].append(idx)
        node_to_segments[n2].append(idx)
    
    # Calculate cumulative points per segment
    # Fix: Extract the integer value from each pts_per_segment element
    cumulative_pts = [0]
    for i in range(len(pts_per_segment)):
        # If pts_per_segment is a list of lists, extract the value correctly
        pts_value = pts_per_segment[i] if isinstance(pts_per_segment[i], int) else pts_per_segment[i][0]
        cumulative_pts.append(cumulative_pts[-1] + pts_value)
    
    branching_angles = []
    # Step 2: Loop through nodes of degree 3
    for node_idx, seg_indices in node_to_segments.items():
        if len(seg_indices) != 3:
            continue  # skip non-degree-3 nodes
        
        # Step 3: Identify parent (segment with largest radius)
        radii = [segment_radii[i] for i in seg_indices]
        parent_idx = seg_indices[np.argmax(radii)]
        child_indices = [i for i in seg_indices if i != parent_idx]
        
        # Step 4: Find central node coordinates
        center = node_coords[node_idx]
        
        # Step 5: For each child segment, find the second point along the segment
        child_vectors = []
        for seg_idx in child_indices:
            n1, n2 = segments[seg_idx]
            
            # Determine if central node is first or second in segment
            if n1 == node_idx:
                # Option 1: [central_node_index, child_node_index]
                # Second point is at start of segment + 1
                point_idx = cumulative_pts[seg_idx] + 1
            else:
                # Option 2: [child_node_index, central_node_index]
                # Second point is at end of segment - 1
                point_idx = cumulative_pts[seg_idx + 1] - 2
            
            # Get coordinates of second point
            child_point = np.array(points[point_idx][1:4])  # [z, y, x]
            
            # Calculate vector from center to child point
            vec = child_point - center
            child_vectors.append(vec)
        
        # Normalize child vectors
        norm1 = np.linalg.norm(child_vectors[0])
        v1 = child_vectors[0] / norm1 if norm1 > 0 else np.zeros_like(child_vectors[0])
        norm2 = np.linalg.norm(child_vectors[1])
        v2 = child_vectors[1] / norm2 if norm2 > 0 else np.zeros_like(child_vectors[1])
        
        # Step 6: Compute angle in degrees
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        branching_angles.append(angle_deg)
    
    return branching_angles

''' 
# Global branching angle (measurement points located at segment end points) Note: global method is better for comparison of different alogrithms. ...
# ... The local approach measures the 1st point along the segment but the point spacing varies between algorithms.)
def calculate_branching_angles_global(nodes, segments, segment_radii, segment_lengths):
    # Convert nodes to dictionary for faster access
    node_coords = {int(n[0]): np.array([n[1], n[2], n[3]]) for n in nodes}  # index: [z, y, x]
    
    # Step 1: Build degree map for each node
    from collections import defaultdict
    node_to_segments = defaultdict(list)  # node index -> list of segment indices
    for idx, (n1, n2) in enumerate(segments):
        node_to_segments[n1].append(idx)
        node_to_segments[n2].append(idx)
    
    branching_angles = []

    # Step 2: Loop through nodes of degree 3
    for node_idx, seg_indices in node_to_segments.items():
        if len(seg_indices) != 3:
            continue  # skip non-degree-3 nodes

        # Step 3: Identify parent (segment with largest radius)
        radii = [segment_radii[i] for i in seg_indices]
        parent_idx = seg_indices[np.argmax(radii)]
        child_indices = [i for i in seg_indices if i != parent_idx]

        # Step 4: Find central node coordinates
        center = node_coords[node_idx]

        # Step 5: For each child segment, find the *other* node to get direction
        child_vectors = []
        for seg_idx in child_indices:
            n1, n2 = segments[seg_idx]
            other_node = n2 if n1 == node_idx else n1
            vec = node_coords[other_node] - center
            child_vectors.append(vec)

        # Normalize child vectors
        norm1 = np.linalg.norm(child_vectors[0])
        v1 = child_vectors[0] / norm1 if norm1 > 0 else np.zeros_like(child_vectors[0])
        norm2 = np.linalg.norm(child_vectors[1])
        v2 = child_vectors[1] / norm2 if norm2 > 0 else np.zeros_like(child_vectors[1])

        # Step 6: Compute angle in degrees
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        branching_angles.append(angle_deg)

    return branching_angles


def calculate_vessel_density(num_segments, image_size, image_resolution):
    # Calculate total volume in µm³
    x_vox, y_vox, z_vox = image_size
    voxel_volume_um3 = image_resolution ** 3
    total_volume_um3 = x_vox * y_vox * z_vox * voxel_volume_um3

    # Convert µm³ to mm³
    total_volume_mm3 = total_volume_um3 / 1e9

    # Vessel density = segments / total volume in mm³
    vessel_density_mm3 = num_segments / total_volume_mm3

    return vessel_density_mm3

'''
def write_to_am(am_file_write_path, nodes, connections, points, radii, pts_per_segment):
    
    # Get counts
    num_vertices = len(nodes)
    num_edges = len(connections)
    num_points = len(points)
    
    # Writing header as string
    output_string = "# Avizo 3D ASCII 3.0\n \n \n"
    output_string += f"define VERTEX {num_vertices} \ndefine EDGE {num_edges} \ndefine POINT {num_points} \n \n"
    output_string += """Parameters {\n    ContentType "HxSpatialGraph" \n} \n \n"""
    output_string += "VERTEX { float[3] VertexCoordinates } @1 \n"
    output_string += "EDGE { int[2] EdgeConnectivity } @2 \n"
    output_string += "EDGE { int NumEdgePoints } @3 \n"
    output_string += "POINT { float[3] EdgePointCoordinates } @4 \n"
    output_string += "POINT { float thickness } @5 \n\n"  
    output_string += "@1\n"
    for node in nodes:
        output_string += f"{float(node[3]):.15f} {float(node[2]):.15f} {float(node[1]):.15f}\n"
    output_string += "\n@2\n"
    for connection in connections:
        output_string += f"{connection[0]} {connection[1]} \n"
    output_string += "\n@3\n"
    for pts in pts_per_segment:
        output_string +=f"{pts[0]}\n"  
    output_string += "\n@4\n"
    for point in points:
        output_string += f"{float(point[3]):.15e} {float(point[2]):.15e} {float(point[1]):.15e}\n"
    output_string += "\n@5\n"
    for radius in radii:
        output_string += f"{float(radius[0]):.15e}\n"
    # Write to file
    with open(am_file_write_path, 'w') as f:
        f.write(output_string)
'''



def main():
    #------------------------------------3 User Inputs--------------------------
    am_folder_path = r"D:\Skeletonization\NewVersion\Latin Hypercube Sampling\Min_Skel_Metric_SpatialGraphs_HiP_CT" # Path to folder containing spatial graph files (_.am) to process
    image_resolution = 1  # microns per voxel (Default:1 - only change if the spatial graph resolution does not match the image resolution)
    image_size = [2216, 2216, 2216]  # in voxels

    # -------------------------- Init CSV Output Dictionaries --------------------------
    segment_radii_dict = {}
    segment_lengths_dict = {}
    length_diameter_ratio_dict = {}
    branching_angles_dict = {}
    tortuosity_dict = {}

    max_segments = 0

    print("Processing .am files...\n")
    print("Units: Vessel density in segments/mm³, radii and lengths in µm, branching angles in degrees, all other measures are dimensionless.\n")
    print("Nodes   Segments  Points   Subgraphs   Vessel_Density  Seg_Rad_Avg   Seg_Rad_Std  Seg_Len_Avg   Seg_Len_Std  LDR_Avg   LDR_Std  Tort_Avg  Tort_Std Branch_Avg  Branch_Std  Spatial_Graph")

    for filename in os.listdir(am_folder_path):
        if filename.endswith(".am"):
            am_file_path = os.path.join(am_folder_path, filename)
            num_vertices, num_edges, num_points, nodes, segments, pts_per_segment, points, radii = read_am(am_file_path)
            no_subgraphs = calculate_subgraphs(segments)

            # Vessel density
            num_segments = len(segments)
            vessel_density = calculate_vessel_density(num_segments, image_size, image_resolution)

            # Segment Radii
            segment_radii = np.array(calculate_segment_mean_radii(pts_per_segment, radii)) * image_resolution
            segment_radii_dict[filename] = segment_radii
            if len(segment_radii) > max_segments:
                max_segments = len(segment_radii)
            mean_r = np.mean(segment_radii)
            std_r = np.std(segment_radii)

            # Segment Length, LDR, and Tortuosity
            segment_lengths, length_diameter_ratio, segment_tortuosity = calculate_segment_lengths_LDR_tortuosity(
                segment_radii, pts_per_segment, points
            )
            segment_lengths = np.array(segment_lengths) * image_resolution
            segment_lengths_dict[filename] = segment_lengths
            mean_len = np.mean(segment_lengths)
            std_len = np.std(segment_lengths)

            length_diameter_ratio_dict[filename] = length_diameter_ratio
            ldr_array = np.array(length_diameter_ratio)
            mean_ldr = np.nanmean(ldr_array)
            std_ldr = np.nanstd(ldr_array)

            tortuosity_dict[filename] = segment_tortuosity
            tort_array = np.array(segment_tortuosity)
            mean_tort = np.nanmean(tort_array)
            std_tort = np.nanstd(tort_array)

            # Branching Angles
            branching_angles = calculate_branching_angles_global(nodes, segments, segment_radii, segment_lengths)
            branching_angles_dict[filename] = branching_angles
            branch_array = np.array(branching_angles)
            mean_branch = np.nanmean(branch_array)
            std_branch = np.nanstd(branch_array)

            # --- Print all metrics in a single line ---
            print(f"{num_vertices:<7} {num_edges:<9} {num_points:<8} {no_subgraphs:<11} "
                  f"{vessel_density:<15.4f} {mean_r:<13.4f} {std_r:<12.4f} "
                  f"{mean_len:<13.4f} {std_len:<12.4f} {mean_ldr:<9.4f} {std_ldr:<8.4f} "
                  f"{mean_tort:<9.4f} {std_tort:<8.4f} {mean_branch:<11.4f} {std_branch:<10.4f}  {filename:<15}")

    # -------------------------- CSV Writers --------------------------
    def write_dict_to_csv(data_dict, output_name, label):
        output_path = os.path.join(am_folder_path, output_name)
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ["Segment_Index"] + list(data_dict.keys())
            writer.writerow(headers)
            for i in range(max_segments):
                row = [i]
                for key in data_dict:
                    values = data_dict[key]
                    row.append(values[i] if i < len(values) else "")
                writer.writerow(row)
        print(f"{label} saved to {output_path}")

    # -------------------------- Write All CSVs --------------------------
    write_dict_to_csv(segment_radii_dict, "mean_segment_radii.csv", "Mean Segment Radii")
    write_dict_to_csv(segment_lengths_dict, "segment_lengths.csv", "Segment Lengths")
    write_dict_to_csv(length_diameter_ratio_dict, "length_diameter_ratios.csv", "Length-Diameter Ratios")
    write_dict_to_csv(branching_angles_dict, "branching_angles.csv", "Branching Angles")
    write_dict_to_csv(tortuosity_dict, "segment_tortuosity.csv", "Segment Tortuosity")

if __name__ == "__main__":
    main()
    