
import unifac_database
import pandas as pd
import numpy as np
from collections import Counter

def get_unifac_parameters(selected_components):
    """Retrieve UNIFAC parameters dynamically and return in a single 'params' object."""

    unifac_params = {"NG": 0, "v": [], "Rk": [], "Qk": [], "groups": set()}
    component_group_counts = []
    unique_groups = set()

    for comp in selected_components:
        if comp in unifac_database.unifac_molecules:
            # Count occurrences of functional groups in the molecule
            group_counts = Counter(unifac_database.unifac_molecules[comp])

            # Ensure groups are added in the correct order
            unifac_params["groups"].update(group_counts.keys())
            
            # Store the unique subgroups
            unique_groups.update(group_counts.keys())

            # Store the count for this component
            component_group_counts.append(group_counts)

    # Convert groups to a sorted list for consistent indexing
    sorted_groups = sorted(unifac_params["groups"])
    unifac_params["NG"] = len(sorted_groups)

    # Create 'v' matrix (rows = components, columns = functional groups)
    for group_count in component_group_counts:
        v_row = [group_count.get(group, 0) for group in sorted_groups]
        unifac_params["v"].append(v_row)

    # Fill Rk and Qk arrays based on sorted groups
    for group in sorted_groups:
        data = unifac_database.unifac_groups[group]
        unifac_params["Rk"].append(data["Rk"])
        unifac_params["Qk"].append(data["Qk"])

    # Retrieve interaction parameters
    interaction_params = get_interaction_parameters(sorted_groups, unifac_database.list_Interactions)

    # Merge everything into a single 'params' dictionary
    params = unifac_params
    params["a"] = interaction_params  # Add interaction parameter matrix

    return params


def get_interaction_parameters(selected_groups, list_Interactions):
    """Retrieve UNIFAC-LLE interaction parameters for the selected functional groups."""

    # Convert subgroup names to main group numbers
    main_groups = [unifac_database.unifac_groups[group]["main_group"] for group in selected_groups]

    # Convert list into a dictionary for faster lookups
    interaction_dict = {(entry['i'], entry['j']): (entry['Aij'], entry['Aji']) for entry in list_Interactions}

    # interaction_matrix = {}

    # Create an interaction matrix with dimensions based on the number of subgroups (NG × NG)
    NG = len(selected_groups)
    interaction_matrix = [[0.0] * NG for _ in range(NG)]

    # Assign interaction parameters while preserving each subgroup instance
    for i, group_i in enumerate(selected_groups):
        for j, group_j in enumerate(selected_groups):
            main_i = unifac_database.unifac_groups[group_i]["main_group"]
            main_j = unifac_database.unifac_groups[group_j]["main_group"]

            if (main_i, main_j) in interaction_dict:
                interaction_matrix[i][j] = interaction_dict[(main_i, main_j)][0]  # Aij
                interaction_matrix[j][i] = interaction_dict[(main_i, main_j)][1]  # Aji
            elif (main_j, main_i) in interaction_dict:
                interaction_matrix[i][j] = interaction_dict[(main_j, main_i)][1]  # Aji
                interaction_matrix[j][i] = interaction_dict[(main_j, main_i)][0]  # Aij
            else:
                interaction_matrix[i][j] = 0.0  # Default if no interaction is found

    return interaction_matrix

# Function to print UNIFAC parameters in a clearer format
def print_unifac_parameters(unifac_params):
    print("\n--- UNIFAC-LLE Parameters ---")
    print(f"NG (Number of Functional Groups): {unifac_params['NG']}\n")

    print("v (Group Contribution Matrix):")
    for idx, v_entry in enumerate(unifac_params["v"]):
        print(f"  Group {idx+1}: {v_entry}")
    print()

    print("Rk (Volume Parameters):")
    print(unifac_params["Rk"])
    print()

    print("Qk (Surface Area Parameters):")
    print(unifac_params["Qk"])
    print()

    print("Groups Present in the Mixture:")
    print(unifac_params["groups"])
    print()

def print_interaction_matrix(interaction_params, selected_groups):
    """Prints the UNIFAC interaction parameters (Aij and Aji) as a matrix."""

    # Sort selected groups to ensure consistent ordering
    sorted_groups = sorted(selected_groups)

    # Create an empty DataFrame for storing interaction parameters
    matrix_size = len(sorted_groups)
    interaction_matrix = np.zeros((matrix_size, matrix_size))

    # Fill the matrix correctly, ensuring no overwriting
    for (i, j), (Aij, Aji) in interaction_params.items():
        if i in sorted_groups and j in sorted_groups:
            row = sorted_groups.index(i)
            col = sorted_groups.index(j)
            
            # Only set Aij if it's not already written
            if interaction_matrix[row, col] == 0:
                interaction_matrix[row, col] = Aij  # Aij at (i, j)
            
            # Only set Aji if it's not already written
            if interaction_matrix[col, row] == 0:
                interaction_matrix[col, row] = Aji  # Aji at (j, i)

    # Convert to a Pandas DataFrame for easy visualization
    interaction_df = pd.DataFrame(interaction_matrix, index=sorted_groups, columns=sorted_groups)

    print("\n--- Corrected UNIFAC Interaction Matrix (Aij) ---")
    print(interaction_df)

    return interaction_df