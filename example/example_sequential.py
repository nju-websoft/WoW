import numpy as np
import pywowlib


# --- Helper Function ---
def print_results(index_name, filter_name, results):
    """Prints search results, clarifying that the second item is the Vector ID (0 to N-1)."""
    print(f"\n--- Results from {index_name} using {filter_name} ---")
    if not results:
        print("  No results found.")
        return
    print("  [Distance, Vector ID]")  # Vector ID is 0-based
    for dist, vec_id in results:
        print(f"  [{dist:.4f}, {vec_id}]")
    print("-" * (len(index_name) + len(filter_name) + 25))


# =========================================================================
# == Example Usage: WoWIndexSequential (Vector ID 0..N-1 IS the Filter) ==
# =========================================================================
print("\n" + "=" * 70)
print("== Testing WoWIndexSequential (Vector ID 0..N-1 is ALSO the Filter) ==")
print("=" * 70)

att_type = "label"  # Vector ID is the filter attribute

# 1. Parameters
num_vectors_seq = 100000  # Number of vectors to insert
max_elements_seq = num_vectors_seq + 20  # Must be >= num_vectors_seq
vec_d_seq = 16
M_seq = 16
efc_seq = 128
space_seq = "ip"
vec_dtype = np.float32  # Data type for vectors

# 2. Create WoWIndexSequential instance by setting att_type to "label"
try:
    seq_index = pywowlib.WoWIndex(
        max_elements_seq, vec_d_seq, M_seq, efc_seq, space_seq, att_type=att_type
    )
    print(
        f"Created WoWIndexSequential instance (max_elements={max_elements_seq}, vec_d={vec_d_seq}, space='{space_seq}')"
    )
except Exception as e:
    print(f"Error creating WoWIndexSequential: {e}")
    exit()

# 3. Prepare data
# Vector IDs are 0 to num_vectors_seq - 1. These are also the filter attributes.
vector_ids_seq = list(range(num_vectors_seq))
# Generate vectors
vectors_seq = np.random.rand(num_vectors_seq, vec_d_seq).astype(vec_dtype)
vectors_seq /= np.linalg.norm(vectors_seq, axis=1, keepdims=True)  # Normalize for IP

# 4. Insert data into WoWIndexSequential
print(f"\nInserting {num_vectors_seq} vectors into WoWIndexSequential...")

try:
    # insert(vector_id (0..N-1), vector_data)
    # The vector ID is automatically used as the filter attribute internally.
    # seq_index.insert(vector_ids_seq[0], vectors_seq[0], vector_ids_seq[0])

    seq_index.bulk_insert(vector_ids_seq, vectors_seq, vector_ids_seq, threads=4)
except Exception as e:
    print(f"  ERROR inserting: {e}")
print("Insertion finished.")

seq_index.save("example_sequential_index.wow")  # Save the index to a file
print("Index saved to 'example_sequential_index.wow'.")
seq_index = pywowlib.WoWIndexLoad(
    location="example_sequential_index.wow", space_name=space_seq, att_type=att_type
)  # Load the index from a file

# 5. Prepare query vector
query_vec_seq = np.random.rand(vec_d_seq).astype(vec_dtype)
query_vec_seq /= np.linalg.norm(query_vec_seq)  # Normalize
print(f"\nGenerated query vector (shape: {query_vec_seq.shape}) for IP space")

# 6. Prepare Filters for WoWIndexSequential (filtering directly by Vector ID 0..N-1)
k_seq = 5
efs_seq = 100

# Filter 1: WoWRangeLabelFilter - Filter by Vector ID range
# Find vectors whose Vector ID is between 50 and 99 (inclusive)
range_id_filter = pywowlib.WoWRangeFilter(att_type, 50, 99)  # Use 0-based IDs
print(
    f"\nCreated RangeLabelFilter (Filtering by Vector ID): {range_id_filter.l_} <= Vector ID <= {range_id_filter.u_}"
)

# Filter 2: WoWBitsetLabelFilter - Filter by specific Vector IDs (0-based)
# Allow only vectors with IDs 0, 10, 20, ..., 140
ids_to_allow = list(range(0, num_vectors_seq, 10))
# Max label for bitset needs to accommodate the highest ID (num_vectors_seq - 1)
max_id_val_bitset = num_vectors_seq  # Size must be > max ID
bitset_id_filter = pywowlib.WoWSetFilter(att_type, max_id_val_bitset)
bitset_id_filter.clear()  # Clear the bitset, please manually clear and set the IDs
for vec_id in ids_to_allow:
    bitset_id_filter.set(vec_id)
print(
    f"Created BitsetLabelFilter (Allowing specific 0-based Vector IDs): {ids_to_allow[0:10]}"
)

# 7. Perform Searches on WoWIndexSequential
# Results contain Vector IDs (0 to N-1) that satisfy the filter (which checks the ID itself).
try:
    results_range_id = seq_index.searchKNN(
        query_vec_seq, efs_seq, k_seq, range_id_filter
    )
    print_results(
        "WoWIndexSequential", "RangeLabelFilter (Vector IDs 50-99)", results_range_id
    )
    # Expect Vector IDs within the range [50, 99]

    results_bitset_id = seq_index.searchKNN(
        query_vec_seq, efs_seq, k_seq, bitset_id_filter
    )
    print_results(
        "WoWIndexSequential",
        f"BitsetLabelFilter (Vector IDs {ids_to_allow[0:10]})",
        results_bitset_id,
    )
    # Expect Vector IDs from the allowed list (0, 10, 20...)

except Exception as e:
    print(f"\n!!!!!!!!!!\nSearch failed on WoWIndexSequential: {e}\n!!!!!!!!!!")

print("\nExample finished.")
