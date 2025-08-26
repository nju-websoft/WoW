import numpy as np
import pywowlib  # Import the package
import time


def print_results(index_name, filter_name, results):  # (Same as before)
    print(f"\n--- Results from {index_name} using {filter_name} ---")
    if not results:
        print("  No results found.")
        return
    print("  [Distance, Vector ID]")
    for dist, vec_id in results:
        print(f"  [{dist:.4f}, {vec_id}]")
    print("-" * (len(index_name) + len(filter_name) + 25))


print("=" * 70)
print("== Testing Unified WoWIndex Factory with POD types ==")
print("=" * 70)

nb = 100000
vec_d = 64
vecs = np.random.rand(nb, vec_d).astype(np.float32)

# --- Create an index with int32 attributes ---
try:
    print("\n--- Index: Int32 Attributes ---")
    att_list = np.random.randint(0, 1000, nb).astype(np.int32)
    index_i32 = pywowlib.WoWIndex(
        max_elements=nb, vec_d=vec_d, M=16, efc=100, space_name="l2", att_type="int32"
    )
    print(f"Created: {type(index_i32)}")  # Should be <class 'pywowlib._pywowlib_core._WoWIndexInt32Attr'>
    try:
        index_i32.bulk_insert(
            list(range(nb)), vecs, att_list.tolist()
        )  # Insert all vectors with their attributes
    except Exception as e:
        print(f"  ERROR inserting: {e}")
    print("Insertion finished.")
    # Search for the nearest neighbors of a random vector
    query_vector = np.random.rand(vec_d).astype(np.float32)
    filt_i32 = pywowlib.WoWRangeFilter(
        att_type="int32", lower_bound=75, upper_bound=150
    )  # Filter for int32 attributes
    results = index_i32.searchKNN(query_vector, efs=1000, k=10, filter=filt_i32)
    # sort results by distance
    results.sort(key=lambda x: x[0])
    print_results("WoWIndex(att_type='int32')", "RangeFilter(att_type='int32'), efs=1000", results)
    results = index_i32.searchKNN(query_vector, efs=100, k=10, filter=filt_i32)
    results.sort(key=lambda x: x[0])
    print_results("WoWIndex(att_type='int32')", "RangeFilter(att_type='int32'), efs=100", results)
    # ground truth
    

except Exception as e:
    print(f"Error with int32 index: {e}")

# --- Create an index with FixedString<16> attributes ---
try:
    print("\n--- Index: String16 Attributes ---")
    index_s16 = pywowlib.WoWIndex(
        max_elements=1000, vec_d=16, M=16, efc=100, space_name="ip", att_type="string16"
    )
    print(f"Created: {type(index_s16)}")

    vecs_s16 = np.random.rand(3, 16).astype(np.float32)
    index_s16.insert(10, vecs_s16[0], "alpha")
    index_s16.insert(
        11, vecs_s16[1], "beta_long_string"
    )  # Will be truncated by FixedString<16>
    index_s16.insert(12, vecs_s16[2], "gamma")

    # User can also explicitly create FixedString objects if they want more control
    # attr_explicit_fs16 = pywowlib.FixedString16("delta")
    # index_s16.insert(13, np.random.rand(16).astype(np.float32), attr_explicit_fs16)

    filt_s16 = pywowlib.WoWSetFilter(att_type="string16")
    filt_s16.add("alpha")
    filt_s16.add("gamma")

    results = index_s16.searchKNN(
        np.random.rand(16).astype(np.float32), 50, 3, filt_s16
    )
    print_results(
        "WoWIndex(att_type='string16')", "SetFilter(att_type='string16')", results
    )

    # Batch insert example
    ids_batch = [20, 21, 22]
    vecs_batch = np.random.rand(len(ids_batch), 16).astype(np.float32)
    attrs_batch = [
        "zeta_batch",
        "eta_batch_long",
        "theta_batch",
    ]  # List of Python strings
    index_s16.bulk_insert(ids_batch, vecs_batch, attrs_batch)
    print("Batch inserted string16 attributes.")

except Exception as e:
    print(f"Error with string16 index: {e}")


print("\nExample finished.")
