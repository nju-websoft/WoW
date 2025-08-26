# PyWoWLib: Python Bindings for WoWIndex

**PyWoWLib** provides Python bindings for the C++ `WoWIndex` library, enabling efficient approximate nearest neighbor (ANN) search with attribute filtering capabilities. This implementation uses [pybind11](https://github.com/pybind/pybind11) and focuses on indices where attributes are restricted to predefined Plain Old Data (POD) types (integers, floats, fixed-size strings) chosen at index creation time.

## Key Features

*   **Fast ANN Search:** Leverages the underlying C++ `WoWIndex` for efficient similarity search.
*   **Attribute Filtering:** Supports filtering search results based on associated attributes during KNN search using range, set, or bitset filters.
*   **Predefined POD Attribute Types:** Ensures type safety and efficient memory handling by requiring attributes to be one of several supported C++ POD types (`int32`, `int64`, `float`, `double`, fixed-size strings, etc.).
*   **Fixed-Size String Support:** Includes bindings for `FixedString<N>` types for safe handling of string attributes within the C++ core.
*   **Simplified Python API:** Provides a factory function (`pywowlib.WoWIndex`) to create index instances by specifying the desired attribute type as a string (e.g., `"int32"`, `"string16"`).
*   **Batch Insertion:** Supports efficient, parallel insertion of multiple vectors using OpenMP (requires compiler support).
*   **Filter Factories:** Helper functions (`pywowlib.WoWRangeFilter`, `pywowlib.WoWSetFilter`) to create filters corresponding to the index's attribute type.

## Installation

### Prerequisites

*   **Python:** 3.7 or higher.
*   **C++ Compiler:** A modern C++ compiler supporting C++17.
*   **CMake:** Recommended for managing the build process (pybind11 integrates well with it).
*   **(Optional but Recommended) OpenMP:** Required for parallel batch insertion (`bulk_insert`). Ensure your compiler supports it and the necessary runtime libraries are installed.
    *   Linux: Often included with GCC/Clang (e.g., `libgomp1`).
    *   macOS: Install via Homebrew (`brew install libomp`).
    *   Windows: Included with Visual Studio C++ workloads.
*   **Pip:** Python package installer.

### Build Dependencies

Install necessary Python packages for building:
```bash
pip install pybind11>=2.6 numpy setuptools wheel
```

### Installing from Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ziqiwww/wowlib.git
    cd wowlib/python
    ```
2.  **Install the package:**
    ```bash
    pip install . -v
    ```
    *   The `-v` flag provides verbose output, useful for seeing compiler messages and confirming OpenMP flags are used.
    *   This command invokes `setup.py`, compiles the C++ extension, and installs the `pywowlib` package into your Python environment.

    **For development:** Use an editable install to make changes to the Python wrapper (`__init__.py`) effective immediately (C++ changes still require reinstalling/rebuilding):
    ```bash
    pip install -e . -v
    ```

## Basic Usage (Quick Start)

```python
import pywowlib
import numpy as np

# 1. Define Index Parameters
num_items = 1000
dim = 64
attribute_type = "int32" # Choose the attribute type for this index

# 2. Create Index using the factory
index = pywowlib.WoWIndex(
    max_elements=num_items + 10, # Allow some extra space
    vec_d=dim,
    M=16,            # Maximum number of connections per node
    efc=100,         # Beam search width during construction
    o=4,             # Window boosting base, suggested to be 2 or 4
    space_name='l2', # Distance space ('l2', 'ip', 'cosine')
    att_type=attribute_type
)
print(f"Created index with attribute type: {attribute_type}")

# 3. Prepare Data
# Vector IDs (labels) should be 0 to N-1
vector_ids = list(range(num_items))
# Vectors (NumPy array, correct dtype)
vectors = np.random.rand(num_items, dim).astype(np.float32)
# Attributes (must match att_type, e.g., Python ints for "int32")
attributes = [np.int32(i % 50) for i in range(num_items)]

# 4. Insert Data (using parallel batch insert)
print("Inserting data...")
index.bulk_insert(vector_ids, vectors, attributes, threads=4) # Use 'bulk_insert'
print("Insertion complete.")

# 5. Prepare Query and Filter
query_vector = np.random.rand(dim).astype(np.float32)

# Create a filter matching the attribute type
# Find items where attribute (int32) is between 10 and 20
my_filter = pywowlib.WoWRangeFilter(
    att_type=attribute_type,
    lower_bound=np.int32(10),
    upper_bound=np.int32(20)
)

# 6. Search
k = 5 # Number of neighbors
efs = 50 # Beam search width for the search phase, should be larger than k.
print("\nSearching...")
results = index.searchKNN(query_vector, efs=efs, k=k, filter=my_filter)

# 7. Process Results
print("\nSearch Results ([Distance, Vector ID]):")
if results:
    for distance, vec_id in results:
        print(f"  [{distance:.4f}, {vec_id}]")
else:
    print("  No results found matching the filter.")

# 8. Save Index (Optional)
# index.save("./my_index.wow")
```

## Detailed API

### Index Creation (`pywowlib.WoWIndex`)

Use the factory function to create or load an index instance.

```python
index = pywowlib.WoWIndex(
    max_elements: int,          # Max number of vectors the index can hold
    vec_d: int,                 # Dimension of the vectors
    M: int,                     # WoW parameter (connections per node)
    efc: int,                   # WoW parameter (efConstruction)
    space_name: str,            # Distance space: 'l2', 'ip', 'cosine'
    att_type: str,              # Specifies the C++ attribute type (see below)
    o: int = 4,                 # (Not recommended to set manually) WoW parameter, window boosting base
    wp: int = 11,               # (Not recommended to set manually) WoW parameter, expected number of windows
    auto_raise_wp: bool = True, # (Not recommended to set manually) WoW parameter, auto-raise window count according to the number of inserted vectors by calculating log_o(max_elements/2)
)

index = pywowlib.WoWIndexLoad(
    location: str,              # Path to the saved index file
    att_type: str,              # Must match the type of the saved index!
    space_name: str,            # Must match the space of the saved index
)
```

*   **`att_type` (string):** Specifies the C++ POD type used for storing and filtering attributes. This choice determines which underlying C++ index specialization is created. Supported values:
    *   `"int32"`: `int32_t`
    *   `"int64"`: `int64_t`
    *   `"uint32"`: `uint32_t`
    *   `"uint64"`: `uint64_t`
    *   `"float32"` or `"float"`: `float`
    *   `"float64"` or `"double"`: `double`
    *   `"string16"`: Fixed-size string (16 bytes incl. null terminator).
    *   `"string32"`: Fixed-size string (32 bytes incl. null terminator).
    *   `"label"`: Uses the internal `LabelType` (typically `uint64_t`) as the attribute (equivalent to `"uint64"` if `LabelType` is `uint64_t`). Creates the same index type as `WoWIndexSequential` internally.
*   **`location` (string, optional):** If provided, attempts to load a previously saved index from the given path instead of creating a new one.
    *   **Warning:** When loading, you *must* still provide an `att_type` string that *exactly matches* the C++ attribute type the index was originally saved with. This factory cannot verify the type from the file. Using the wrong `att_type` during load will lead to undefined behavior or crashes.

### Attribute Types

The `att_type` chosen during index creation dictates the type of attribute data you pass to `insert` and use in filters.

*   **Numeric Types:** Pass standard Python `int` or `float`. Pybind11 handles conversion to the corresponding C++ type (e.g., Python `int` -> C++ `int32_t` if `att_type="int32"`). Be mindful of potential overflow if Python integers exceed the C++ type's range.
*   **String Types (`"stringN"`):** Pass standard Python `str`.
    *   These map to C++ `FixedString<N>`.
    *   Strings longer than `N-1` characters **will be silently truncated** during insertion or when used in filters.
    *   You can also explicitly create `FixedString` objects in Python if needed:
        ```python
        fs16 = pywowlib.FixedString16("my_tag")
        print(fs16.value)      # Output: my_tag
        print(fs16.capacity()) # Output: 16

        fs16_long = pywowlib.FixedString16("this_string_is_definitely_longer_than_15")
        print(fs16_long.value) # Output: this_string_is_ (truncated)
        ```

### Insertion

*   **Single Insert:**
    ```python
    index.insert(
        label: int,          # Vector ID (typically 0 to max_elements-1)
        vector: np.ndarray,  # 1D NumPy array, dtype=np.float32, shape=(vec_d,)
        attribute: Any,      # Python object matching the index's 'att_type'
                             # (e.g., int for "int32", str for "string16")
        replace_deleted: bool = False # Optional flag (behavior defined by C++ core)
    )
    ```

*   **Batch Parallel Insert:** Uses OpenMP in the C++ binding layer for parallelism. `WoWIndex::insert` method is thread-safe.
    ```python
    index.bulk_insert( # Renamed from insert_batch_parallel for better Python feel
        vector_ids: List[int],        # List of Vector IDs
        vectors_batch: np.ndarray,    # 2D NumPy array, shape=(N, vec_d), dtype=np.float32
        attributes_batch: List[Any],  # List of attributes matching 'att_type' (len=N)
        replace_deleted: bool = False, # Optional flag
        threads: int = 4              # Number of OpenMP threads to suggest
    )
    ```

### Filter Creation

Use factory functions matching the index `att_type`.

*   **Range Filter:** Selects items whose attribute falls within `[lower_bound, upper_bound]`.
    ```python
    range_filter = pywowlib.WoWRangeFilter(
        att_type: str,    # Must match index's att_type
        lower_bound: Any, # Value matching att_type
        upper_bound: Any  # Value matching att_type
    )
    # Example:
    # filter_int = pywowlib.WoWRangeFilter("int32", 10, 20)
    # filter_str = pywowlib.WoWRangeFilter("string16", "cat", "dog")
    ```

*   **Set Filter:** Selects items whose attribute is present in the specified set.
    ```python
    set_filter = pywowlib.WoWSetFilter(
        att_type: str     # Must match index's att_type
    )
    set_filter.add(attribute_value: Any) # Add allowed attributes (must match att_type)
    # Example:
    # filter_float = pywowlib.WoWSetFilter("float32")
    # filter_float.add(1.0)
    # filter_float.add(3.14)
    # print(filter_float.allowed_set) # View the set (read-only property)
    ```

*   **Bitset Label Filter:** Filters directly by Vector ID (label), independent of `att_type`. Useful for excluding specific known items or only searching within a subset of IDs.

**Attention!!!** Bitset label filters are only compatible with indices created with `att_type="label"`. And is created by inserting labels as attribute values. They are not compatible with other attribute types and the result quality is not guaranteed.

    ```python
    bitset_filter = pywowlib.WoWBitsetLabelFilter(
        max_label: int # Max Vector ID expected + 1
    )
    bitset_filter.set(vector_id_to_allow: int)
    bitset_filter.reset(vector_id_to_disallow: int)
    # bitset_filter.test(vector_id: int) -> bool
    # bitset_filter.clear()
    ```

### Searching (`searchKNN`)

```python
results: List[Tuple[float, int]] = index.searchKNN(
    query_vec: np.ndarray, # 1D NumPy array, shape=(vec_d,), dtype=np.float32
    efs: int,              # HNSW search parameter (exploration factor)
    k: int,                # Number of nearest neighbors to return
    filter: Any            # A filter object created via factories or WoWBitsetLabelFilter
)
```

*   Returns a list of `(distance, vector_id)` tuples, sorted by distance.
*   The `filter` object must be compatible with the index (e.g., use `WoWRangeFilter("int32", ...)` with an index created using `att_type="int32"`). Passing an incompatible filter will likely result in a C++ runtime error or incorrect results.
*   Passing `filter=None` gives you the nearest neighbors without any filtering, the performance is equivalent to searching on the bottom level of the HNSW graph.

### Saving and Loading

*   **Save:**
    ```python
    index.save("path/to/my_index.wow")
    ```
*   **Load:** Use the `location` argument in the `pywowlib.WoWIndex` factory.
    ```python
    loaded_index = pywowlib.WoWIndexLoad(
        location="path/to/my_index.wow",
        att_type="int32", # MUST match the type of the saved index!
        space_name="l2"   # Must match space of the saved index
        # Other parameters like max_elements, vec_d, M, etc. are read from the file
    )
    ```

## Full Example
Please check `wow/example/example_arbitrary,sequential.py` for a complete example demonstrating the usage of the library, including index creation, insertion, filtering, and searching.

## Building from Source (Details)

If you need to build the package locally (e.g., for development or if a pre-built wheel is not available for your platform):

1.  Ensure all prerequisites listed under Installation are met.
2.  Clone the repository.
3.  Navigate to the root directory (containing `setup.py`).
4.  **(Optional) Clean previous builds:**
    ```bash
    rm -rf build dist pywowlib.egg-info pywowlib/_pywowlib_core*.so
    python setup.py clean --all
    ```
5.  **Build the extension:**
    *   **Editable install (recommended for development):**
        ```bash
        pip install -e . -v
        ```
    *   **Standard install:**
        ```bash
        pip install . -v
        ```
    *   **Build extension in place (for testing without installing):**
        ```bash
        python setup.py build_ext --inplace
        ```
        (Make sure your Python path includes the current directory or the `pywowlib` subdirectory).

Check the build output carefully for any C++ compiler warnings or errors.