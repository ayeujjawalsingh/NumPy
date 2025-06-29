# 📊 NumPy for Python: Complete Guide

## 🔹 What is Numpy?

**NumPy (Numerical Python)** is a fundamental Python library for numerical computing, especially for working with arrays. It provides the `ndarray` object, a fast and efficient multidimensional array, along with many routines for linear algebra, Fourier transforms, and more. Compared to Python lists, NumPy arrays are stored in contiguous memory and are optimized for performance on modern CPUs.

---

## 🔍 Why Use NumPy?

- **High performance**: Up to 50× faster than standard Python lists.
- **Multidimensional array support**: Efficient manipulation and broadcasting.
- **Rich ecosystem**: Integrates well with SciPy, pandas, matplotlib, etc.
- **Open source**: Widely used in the scientific and data science community.

---

## 🛠 Installation

```bash
pip install numpy
```

## 🔹 Getting Started

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)             # [1 2 3 4 5]
print(type(arr))       # <class 'numpy.ndarray'>
print(np.__version__)  # Check version
```

## 🔹 Creating Arrays

```python
# 1D array
np.array([1, 2, 3])

# 2D array
np.array([[1, 2], [3, 4]])

# 3D array
np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

## 🔹 Indexing
**1-D arrays**

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr[0])  # 1 (first element)
print(arr[2])  # 3 (third element)
print(arr[1] + arr[3])  # 2 + 4 = 6
```

**2-D arrays**

```python
import numpy as np

arr2d = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10]])
print(arr2d[0, 1])  # 2 (1st row, 2nd column)
print(arr2d[1, 4])  # 10 (2nd row, 5th column)
```

## 🔹 Slicing

Slicing in NumPy works like Python lists: `arr[start:stop]` extracts elements from index `start` up to (but not including) `stop`. If you omit `start`, it defaults to the beginning; if you omit `stop`, it goes to the end. Negative indices count from the end of the array.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5])    # [2 3 4 5] (from index 1 to 4)
print(arr[:4])     # [1 2 3 4] (up to index 4, not including 4)
print(arr[4:])     # [5 6 7] (from index 4 to end)
print(arr[-3:-1])  # [5 6] (third from end to second from end)
```

You can also specify a step: `arr[start:stop:step]` takes every step-th element.

```python
print(arr[::2])   # [1 3 5 7] (every other element)
print(arr[1:6:2]) # [2 4 6] (from index 1 to 5, step 2)
```

For **2-D arrays**, you can slice rows and columns:

```python
arr2d = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10]])
# Slice columns 1 to 3 (index 1 inclusive to 4 exclusive) of row 1:
print(arr2d[1, 1:4])  # [7 8 9]

# From both rows, get column 2:
print(arr2d[:, 2])    # [3 8] (3rd column of each row)
```

## 🔹 Data Types 

Every NumPy array has a data type (dtype) that tells you what kind of data it contains. You can check arr.dtype to see the type:

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr.dtype)  # e.g. int64

arr_str = np.array(['apple', 'banana', 'cherry'])
print(arr_str.dtype)  # <U6 (Unicode string)
```

When creating an array, you can specify the desired data type with the `dtype` argument:

```python
arr = np.array([1, 2, 3, 4], dtype='i4')  # 4-byte (32-bit) integer
print(arr, arr.dtype)   # e.g. [1 2 3 4] int32

arr = np.array([1.0, 2.0, 3.0], dtype=float)
print(arr.dtype)        # float64
```

If you try to force a type that doesn’t make sense (e.g., converting 'a' to integer), NumPy will raise a ValueError.

To convert an existing array to a different type, use the `astype()` method. This creates a new array (copy) with the desired type:

```python
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype(int)
print(newarr, newarr.dtype)  # [1 2 3] int64
```

## 🔹 Copy vs View

By default, assigning or slicing an array doesn’t copy it; it only creates a **view** (a new array object that references the same data). A **view** is linked to the original array: changes in one affect the other​. In contrast, a copy is an independent array with its own data​.

**Copy:** Use arr.copy() to make a full copy of the array. Changes to the original do not affect the copy​.

**View:** Use arr.view() (or slicing) to get a view. The view shares data with the original; modifying one changes the other​.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
copy_arr = arr.copy()
arr[0] = 42
print(arr)      # [42  2  3  4  5]
print(copy_arr) # [1 2 3 4 5] – copy was not affected

arr = np.array([1, 2, 3, 4, 5])
view_arr = arr.view()
arr[1] = 99
print(arr)      # [ 1 99  3  4  5]
print(view_arr) # [ 1 99  3  4  5] – view reflects change
```

You can check whether an array owns its data using the `base` attribute: `None` means it owns its data (copy), otherwise it points to the original (view).

## 🔹 Array Shape

The shape of an array is a tuple giving the size in each dimension​. For example, a 2×4 matrix has shape `(2, 4)`. You can get an array’s shape with `arr.shape`:

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]])
print(arr.shape)   # (2, 4) – 2 rows, 4 columns
```

If you create an array with the `ndmin` argument, you can force a minimum number of dimensions:

```python
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print(arr.shape)  # e.g. (1,1,1,1,4) – 5 dimensions
```

You can also use attributes like `arr.ndim` (number of dimensions) and `arr.size` (total number of elements) for more info.

## 🔹 Array Reshape

You can **reshape** an array to a different shape as long as the total number of elements remains the same​. Use `arr.reshape(new_shape)` to create a reshaped view (or copy in some cases).

Example – reshape a 1-D array of length 12 into a 2-D array 4×3:

```python
import numpy as np

arr = np.arange(1, 13)   # [1,2,...,12]
newarr = arr.reshape(4, 3)
print(newarr)
print(newarr.shape)      # (4, 3)
```

Example – reshape to 3-D:

```python
arr = np.arange(1, 13)
newarr = arr.reshape(2, 3, 2)
print(newarr.shape)  # (2, 3, 2)
```

**Constraint:** You can only reshape if the total elements match. For instance, you cannot reshape 8 elements into a 3×3 array (which requires 9 elements)​. Attempting to do so will raise an error.

You can use one **unknown dimension** `(set as -1)`, and NumPy will automatically calculate its size:

```python
arr = np.arange(1, 9)
newarr = arr.reshape(2, 2, -1)
print(newarr.shape)  # (2, 2, 2)
```

## 🔹 Array Iterating

You can use Python loops to iterate over NumPy arrays. The behavior depends on the dimensions:
- 1-D array: Looping directly yields each element.
- 2-D array: Looping yields each row (a 1-D array). Use nested loops to access elements.
- ND arrays: Looping goes along the first dimension. Use nested loops for deeper access.

Example – 1-D:

```python
import numpy as np

arr = np.array([1, 2, 3])
for x in arr:
    print(x)  # 1, then 2, then 3
```

Example – 2-D:

```python
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
for row in arr2:
    print(row)      # prints each sub-array

for row in arr2:
    for val in row:
        print(val)  # prints each element
```

Example – 3-D:

```python
arr3 = np.array([[[1, 2], [3, 4]],
                 [[5, 6], [7, 8]]])
for x in arr3:
    print(x)  # prints each 2-D block

# Nested loops to reach scalars:
for x in arr3:
    for y in x:
        for z in y:
            print(z)  # prints 1,2,3,4,5,6,7,8
```

For convenience, NumPy provides np.nditer() to iterate over **every element** of an array regardless of dimension:

```python
for x in np.nditer(arr3):
    print(x)  # iterates through all scalar elements
```

And if you need index positions, use `np.ndenumerate()` to get tuples of (index, value):

```python
arr = np.array([10, 20, 30])
for idx, x in np.ndenumerate(arr):
    print(idx, x)  # (0,) 10, (1,) 20, (2,) 30
```

## 🔹 Array Join

**Joining** arrays means concatenating them along an existing axis. Use functions like `np.concatenate`, `np.stack`, or helper functions `hstack/vstack/dstack`.


**np.concatenate:** Joins a sequence of arrays along a given axis (default axis 0)​
```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)  # [1 2 3 4 5 6]

# Concatenate 2-D arrays along columns (axis=1)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.concatenate((A, B), axis=1)
print(C)
# [[1 2 5 6]
#  [3 4 7 8]]  (2×4)
```

**np.stack:** Similar to concatenate, but joins along a new axis​

```python
arr = np.stack((arr1, arr2), axis=1)
print(arr)
# [[1 4]
#  [2 5]
#  [3 6]]  (stacked as columns)
```

**Horizontal/Vertical stacking:** Helper functions for common cases.
- `np.hstack((A, B)):` Stack arrays horizontally (column-wise). Equivalent to axis=1 for 1-D or row-wise for 2-D.
- `np.vstack((A, B)):` Stack vertically (row-wise)​. Equivalent to axis=0.
- `np.dstack((A, B)):` Stack along the third axis (depth).

```python
arr = np.hstack((arr1, arr2))
print(arr)  # [1 2 3 4 5 6] (same as concatenate for 1-D)

arr = np.vstack((arr1, arr2))
print(arr)
# [[1 2 3]
#  [4 5 6]]  (two rows)

# dstack example (for 2-D inputs, stacks depth-wise)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.dstack((arr1, arr2))
print(arr.shape)  # (1, 3, 2) – one "layer" of shape 3×2
```

## 🔹 Array Split

Splitting breaks one array into multiple sub-arrays. Use functions like `np.array_split`, `np.split`, `np.hsplit`, `np.vsplit`, and `np.dsplit`.

- `np.array_split:` Splits an array into N parts. Returns a list of sub-arrays​. It can handle cases where the array cannot be evenly divided (it will make smaller last sub-array as needed)​

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(newarr)
# [array([1, 2]), array([3, 4]), array([5, 6])]
```

You can then access each piece:

```python
print(newarr[0], newarr[1], newarr[2])  
# array([1,2]) array([3,4]) array([5,6])
```

- `np.split:` Similar to array_split but requires the array to split evenly; otherwise it raises an error. (Use array_split for flexibility)​.

**2-D splitting:** By default, `np.array_split` splits along the first axis (rows). You can specify an axis:

```python
arr2 = np.arange(1, 19).reshape(6, 3)  # 6×3 array
splits = np.array_split(arr2, 3)  # split into 3 blocks of 2 rows each
print(splits[0])  # first block 2×3 array

# Split columns instead (axis=1):
col_splits = np.array_split(arr2, 3, axis=1)
print(col_splits[0])  # first block (all 6 rows, columns 0-0)
```

You can also use helper functions:

- `np.hsplit(arr, 3)`: split into 3 parts along columns (like axis=1)​.
- `np.vsplit(arr, 3)`: split into 3 parts along rows (like axis=0).
- `np.dsplit(arr, 3)`: split along the third axis for 3-D arrays.

## 🔹 Array Search

NumPy provides methods to search arrays:

- `np.where(condition)`: Returns the indices where the condition is `True​`. For example:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 4, 4])
idx = np.where(arr == 4)
print(idx)  # (array([3, 5, 6]),) – indices of value 4

# Find even values:
even_idx = np.where(arr % 2 == 0)
print(even_idx)  # indices of even elements
```

- `np.searchsorted(sorted_arr, values, side='left' or 'right')`: Finds indices where elements should be inserted to maintain order​. This is a binary search on a sorted array.

```python
sorted_arr = np.array([6, 7, 8, 9])
print(np.searchsorted(sorted_arr, 7))             # 1 (leftmost insertion index)
print(np.searchsorted(sorted_arr, 7, side='right'))  # 2 (rightmost insertion index)

# Multiple values:
arr = np.array([1, 3, 5, 7])
positions = np.searchsorted(arr, [2, 4, 6])
print(positions)  # [1 2 3] (insert 2,4,6)
```

## 🔹 Array Sort

To sort an array, use `np.sort(arr)`. This returns a sorted copy, leaving the original unchanged.

```python
import numpy as np

arr = np.array([3, 2, 0, 1])
sorted_arr = np.sort(arr)
print(sorted_arr)  # [0 1 2 3]

# Sort different types:
print(np.sort(np.array(['banana', 'cherry', 'apple'])))
# ['apple' 'banana' 'cherry']

print(np.sort(np.array([True, False, True])))
# [False  True  True] (False < True)
```

For **2-D arrays**, `np.sort(arr)` sorts each row (i.e. each sub-array) individually​:

```python
arr2 = np.array([[3, 2, 4],
                 [5, 0, 1]])
print(np.sort(arr2))
# [[2 3 4]
#  [0 1 5]]  (each row sorted)
```

## 🔹 Array Filter

Filtering means selecting elements based on a condition, producing a new array. In NumPy, you use a **boolean mask** (boolean array) to filter. Each `True` in the mask means "keep the element at this position", and `False` means "discard it".

Example – using a hard-coded mask:

```python
import numpy as np

arr = np.array([41, 42, 43, 44])
mask = [True, False, True, False]
newarr = arr[mask]
print(newarr)  # [41 43] – keeps only indices 0 and 2
```

More commonly, you create the mask from a condition:

```python
arr = np.array([41, 42, 43, 44])
mask = arr > 42   # array([False, False,  True,  True])
print(arr[mask])  # [43 44] (values greater than 42)

arr = np.array([1, 2, 3, 4, 5, 6, 7])
even_mask = (arr % 2 == 0)
print(arr[even_mask])  # [2 4 6] (even numbers)

# You can do it in one line:
print(arr[arr % 2 == 0])  # same result
```

Filtering creates a new array of only the elements where the condition is true. This technique is very powerful for selecting data in NumPy.
