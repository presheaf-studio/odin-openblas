package openblas

import lapack "./f77"
import "core:math"

// ===================================================================================
// PACKED SYMMETRIC MATRIX CONVERSIONS
// ===================================================================================
//
// This file provides conversion operations between packed symmetric matrices
// and other matrix formats. Packed storage stores only the upper or lower triangle
// of symmetric matrices in a 1D array, saving approximately 50% memory.
//
// Conversion operations:
// - pack_symmetric: Convert full matrix to packed storage
// - unpack_symmetric: Convert packed storage to full matrix
// - create_packed_symmetric: Allocating conversion from full to PackedSymmetric
// - extract_full_symmetric: Allocating conversion from PackedSymmetric to full

// ===================================================================================
// MEMORY LAYOUT CONVERSIONS
// ===================================================================================

// Convert full symmetric matrix to packed storage
// Non-allocating version: requires pre-allocated packed array
pack_symmetric :: proc(
	A: []$T, // Full matrix (n×n) in column-major order
	AP: []T, // Pre-allocated packed array (n*(n+1)/2 elements)
	n: int, // Matrix dimension
	lda: int, // Leading dimension of A
	uplo: MatrixRegion = .Upper,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(A) >= n * lda, "Full matrix array too small")

	idx := 0
	switch uplo {
	case .Upper:
		// Pack upper triangle column by column
		for j in 0 ..< n {
			for i in 0 ..= j {
				AP[idx] = A[i + j * lda]
				idx += 1
			}
		}
	case .Lower:
		// Pack lower triangle column by column
		for j in 0 ..< n {
			for i in j ..< n {
				AP[idx] = A[i + j * lda]
				idx += 1
			}
		}
	case .Full:
		panic("Full storage not supported for packed format")
	}
}

// Convert packed storage to full symmetric matrix
// Non-allocating version: requires pre-allocated full matrix
unpack_symmetric :: proc(
	AP: []$T, // Packed array
	A: []T, // Pre-allocated full matrix (n×n)
	n: int, // Matrix dimension
	lda: int, // Leading dimension of A
	uplo: MatrixRegion = .Upper,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(A) >= n * lda, "Full matrix array too small")

	// Initialize full matrix to zero
	for i in 0 ..< n * lda {
		A[i] = T(0)
	}

	idx := 0
	switch uplo {
	case .Upper:
		// Unpack upper triangle and mirror to lower
		for j in 0 ..< n {
			for i in 0 ..= j {
				val := AP[idx]
				A[i + j * lda] = val
				if i != j {
					A[j + i * lda] = val // Mirror to lower triangle
				}
				idx += 1
			}
		}
	case .Lower:
		// Unpack lower triangle and mirror to upper
		for j in 0 ..< n {
			for i in j ..< n {
				val := AP[idx]
				A[i + j * lda] = val
				if i != j {
					A[j + i * lda] = val // Mirror to upper triangle
				}
				idx += 1
			}
		}
	case .Full:
		panic("Full storage not supported for packed format")
	}
}

// ===================================================================================
// ALLOCATING CONVERSION FUNCTIONS
// ===================================================================================

// Create PackedSymmetric from full matrix (allocating version for convenience)
create_packed_symmetric :: proc(
	A: []$T, // Full matrix
	n: int, // Matrix dimension
	lda: int, // Leading dimension of A
	uplo: MatrixRegion = .Upper,
	allocator := context.allocator,
) -> PackedSymmetric(T) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	packed_size := packed_storage_size(n)
	data := make([]T, packed_size, allocator)

	pack_symmetric(A, data, n, lda, uplo)

	return PackedSymmetric(T){data = data, n = n, uplo = uplo}
}

// Extract full matrix from PackedSymmetric (allocating version for convenience)
extract_full_symmetric :: proc(
	packed: ^PackedSymmetric($T),
	lda: int, // Leading dimension for output
	allocator := context.allocator,
) -> []T where T == f32 || T == f64 || T == complex64 || T == complex128 {
	if lda < packed.n {
		panic("Leading dimension too small")
	}

	A := make([]T, packed.n * lda, allocator)
	unpack_symmetric(packed.data, A, packed.n, lda, packed.uplo)

	return A
}

// ===================================================================================
// BATCH CONVERSION OPERATIONS
// ===================================================================================

// Convert multiple packed matrices to full format in batch
batch_extract_packed_symmetric :: proc(packed_matrices: []^PackedSymmetric($T), lda: int, allocator := context.allocator) -> [][]T {
	results := make([][]T, len(packed_matrices), allocator)

	for i, packed in packed_matrices {
		results[i] = extract_full_symmetric(packed, lda, allocator)
	}

	return results
}

// Convert multiple full matrices to packed format in batch
batch_create_packed_symmetric :: proc(full_matrices: [][]$T, n: int, lda: int, uplo: MatrixRegion = .Upper, allocator := context.allocator) -> []PackedSymmetric(T) {
	results := make([]PackedSymmetric(T), len(full_matrices), allocator)

	for i, A in full_matrices {
		results[i] = create_packed_symmetric(A, n, lda, uplo, allocator)
	}

	return results
}

// ===================================================================================
// IN-PLACE CONVERSION UTILITIES
// ===================================================================================

// Convert upper triangle storage to lower triangle storage (or vice versa)
// This operation modifies the packed matrix in-place
transpose_packed_storage :: proc(packed: ^PackedSymmetric($T)) {
	n := packed.n
	temp_data := make([]T, len(packed.data))
	defer delete(temp_data)

	if packed.uplo == .Upper {
		// Convert upper to lower storage
		idx := 0
		for j in 0 ..< n {
			for i in j ..< n {
				// Get element (i,j) from upper storage
				upper_idx := j + i * (i + 1) / 2
				temp_data[idx] = packed.data[upper_idx]
				idx += 1
			}
		}
		packed.uplo = .Lower
	} else {
		// Convert lower to upper storage
		idx := 0
		for j in 0 ..< n {
			for i in 0 ..= j {
				// Get element (i,j) from lower storage
				lower_idx := (j - i) + i * (2 * n - i - 1) / 2
				temp_data[idx] = packed.data[lower_idx]
				idx += 1
			}
		}
		packed.uplo = .Upper
	}

	copy(packed.data, temp_data)
}

// ===================================================================================
// CONVERSION VALIDATION
// ===================================================================================

// Verify that a packed matrix correctly represents a symmetric matrix
// by converting to full and checking symmetry
verify_packed_symmetry :: proc(packed: ^PackedSymmetric($T), tolerance: T) -> bool {
	// Extract to full matrix
	A := extract_full_symmetric(packed, packed.n)
	defer delete(A)

	// Check symmetry: A[i,j] == A[j,i]
	for i in 0 ..< packed.n {
		for j in 0 ..< packed.n {
			diff := A[i + j * packed.n] - A[j + i * packed.n]
			when T == complex64 || T == complex128 {
				if abs(diff) > tolerance {
					return false
				}
			} else {
				if abs(diff) > tolerance {
					return false
				}
			}
		}
	}

	return true
}

// Compare packed and full representations for equivalence
compare_packed_full :: proc(packed: ^PackedSymmetric($T), A: []T, lda: int, tolerance: T) -> bool {
	if lda < packed.n {
		return false
	}

	// Check each element
	for i in 0 ..< packed.n {
		for j in 0 ..< packed.n {
			packed_val := packed_symmetric_get(packed.data, packed.n, i, j, packed.uplo)
			full_val := A[i + j * lda]

			diff := packed_val - full_val
			when T == complex64 || T == complex128 {
				if abs(diff) > tolerance {
					return false
				}
			} else {
				if abs(diff) > tolerance {
					return false
				}
			}
		}
	}

	return true
}
