package openblas

import lapack "./f77"
import "core:math"

// ===================================================================================
// PACKED TRIANGULAR MATRIX CONVERSIONS
// ===================================================================================
//
// This file provides conversion operations between packed triangular matrices
// and other matrix formats. Packed storage stores only the upper or lower triangle
// of triangular matrices in a 1D array, requiring n*(n+1)/2 elements instead of n²
// for an n×n matrix.
//
// Conversion operations:
// - pack_triangular: Convert full matrix to packed storage
// - unpack_triangular: Convert packed storage to full matrix
// - create_packed_triangular: Allocating conversion from full to PackedTriangular
// - extract_full_triangular: Allocating conversion from PackedTriangular to full
// - convert_packed_to_full_triangular: LAPACK conversion to full format
// - convert_packed_to_rfp_triangular: LAPACK conversion to RFP format

// ===================================================================================
// MEMORY LAYOUT CONVERSIONS
// ===================================================================================

// Convert full triangular matrix to packed storage
// Non-allocating version: requires pre-allocated packed array
pack_triangular :: proc(
	A: []$T, // Full matrix (n×n) in column-major order
	AP: []T, // Pre-allocated packed array (n*(n+1)/2 elements)
	n: int, // Matrix dimension
	lda: int, // Leading dimension of A
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
) {
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

// Extract triangular matrix from packed format to full matrix
// Non-allocating version: requires pre-allocated full matrix
unpack_triangular :: proc(
	AP: []$T, // Packed array
	A: []T, // Pre-allocated full matrix (n×n)
	n: int, // Matrix dimension
	lda: int, // Leading dimension of A
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
	diag: DiagonalType = .NonUnit, // Diagonal type
) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(A) >= n * lda, "Full matrix array too small")

	// Initialize full matrix to zero
	for i in 0 ..< n * lda {
		A[i] = T(0)
	}

	idx := 0
	switch uplo {
	case .Upper:
		// Unpack upper triangle
		for j in 0 ..< n {
			for i in 0 ..= j {
				if i == j && diag == .Unit {
					A[i + j * lda] = T(1) // Unit diagonal
				} else {
					A[i + j * lda] = AP[idx]
				}
				if !(i == j && diag == .Unit) {
					idx += 1
				}
			}
		}
	case .Lower:
		// Unpack lower triangle
		for j in 0 ..< n {
			for i in j ..< n {
				if i == j && diag == .Unit {
					A[i + j * lda] = T(1) // Unit diagonal
				} else {
					A[i + j * lda] = AP[idx]
				}
				if !(i == j && diag == .Unit) {
					idx += 1
				}
			}
		}
	case .Full:
		panic("Full storage not supported for packed format")
	}
}

// ===================================================================================
// ALLOCATING CONVERSION FUNCTIONS
// ===================================================================================

// Create PackedTriangular from full matrix (allocating version for convenience)
create_packed_triangular :: proc(
	A: []$T, // Full matrix
	n: int, // Matrix dimension
	lda: int, // Leading dimension of A
	uplo: MatrixRegion = .Upper,
	diag: DiagonalType = .NonUnit,
	allocator := context.allocator,
) -> PackedTriangular(T) {
	packed_size := packed_storage_size(n)
	data := make([]T, packed_size, allocator)

	pack_triangular(A, data, n, lda, uplo)

	return PackedTriangular(T){data = data, n = n, uplo = uplo, diag = diag}
}

// Extract full matrix from PackedTriangular (allocating version for convenience)
extract_full_triangular :: proc(
	packed: ^PackedTriangular($T),
	lda: int, // Leading dimension for output
	allocator := context.allocator,
) -> []T {
	if lda < packed.n {
		panic("Leading dimension too small")
	}

	A := make([]T, packed.n * lda, allocator)
	unpack_triangular(packed.data, A, packed.n, lda, packed.uplo, packed.diag)

	return A
}

// ===================================================================================
// LAPACK FORMAT CONVERSION ROUTINES
// ===================================================================================

// Convert packed triangular to full triangular format
convert_packed_to_full_triangular :: proc(
	AP: []$T, // Triangular packed matrix
	A: []T, // Pre-allocated full matrix [n×n]
	n: int, // Matrix dimension
	lda: int, // Leading dimension of A
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")
	assert(len(A) >= n * lda, "Full matrix array too small")
	assert(lda >= n, "Leading dimension too small")

	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)
	lda_blas := Blas_Int(lda)

	when T == f32 {
		lapack.stpttr_(&uplo_c, &n_blas, raw_data(AP), raw_data(A), &lda_blas, &info)
	} else when T == f64 {
		lapack.dtpttr_(&uplo_c, &n_blas, raw_data(AP), raw_data(A), &lda_blas, &info)
	} else when T == complex64 {
		lapack.ctpttr_(&uplo_c, &n_blas, raw_data(AP), raw_data(A), &lda_blas, &info)
	} else when T == complex128 {
		lapack.ztpttr_(&uplo_c, &n_blas, raw_data(AP), raw_data(A), &lda_blas, &info)
	}

	ok = (info == 0)
	return info, ok
}

// Convert packed triangular to RFP (Rectangular Full Packed) format
convert_packed_to_rfp_triangular :: proc(
	AP: []$T, // Triangular packed matrix
	ARF: []T, // Pre-allocated RFP array
	n: int, // Matrix dimension
	transr: RFPTranspose = .NORMAL, // RFP transpose option
	uplo: MatrixRegion = .Upper, // Upper or lower triangular
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_packed_storage(n, len(AP)), "Packed array too small")

	// RFP requires n*(n+1)/2 elements but in different layout
	expected_arf_size := n * (n + 1) / 2
	assert(len(ARF) >= expected_arf_size, "RFP array too small")

	transr_c := u8(transr)
	uplo_c := u8(uplo)
	n_blas := Blas_Int(n)

	when T == f32 {
		lapack.stpttf_(&transr_c, &uplo_c, &n_blas, raw_data(AP), raw_data(ARF), &info)
	} else when T == f64 {
		lapack.dtpttf_(&transr_c, &uplo_c, &n_blas, raw_data(AP), raw_data(ARF), &info)
	} else when T == complex64 {
		lapack.ctpttf_(&transr_c, &uplo_c, &n_blas, raw_data(AP), raw_data(ARF), &info)
	} else when T == complex128 {
		lapack.ztpttf_(&transr_c, &uplo_c, &n_blas, raw_data(AP), raw_data(ARF), &info)
	}

	ok = (info == 0)
	return info, ok
}

// ===================================================================================
// BATCH CONVERSION OPERATIONS
// ===================================================================================

// Convert multiple packed matrices to full format in batch
batch_extract_packed_triangular :: proc(packed_matrices: []^PackedTriangular($T), lda: int, allocator := context.allocator) -> [][]T {
	results := make([][]T, len(packed_matrices), allocator)

	for i, packed in packed_matrices {
		results[i] = extract_full_triangular(packed, lda, allocator)
	}

	return results
}

// Convert multiple full matrices to packed format in batch
batch_create_packed_triangular :: proc(full_matrices: [][]$T, n: int, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit, allocator := context.allocator) -> []PackedTriangular(T) {
	results := make([]PackedTriangular(T), len(full_matrices), allocator)

	for i, A in full_matrices {
		results[i] = create_packed_triangular(A, n, lda, uplo, diag, allocator)
	}

	return results
}

// ===================================================================================
// IN-PLACE CONVERSION UTILITIES
// ===================================================================================

// Convert upper triangle storage to lower triangle storage (or vice versa)
// This operation modifies the packed matrix in-place
transpose_packed_storage :: proc(packed: ^PackedTriangular($T)) {
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

// Verify that a packed matrix correctly represents a triangular matrix
// by converting to full and checking triangular property
verify_packed_triangular :: proc(packed: ^PackedTriangular($T), tolerance: T) -> bool {
	// Extract to full matrix
	A := extract_full_triangular(packed, packed.n)
	defer delete(A)

	// Check triangular property
	for i in 0 ..< packed.n {
		for j in 0 ..< packed.n {
			element := A[i + j * packed.n]

			if packed.uplo == .Upper && i > j {
				// Upper triangular: below diagonal should be zero
				when is_complex(T) {
					if abs(element) > tolerance {
						return false
					}
				} else {
					if abs(element) > tolerance {
						return false
					}
				}
			} else if packed.uplo == .Lower && i < j {
				// Lower triangular: above diagonal should be zero
				when is_complex(T) {
					if abs(element) > tolerance {
						return false
					}
				} else {
					if abs(element) > tolerance {
						return false
					}
				}
			}
		}
	}

	return true
}

// Compare packed and full representations for equivalence
compare_packed_full :: proc(packed: ^PackedTriangular($T), A: []T, lda: int, tolerance: T) -> bool {
	if lda < packed.n {
		return false
	}

	// Check each element in the triangular region
	for i in 0 ..< packed.n {
		for j in 0 ..< packed.n {
			// Only check elements in the stored triangle
			should_check := false
			if packed.uplo == .Upper && i <= j {
				should_check = true
			} else if packed.uplo == .Lower && i >= j {
				should_check = true
			}

			if should_check {
				packed_val := packed_triangular_get(packed.data, packed.n, i, j, packed.uplo)
				full_val := A[i + j * lda]

				diff := packed_val - full_val
				when is_complex(T) {
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
	}

	return true
}

// ===================================================================================
// ELEMENT ACCESS UTILITIES
// ===================================================================================

// Get element from packed triangular matrix
packed_triangular_get :: proc(data: []$T, n: int, i, j: int, uplo: MatrixRegion) -> T {
	assert(i >= 0 && i < n && j >= 0 && j < n, "Index out of bounds")

	switch uplo {
	case .Upper:
		if i <= j {
			idx := i + j * (j + 1) / 2
			return data[idx]
		} else {
			return T(0) // Below diagonal is zero
		}
	case .Lower:
		if i >= j {
			idx := i + (2 * n - j - 1) * j / 2
			return data[idx]
		} else {
			return T(0) // Above diagonal is zero
		}
	case .Full:
		panic("Full storage not supported for packed format")
	}
}

// Set element in packed triangular matrix
packed_triangular_set :: proc(data: []$T, n: int, i, j: int, value: T, uplo: MatrixRegion) {
	assert(i >= 0 && i < n && j >= 0 && j < n, "Index out of bounds")

	switch uplo {
	case .Upper:
		if i <= j {
			idx := i + j * (j + 1) / 2
			data[idx] = value
		} else {
			panic("Cannot set element below diagonal in upper triangular matrix")
		}
	case .Lower:
		if i >= j {
			idx := i + (2 * n - j - 1) * j / 2
			data[idx] = value
		} else {
			panic("Cannot set element above diagonal in lower triangular matrix")
		}
	case .Full:
		panic("Full storage not supported for packed format")
	}
}
