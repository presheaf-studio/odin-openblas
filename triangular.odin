package openblas

import lapack "./f77"
import "base:intrinsics"

// ===================================================================================
// TRIANGULAR MATRIX OPERATIONS
// ===================================================================================
//
// This file provides the base types and utilities for full-storage triangular matrices.
// Triangular matrices store the full n×n matrix but use only the upper or lower triangle.
//
// Related files:
// - packed_triangular.odin - Packed storage triangular matrices (TP prefix functions)
// - triangular_linear.odin - Linear solvers and inversion (TRTRS, TRTRI)
// - triangular_operations.odin - Condition numbers and operations (TRCON)
// - Triangular banded functions are in banded_*.odin files (TB prefix)

// ===================================================================================
// TRIANGULAR MATRIX TYPE
// ===================================================================================

// Triangular matrix with full storage
Triangular :: struct($T: typeid) {
	data: []T, // Matrix data in column-major order
	n:    int, // Matrix dimension (n×n)
	lda:  int, // Leading dimension (>= n)
	uplo: MatrixRegion, // Upper or lower triangular
	diag: DiagonalType, // Unit or non-unit diagonal
}


// ===================================================================================
// VALIDATION AND UTILITIES
// ===================================================================================

// Validate triangular matrix dimensions and storage
validate_triangular :: proc(n, lda: int, data_len: int) -> bool {
	return n >= 0 && lda >= max(1, n) && data_len >= n * lda
}

// Get the storage requirement for a triangular matrix
triangular_storage_size :: proc(n, lda: int) -> int {
	return n * lda
}

// Create triangular matrix view (non-allocating)
make_triangular :: proc(data: []$T, n, lda: int, uplo: MatrixRegion = .Upper, diag: DiagonalType = .NonUnit) -> (tri: Triangular(T), ok: bool) {
	if !validate_triangular(n, lda, len(data)) {
		return {}, false
	}

	return Triangular(T){data = data, n = n, lda = lda, uplo = uplo, diag = diag}, true
}

// Access element (i,j) of triangular matrix
triangular_get :: proc(tri: ^Triangular($T), i, j: int) -> T {
	assert(i >= 0 && i < tri.n && j >= 0 && j < tri.n, "Index out of bounds")

	// Check if element is in the stored triangle
	switch tri.uplo {
	case .Upper:
		if i <= j {
			return tri.data[i + j * tri.lda]
		} else if tri.diag == .Unit && i == j {
			return T(1)
		} else {
			return T(0)
		}
	case .Lower:
		if i >= j {
			return tri.data[i + j * tri.lda]
		} else if tri.diag == .Unit && i == j {
			return T(1)
		} else {
			return T(0)
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}
	return T(0)
}

// Set element (i,j) of triangular matrix
triangular_set :: proc(tri: ^Triangular($T), i, j: int, value: T) {
	assert(i >= 0 && i < tri.n && j >= 0 && j < tri.n, "Index out of bounds")

	// Only allow setting elements in the stored triangle
	switch tri.uplo {
	case .Upper:
		if i <= j && !(tri.diag == .Unit && i == j) {
			tri.data[i + j * tri.lda] = value
		} else if tri.diag == .Unit && i == j {
			panic("Cannot set diagonal element of unit triangular matrix")
		} else {
			panic("Cannot set element outside stored triangle")
		}
	case .Lower:
		if i >= j && !(tri.diag == .Unit && i == j) {
			tri.data[i + j * tri.lda] = value
		} else if tri.diag == .Unit && i == j {
			panic("Cannot set diagonal element of unit triangular matrix")
		} else {
			panic("Cannot set element outside stored triangle")
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}
}

// Copy triangular matrix to full matrix (non-allocating)
triangular_to_full :: proc(tri: ^Triangular($T), A: []T, lda_full: int) {
	assert(len(A) >= tri.n * lda_full, "Full matrix array too small")
	assert(lda_full >= tri.n, "Leading dimension too small")

	// Initialize to zero
	for i in 0 ..< tri.n {
		for j in 0 ..< tri.n {
			A[i + j * lda_full] = T(0)
		}
	}

	// Copy stored triangle
	switch tri.uplo {
	case .Upper:
		for j in 0 ..< tri.n {
			for i in 0 ..= j {
				if tri.diag == .Unit && i == j {
					A[i + j * lda_full] = T(1)
				} else {
					A[i + j * lda_full] = tri.data[i + j * tri.lda]
				}
			}
		}
	case .Lower:
		for j in 0 ..< tri.n {
			for i in j ..< tri.n {
				if tri.diag == .Unit && i == j {
					A[i + j * lda_full] = T(1)
				} else {
					A[i + j * lda_full] = tri.data[i + j * tri.lda]
				}
			}
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}
}

// Copy full matrix to triangular (extracting the relevant triangle)
full_to_triangular :: proc(A: []$T, lda_full: int, tri: ^Triangular(T)) {
	assert(len(A) >= tri.n * lda_full, "Full matrix array too small")
	assert(lda_full >= tri.n, "Leading dimension too small")

	switch tri.uplo {
	case .Upper:
		for j in 0 ..< tri.n {
			for i in 0 ..= j {
				if !(tri.diag == .Unit && i == j) {
					tri.data[i + j * tri.lda] = A[i + j * lda_full]
				}
			}
		}
	case .Lower:
		for j in 0 ..< tri.n {
			for i in j ..< tri.n {
				if !(tri.diag == .Unit && i == j) {
					tri.data[i + j * tri.lda] = A[i + j * lda_full]
				}
			}
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}
}

// Check if triangular matrix is well-conditioned (rough estimate)
is_well_conditioned_triangular_rough :: proc(tri: ^Triangular($T), threshold: T) -> bool where is_float(T) {
	// Simple check: examine diagonal elements
	if tri.diag == .Unit {
		return true // Unit triangular matrices are always well-conditioned
	}

	min_diag := T(max(T))
	max_diag := T(0)

	for i in 0 ..< tri.n {
		diag_val := abs(tri.data[i + i * tri.lda])
		min_diag = min(min_diag, diag_val)
		max_diag = max(max_diag, diag_val)
	}

	if min_diag == 0 {
		return false // Singular matrix
	}

	// Rough condition number estimate: max_diag / min_diag
	return (max_diag / min_diag) < (T(1) / threshold)
}

// Create identity triangular matrix (non-allocating)
make_identity_triangular :: proc(data: []$T, n, lda: int, uplo: MatrixRegion = .Upper) -> (tri: Triangular(T), ok: bool) {
	if !validate_triangular(n, lda, len(data)) {
		return {}, false
	}

	// Initialize to zero
	for i in 0 ..< n * lda {
		data[i] = T(0)
	}

	// Set diagonal to 1
	for i in 0 ..< n {
		data[i + i * lda] = T(1)
	}

	return make_triangular(data, n, lda, uplo, .NonUnit)
}

// Compute frobenius norm of triangular matrix
frobenius_norm_triangular :: proc {
	frobenius_norm_triangular_real,
	frobenius_norm_triangular_complex,
}

frobenius_norm_triangular_real :: proc(tri: ^Triangular($T)) -> T where is_float(T) {
	sum := T(0)

	switch tri.uplo {
	case .Upper:
		for j in 0 ..< tri.n {
			for i in 0 ..= j {
				if tri.diag == .Unit && i == j {
					sum += T(1) // Unit diagonal
				} else {
					val := tri.data[i + j * tri.lda]
					sum += val * val
				}
			}
		}
	case .Lower:
		for j in 0 ..< tri.n {
			for i in j ..< tri.n {
				if tri.diag == .Unit && i == j {
					sum += T(1) // Unit diagonal
				} else {
					val := tri.data[i + j * tri.lda]
					sum += val * val
				}
			}
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}

	return sqrt(sum)
}

frobenius_norm_triangular_complex :: proc(tri: ^Triangular($Cmplx)) -> $Real where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	sum := Real(0)

	switch tri.uplo {
	case .Upper:
		for j in 0 ..< tri.n {
			for i in 0 ..= j {
				if tri.diag == .Unit && i == j {
					sum += Real(1) // Unit diagonal
				} else {
					val := tri.data[i + j * tri.lda]
					sum += real(val * conj(val))
				}
			}
		}
	case .Lower:
		for j in 0 ..< tri.n {
			for i in j ..< tri.n {
				if tri.diag == .Unit && i == j {
					sum += Real(1) // Unit diagonal
				} else {
					val := tri.data[i + j * tri.lda]
					sum += real(val * conj(val))
				}
			}
		}
	case .Full:
		panic("Full storage not supported for triangular matrices")
	}

	return sqrt(sum)
}
