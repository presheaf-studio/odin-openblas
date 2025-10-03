package openblas

import "base:builtin"
import "base:intrinsics"
import "core:math"
import "core:slice"

// ============================================================================
// SYMMETRIC MATRIX UTILITIES AND TYPE CONVERSIONS
// ============================================================================
// Utilities for working with symmetric matrices in full storage format
// Symmetric matrices have the property: A[i,j] = A[j,i]

// Check if a matrix is symmetric (within tolerance)
is_symmetric :: proc(A: ^Matrix($T), tolerance: T) -> bool where is_float(T) || is_complex(T) {
	if A.rows != A.cols {
		return false
	}

	n := A.rows
	for i in 0 ..< n {
		for j in i + 1 ..< n {
			a_ij := matrix_get(A, i, j)
			a_ji := matrix_get(A, j, i)

			when is_complex(T) {
				// For complex matrices, check if A[j,i] = conj(A[i,j]) for Hermitian
				// or A[j,i] = A[i,j] for symmetric
				diff := abs(a_ij - a_ji)
			} else {
				diff := abs(a_ij - a_ji)
			}

			if diff > tolerance {
				return false
			}
		}
	}
	return true
}

// Check if a matrix is Hermitian (complex version of symmetric)
is_hermitian :: proc(A: ^Matrix($T), tolerance: T) -> bool where is_complex(T) {
	if A.rows != A.cols {
		return false
	}

	n := A.rows
	tolerance_real := real(tolerance)

	for i in 0 ..< n {
		// Diagonal elements must be real for Hermitian matrices
		a_ii := matrix_get(A, i, i)
		if abs(imag(a_ii)) > tolerance_real {
			return false
		}

		for j in i + 1 ..< n {
			a_ij := matrix_get(A, i, j)
			a_ji := matrix_get(A, j, i)

			// For Hermitian: A[j,i] = conj(A[i,j])
			diff := abs(a_ji - conj(a_ij))
			if real(diff) > tolerance_real || imag(diff) > tolerance_real {
				return false
			}
		}
	}
	return true
}

// Convert a general matrix to symmetric format by averaging upper and lower triangles
make_symmetric :: proc(A: ^Matrix($T), uplo := MatrixRegion.Upper) where is_float(T) {
	assert(A.rows == A.cols, "Matrix must be square")

	n := A.rows
	for i in 0 ..< n {
		for j in i + 1 ..< n {
			a_ij := matrix_get(A, i, j)
			a_ji := matrix_get(A, j, i)
			avg := (a_ij + a_ji) / 2

			if uplo == .Upper {
				matrix_set(A, i, j, avg)
				matrix_set(A, j, i, avg)
			} else {
				matrix_set(A, j, i, avg)
				matrix_set(A, i, j, avg)
			}
		}
	}
}

// Convert a general matrix to Hermitian format
make_hermitian :: proc(A: ^Matrix($T), uplo := MatrixRegion.Upper) where is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")

	n := A.rows
	for i in 0 ..< n {
		// Make diagonal elements real
		a_ii := matrix_get(A, i, i)
		matrix_set(A, i, i, complex(real(a_ii), 0))

		for j in i + 1 ..< n {
			a_ij := matrix_get(A, i, j)
			a_ji := matrix_get(A, j, i)

			if uplo == .Upper {
				// Keep upper triangle, set lower as conjugate transpose
				matrix_set(A, j, i, conj(a_ij))
			} else {
				// Keep lower triangle, set upper as conjugate transpose
				matrix_set(A, i, j, conj(a_ji))
			}
		}
	}
}

// Copy only the specified triangle of a symmetric matrix
copy_symmetric_triangle :: proc(src: ^Matrix($T), dst: ^Matrix(T), uplo := MatrixRegion.Upper) {
	assert(src.rows == src.cols && dst.rows == dst.cols, "Matrices must be square")
	assert(src.rows == dst.rows, "Matrix dimensions must match")

	n := src.rows

	// Copy diagonal
	for i in 0 ..< n {
		matrix_set(dst, i, i, matrix_get(src, i, i))
	}

	// Copy specified triangle
	if uplo == .Upper {
		for i in 0 ..< n {
			for j in i + 1 ..< n {
				val := matrix_get(src, i, j)
				matrix_set(dst, i, j, val)
				matrix_set(dst, j, i, val) // Symmetry
			}
		}
	} else {
		for i in 0 ..< n {
			for j in 0 ..< i {
				val := matrix_get(src, i, j)
				matrix_set(dst, i, j, val)
				matrix_set(dst, j, i, val) // Symmetry
			}
		}
	}
}

// Copy only the specified triangle of a Hermitian matrix
copy_hermitian_triangle :: proc(src: ^Matrix($T), dst: ^Matrix(T), uplo := MatrixRegion.Upper) where is_complex(T) {
	assert(src.rows == src.cols && dst.rows == dst.cols, "Matrices must be square")
	assert(src.rows == dst.rows, "Matrix dimensions must match")

	n := src.rows

	// Copy diagonal (should be real for Hermitian)
	for i in 0 ..< n {
		val := matrix_get(src, i, i)
		matrix_set(dst, i, i, complex(real(val), 0)) // Ensure real diagonal
	}

	// Copy specified triangle
	if uplo == .Upper {
		for i in 0 ..< n {
			for j in i + 1 ..< n {
				val := matrix_get(src, i, j)
				matrix_set(dst, i, j, val)
				matrix_set(dst, j, i, conj(val)) // Hermitian property
			}
		}
	} else {
		for i in 0 ..< n {
			for j in 0 ..< i {
				val := matrix_get(src, i, j)
				matrix_set(dst, i, j, val)
				matrix_set(dst, j, i, conj(val)) // Hermitian property
			}
		}
	}
}

// Convert full storage symmetric matrix to packed storage
symmetric_to_packed :: proc(A: ^Matrix($T), uplo := MatrixRegion.Upper, allocator := context.allocator) -> PackedMatrix(T) where is_float(T) || is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")

	n := A.rows
	pm := make_packed_matrix(n, uplo, T, allocator)
	pm.symmetric = true

	if uplo == .Upper {
		for j in 0 ..< n {
			for i in 0 ..= j {
				idx := i + j * (j + 1) / 2
				pm.data[idx] = matrix_get(A, i, j)
			}
		}
	} else {
		for j in 0 ..< n {
			for i in j ..< n {
				idx := i + (2 * n - j - 1) * j / 2
				pm.data[idx] = matrix_get(A, i, j)
			}
		}
	}

	return pm
}

// Convert packed storage to full storage symmetric matrix
packed_to_symmetric :: proc(pm: ^PackedMatrix($T), allocator := context.allocator) -> Matrix(T) where is_float(T) || is_complex(T) {
	A := make_matrix(pm.n, pm.n, T, allocator)

	if pm.uplo == .Upper {
		for j in 0 ..< pm.n {
			for i in 0 ..= j {
				idx := i + j * (j + 1) / 2
				val := pm.data[idx]
				matrix_set(&A, i, j, val)
				if i != j && pm.symmetric {
					when is_complex(T) {
						if pm.symmetric {
							matrix_set(&A, j, i, val) // Symmetric
						} else {
							matrix_set(&A, j, i, conj(val)) // Hermitian
						}
					} else {
						matrix_set(&A, j, i, val) // Symmetric
					}
				}
			}
		}
	} else {
		for j in 0 ..< pm.n {
			for i in j ..< pm.n {
				idx := i + (2 * pm.n - j - 1) * j / 2
				val := pm.data[idx]
				matrix_set(&A, i, j, val)
				if i != j && pm.symmetric {
					when is_complex(T) {
						if pm.symmetric {
							matrix_set(&A, j, i, val) // Symmetric
						} else {
							matrix_set(&A, j, i, conj(val)) // Hermitian
						}
					} else {
						matrix_set(&A, j, i, val) // Symmetric
					}
				}
			}
		}
	}

	return A
}

// Transpose a symmetric matrix in-place (no-op, but useful for API consistency)
transpose_symmetric :: proc(A: ^Matrix($T)) where is_float(T) || is_complex(T) {
	// For symmetric matrices, A^T = A, so this is a no-op
	// This function exists for API consistency
}

// Conjugate transpose (Hermitian transpose) a Hermitian matrix in-place (no-op)
hermitian_transpose :: proc(A: ^Matrix($T)) where is_complex(T) {
	// For Hermitian matrices, A^H = A, so this is a no-op
	// This function exists for API consistency
}

// Scale a symmetric matrix by a scalar (preserves symmetry)
scale_symmetric :: proc(A: ^Matrix($T), alpha: T) where is_float(T) || is_complex(T) {
	// For symmetric matrices, scaling preserves symmetry
	matrix_scale(A, alpha)
}

// Scale a Hermitian matrix by a real scalar (preserves Hermitian property)
scale_hermitian :: proc(A: ^Matrix($Cmplx), alpha: $Real) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	// For Hermitian matrices, scaling by real number preserves Hermitian property
	alpha_complex := complex(alpha, 0)
	matrix_scale(A, alpha_complex)
}


// ============================================================================
// MATRIX PROPERTY QUERIES
// ============================================================================

// Estimate the condition number of a symmetric matrix using 1-norm
// Note: This is a rough estimate, not as accurate as expert drivers
estimate_symmetric_condition :: proc(A: ^Matrix($T), uplo := MatrixRegion.Upper) -> T where is_float(T) {
	// Simple estimate using Frobenius norm ratio
	// For accurate condition numbers, use expert drivers

	n := A.rows
	max_diag: T = 0
	min_diag: T = max(T)

	for i in 0 ..< n {
		val := abs(matrix_get(A, i, i))
		max_diag = max(max_diag, val)
		min_diag = min(min_diag, val)
	}

	if min_diag == 0 {
		return max(T) // Infinite condition number
	}

	return max_diag / min_diag
}

// Check if a symmetric matrix is positive definite (rough check using diagonal)
// Note: This is not a rigorous check - use Cholesky factorization for certainty
is_likely_positive_definite :: proc(A: ^Matrix($T)) -> bool where is_float(T) || is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")

	n := A.rows
	for i in 0 ..< n {
		val := matrix_get(A, i, i)
		when is_complex(T) {
			if real(val) <= 0 {
				return false
			}
		} else {
			if val <= 0 {
				return false
			}
		}
	}
	return true
}
