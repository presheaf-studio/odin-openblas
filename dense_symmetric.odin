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
dns_is_symmetric :: proc(A: ^Matrix($T), tolerance: T) -> bool where is_float(T) || is_complex(T) {
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
dns_is_hermitian :: proc(A: ^Matrix($T), tolerance: T) -> bool where is_complex(T) {
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
dns_make_symmetric :: proc(A: ^Matrix($T), uplo := MatrixRegion.Upper) where is_float(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	// FIXME: should we really be averaging here??
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
dns_make_hermitian :: proc(A: ^Matrix($T), uplo := MatrixRegion.Upper) where is_complex(T) {
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
dns_copy_symmetric_triangle :: proc(src: ^Matrix($T), dst: ^Matrix(T), uplo := MatrixRegion.Upper) {
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
dns_copy_hermitian_triangle :: proc(src: ^Matrix($T), dst: ^Matrix(T), uplo := MatrixRegion.Upper) where is_complex(T) {
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

// Scale a symmetric matrix by a scalar (preserves symmetry)
dns_scale_symmetric :: proc(A: ^Matrix($T), alpha: T) where is_float(T) || is_complex(T) {
	// For symmetric matrices, scaling preserves symmetry
	matrix_scale(A, alpha)
}

// Scale a Hermitian matrix by a real scalar (preserves Hermitian property)
dns_scale_hermitian :: proc(A: ^Matrix($Cmplx), alpha: $Real) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	// For Hermitian matrices, scaling by real number preserves Hermitian property
	alpha_complex := complex(alpha, 0)
	matrix_scale(A, alpha_complex)
}


// ============================================================================
// MATRIX PROPERTY QUERIES
// ============================================================================

// Check if a symmetric matrix is positive definite (rough check using diagonal)
// Note: This is not a rigorous check - use Cholesky factorization for certainty
dns_is_likely_positive_definite :: proc(A: ^Matrix($T)) -> bool where is_float(T) || is_complex(T) {
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
