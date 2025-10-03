package openblas

import "base:builtin"
import "core:math"
import "core:slice"

// ===================================================================================
// SPECIALIZED MATRIX STORAGE TYPES - REORGANIZED
// ===================================================================================
//
// This file now contains only shared types and utilities that don't belong to
// a specific matrix format. Specialized matrix types have been moved to their
// own files for better organization:
//
// - BandedMatrix -> banded.odin + banded_conversion.odin
// - Tridiagonal -> tridiagonal.odin + tridiagonal_conversion.odin
// - PackedSymmetric -> packed_symmetric.odin + packed_symmetric_conversion.odin
// - PackedHermitian -> packed_hermitian.odin + packed_hermitian_conversion.odin
// - PackedTriangular -> packed_triangular.odin + packed_triangular_conversion.odin
// - RFP -> rfp.odin + rfp_conversion.odin
//
// Standard dense matrix (Matrix) is defined in matrix.odin
// ===================================================================================

// ===================================================================================
// LEGACY PACKED MATRIX TYPE - TO BE DEPRECATED
// ===================================================================================
// NOTE: This generic PackedMatrix type is deprecated in favor of the specific
// PackedSymmetric, PackedHermitian, and PackedTriangular types in their respective files.
// It's kept here temporarily for compatibility but should not be used in new code.

// Generic packed matrix type (deprecated - use specific types instead)
PackedMatrix :: struct($T: typeid) where is_float(T) || is_complex(T) {
	data:      []T, // Packed storage array, size n*(n+1)/2
	n:         int, // Matrix dimension (n×n)
	uplo:      UpLo, // Upper or Lower triangle stored
	// For symmetric/Hermitian matrices
	symmetric: bool, // True if matrix is symmetric (real) or Hermitian (complex)
}

// Create a packed matrix from dimensions (deprecated)
make_packed_matrix :: proc(n: int, uplo: UpLo, $T: typeid, allocator := context.allocator) -> PackedMatrix(T) {
	size := n * (n + 1) / 2
	return PackedMatrix(T){n = n, data = make([]T, size, allocator), uplo = uplo}
}

// Access element (i,j) in packed storage (deprecated)
packed_index :: proc(pm: ^PackedMatrix($T), i, j: int) -> int {
	assert(i >= 0 && i < pm.n && j >= 0 && j < pm.n, "Index out of bounds")

	if pm.uplo == .Upper {
		if i <= j {
			return i + j * (j + 1) / 2
		} else if pm.symmetric {
			return j + i * (i + 1) / 2 // Use symmetry
		}
	} else { 	// Lower
		if i >= j {
			return i + (2 * pm.n - j - 1) * j / 2
		} else if pm.symmetric {
			return j + (2 * pm.n - i - 1) * i / 2 // Use symmetry
		}
	}

	panic("Accessing unstored element in packed matrix")
}

// ===================================================================================
// RFP (RECTANGULAR FULL PACKED) FORMAT - MOVED TO rfp.odin
// ===================================================================================

// RFP type has been moved to rfp.odin + rfp_conversion.odin for better organization.
// Use the RFP type and associated functions defined there.

// ===================================================================================
// PROPERTY ENUMS (shared across matrix types)
// ===================================================================================

// Triangle specification
// ===================================================================================
// SHARED PROPERTY ENUMS
// ===================================================================================

// Triangle specification
UpLo :: enum u8 {
	Upper = 'U', // Upper triangle
	Lower = 'L', // Lower triangle
}

// Diagonal type for triangular matrices
Diag :: enum u8 {
	NonUnit = 'N', // Diagonal elements are stored
	Unit    = 'U', // Diagonal elements are assumed to be 1
}

// Side for operations like triangular solve
Side :: enum u8 {
	Left  = 'L', // Operation on the left
	Right = 'R', // Operation on the right
}

// Transpose operations
TransposeState :: enum u8 {
	NoTrans   = 'N', // No transpose
	Trans     = 'T', // Transpose
	ConjTrans = 'C', // Conjugate transpose (Hermitian transpose)
}

// ===================================================================================
// LEGACY CONVERSION UTILITIES - TO BE DEPRECATED
// ===================================================================================
// NOTE: These generic conversion functions are deprecated. Use the specific
// conversion functions in the appropriate *_conversion.odin files instead.

// Convert packed matrix to full matrix (deprecated)
packed_to_full :: proc(pm: ^PackedMatrix($T), allocator := context.allocator) -> Matrix(T) {
	// Create a standard matrix - assuming column-major for LAPACK compatibility
	m := Matrix(T) {
		rows   = Blas_Int(pm.n),
		cols   = Blas_Int(pm.n),
		data   = make([]T, pm.n * pm.n, allocator),
		ld     = Blas_Int(pm.n),
		format = .General,
	}

	for j in 0 ..< pm.n {
		for i in 0 ..< pm.n {
			if (pm.uplo == .Upper && i <= j) || (pm.uplo == .Lower && i >= j) {
				idx := packed_index(pm, i, j)
				// Column-major indexing
				m.data[i + j * pm.n] = pm.data[idx]
				if pm.symmetric && i != j {
					m.data[j + i * pm.n] = pm.data[idx] // Fill symmetric element
				}
			}
		}
	}

	return m
}

// Convert full matrix to packed format (deprecated)
full_to_packed :: proc(m: ^Matrix($T), uplo: UpLo, check_symmetric := false, allocator := context.allocator) -> PackedMatrix(T) {
	assert(m.rows == m.cols, "Matrix must be square for packed storage")

	pm := make_packed_matrix(int(m.rows), uplo, T, allocator)

	if check_symmetric {
		pm.symmetric = is_symmetric(m)
	}

	for j in 0 ..< m.cols {
		for i in 0 ..< m.rows {
			if (uplo == .Upper && i <= j) || (uplo == .Lower && i >= j) {
				idx := packed_index(&pm, i, j)
				// Column-major indexing
				pm.data[idx] = m.data[i + j * m.cols]
			}
		}
	}

	return pm
}

// ===================================================================================
// UTILITY FUNCTIONS FOR MATRIX TYPE ENUMS
// ===================================================================================

// Convert UpLo enum to cstring for LAPACK calls
uplo_to_cstring :: proc(uplo: UpLo) -> cstring {
	switch uplo {
	case .Upper:
		return "U"
	case .Lower:
		return "L"
	}
	return "U" // Default to upper
}
