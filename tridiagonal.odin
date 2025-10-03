package openblas

import "base:builtin"
import "core:mem"

// ===================================================================================
// TRIDIAGONAL MATRIX TYPE DEFINITIONS AND BASIC UTILITIES
//
// This file provides the foundation for all tridiagonal matrix operations.
// Tridiagonal matrices have non-zero elements only on the main diagonal and the
// diagonals immediately above and below it.
//
// Tridiagonal Matrix Formats:
// - GT: General tridiagonal (three diagonals: sub, main, super)
// - ST: Symmetric tridiagonal (symmetric: super = sub for real, super = conj(sub) for complex)
// - PT: Positive definite tridiagonal (special properties for Cholesky)
//
// Storage Format:
// - Three arrays: dl (subdiagonal), d (diagonal), du (superdiagonal)
// - For symmetric/Hermitian: du may share storage with dl (real) or be computed from dl (complex)
// ===================================================================================

// ===================================================================================
// TRIDIAGONAL MATRIX TYPE DEFINITION
// ===================================================================================

// Tridiagonal matrix - special case of banded with kl=ku=1
Tridiagonal :: struct($T: typeid) where is_float(T) || is_complex(T) {
	n:         Blas_Int, // Matrix dimension (n×n)
	dl:        []T, // Subdiagonal, size n-1
	d:         []T, // Main diagonal, size n
	du:        []T, // Superdiagonal, size n-1

	// For symmetric tridiagonal (du = dl for real, du = conj(dl) for complex)
	symmetric: bool,
}

// ===================================================================================
// TRIDIAGONAL MATRIX CREATION
// ===================================================================================

// Create a general tridiagonal matrix
make_tridiagonal :: proc(n: int, $T: typeid, allocator := context.allocator) -> Tridiagonal(T) {
	return Tridiagonal(T){n = Blas_Int(n), dl = make([]T, max(n - 1, 0), allocator), d = make([]T, n, allocator), du = make([]T, max(n - 1, 0), allocator)}
}

// Create symmetric tridiagonal matrix (du is not allocated for real, or computed from dl for complex)
make_symmetric_tridiagonal :: proc(n: int, $T: typeid, allocator := context.allocator) -> Tridiagonal(T) {
	tm := Tridiagonal(T) {
		n         = Blas_Int(n),
		d         = make([]T, n, allocator),
		dl        = make([]T, max(n - 1, 0), allocator),
		symmetric = true,
	}

	when is_float(T) {
		// For real symmetric, du shares storage with dl
		tm.du = tm.dl
	} else {
		// For complex Hermitian, du = conj(dl) (need separate storage)
		tm.du = make([]T, max(n - 1, 0), allocator)
	}

	return tm
}

// ===================================================================================
// TRIDIAGONAL MATRIX INDEXING AND ACCESS
// ===================================================================================

// Get element (i,j) from tridiagonal matrix
tridiagonal_get :: proc(tm: ^Tridiagonal($T), i, j: int) -> (value: T, stored: bool) {
	assert(i >= 0 && i < int(tm.n) && j >= 0 && j < int(tm.n), "Index out of bounds")

	if i == j {
		return tm.d[i], true
	} else if i == j + 1 && i > 0 {
		return tm.dl[j], true
	} else if i == j - 1 && j > 0 {
		when T == complex64 || T == complex128 {
			if tm.symmetric {
				return conj(tm.dl[i]), true // Hermitian case
			}
		}
		return tm.du[i], true
	}

	return T{}, false
}

// Set element (i,j) in tridiagonal matrix
tridiagonal_set :: proc(tm: ^Tridiagonal($T), i, j: int, value: T) -> bool {
	assert(i >= 0 && i < int(tm.n) && j >= 0 && j < int(tm.n), "Index out of bounds")

	if i == j {
		tm.d[i] = value
		return true
	} else if i == j + 1 && i > 0 {
		tm.dl[j] = value
		if tm.symmetric {
			when is_float(T) {
				// Real symmetric: du = dl
			} else {
				// Complex Hermitian: du = conj(dl)
				tm.du[j] = conj(value)
			}
		}
		return true
	} else if i == j - 1 && j > 0 {
		tm.du[i] = value
		if tm.symmetric {
			when is_float(T) {
				tm.dl[i] = value // Real symmetric
			} else {
				tm.dl[i] = conj(value) // Complex Hermitian
			}
		}
		return true
	}

	return false // Element not in tridiagonal structure
}

// ===================================================================================
// TRIDIAGONAL MATRIX PROPERTIES AND VALIDATION
// ===================================================================================

// Check if matrix has valid tridiagonal structure
validate_tridiagonal :: proc(tm: ^Tridiagonal($T)) -> bool {
	if tm.n <= 0 {
		return false
	}
	if len(tm.d) != int(tm.n) {
		return false
	}
	if tm.n > 1 && (len(tm.dl) != int(tm.n - 1) || len(tm.du) != int(tm.n - 1)) {
		return false
	}
	return true
}

// Check if tridiagonal matrix is diagonally dominant
is_diagonally_dominant :: proc(tm: ^Tridiagonal($T)) -> bool {
	n := int(tm.n)

	// First row
	if n > 1 {
		when is_float(T) {
			if abs(tm.d[0]) < abs(tm.du[0]) {
				return false
			}
		} else {
			if abs(tm.d[0]) < abs(tm.du[0]) {
				return false
			}
		}
	}

	// Middle rows
	for i in 1 ..< n - 1 {
		when is_float(T) {
			if abs(tm.d[i]) < abs(tm.dl[i - 1]) + abs(tm.du[i]) {
				return false
			}
		} else {
			if abs(tm.d[i]) < abs(tm.dl[i - 1]) + abs(tm.du[i]) {
				return false
			}
		}
	}

	// Last row
	if n > 1 {
		when is_float(T) {
			if abs(tm.d[n - 1]) < abs(tm.dl[n - 2]) {
				return false
			}
		} else {
			if abs(tm.d[n - 1]) < abs(tm.dl[n - 2]) {
				return false
			}
		}
	}

	return true
}

// ===================================================================================
// MEMORY MANAGEMENT
// ===================================================================================

// Delete tridiagonal matrix data
delete_tridiagonal :: proc(tm: ^Tridiagonal($T), allocator := context.allocator) {
	delete(tm.d, allocator)
	delete(tm.dl, allocator)
	// Only delete du if it's not sharing storage with dl
	when is_complex(T) {
		if tm.symmetric {
			delete(tm.du, allocator)
		}
	} else {
		if !tm.symmetric {
			delete(tm.du, allocator)
		}
	}
	tm.d = nil
	tm.dl = nil
	tm.du = nil
}

// Clone a tridiagonal matrix
clone_tridiagonal :: proc(tm: ^Tridiagonal($T), allocator := context.allocator) -> Tridiagonal(T) {
	clone := Tridiagonal(T) {
		n         = tm.n,
		symmetric = tm.symmetric,
		d         = make([]T, len(tm.d), allocator),
		dl        = make([]T, len(tm.dl), allocator),
	}

	copy(clone.d, tm.d)
	copy(clone.dl, tm.dl)

	// Handle du based on symmetry
	if tm.symmetric {
		when is_float(T) {
			// Share storage for real symmetric
			clone.du = clone.dl
		} else {
			// Allocate and copy for complex Hermitian
			clone.du = make([]T, len(tm.du), allocator)
			copy(clone.du, tm.du)
		}
	} else {
		// Allocate and copy for general tridiagonal
		clone.du = make([]T, len(tm.du), allocator)
		copy(clone.du, tm.du)
	}

	return clone
}

// ===================================================================================
// WORKSPACE AND SIZE QUERIES
// ===================================================================================

// Query workspace requirements for tridiagonal operations
query_workspace_tridiagonal :: proc(operation: string, $T: typeid, n: int) -> (work_size, rwork_size, iwork_size: int) {
	// Default workspace requirements for common operations
	switch operation {
	case "factor":
		// LU factorization for tridiagonal
		work_size = 0
		rwork_size = 0
		iwork_size = 0
	case "solve":
		// Linear solve for tridiagonal
		work_size = 0
		rwork_size = 0
		iwork_size = 0
	case "eigenvalues":
		// Eigenvalue computation
		when is_float(T) {
			work_size = max(1, 2 * n - 2)
		} else {
			work_size = max(1, n)
			rwork_size = max(1, n)
		}
	case "eigenvectors":
		// Eigenvector computation
		when is_float(T) {
			work_size = max(1, 2 * n - 2)
		} else {
			work_size = max(1, 2 * n)
			rwork_size = max(1, 3 * n - 2)
		}
	case:
		// Unknown operation, return minimal workspace
		work_size = n
		rwork_size = 0
		iwork_size = 0
	}

	return
}

// Allocate workspace for tridiagonal operations
allocate_tridiagonal_workspace :: proc(operation: string, $T: typeid, n: int, allocator := context.allocator) -> (work: []T, rwork: []f64, iwork: []Blas_Int) {
	work_size, rwork_size, iwork_size := query_workspace_tridiagonal(operation, T, n)

	if work_size > 0 {
		work = make([]T, work_size, allocator)
	}
	if rwork_size > 0 {
		rwork = make([]f64, rwork_size, allocator)
	}
	if iwork_size > 0 {
		iwork = make([]Blas_Int, iwork_size, allocator)
	}

	return
}
