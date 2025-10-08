package openblas

import "base:builtin"
import "core:mem"

// ===================================================================================
// BANDED MATRIX TYPE DEFINITIONS AND BASIC UTILITIES
//
// This file provides the foundation for all banded matrix operations in OpenBLAS/LAPACK.
// Banded matrices are efficient for storing matrices with non-zero elements concentrated
// around the main diagonal.
//
// Banded Matrix Formats:
// - GB: General banded (general rectangular banded matrices)
// - SB: Symmetric banded (real symmetric banded matrices)
// - HB: Hermitian banded (complex Hermitian banded matrices)
// - PB: Positive definite banded (Cholesky-factorizable banded matrices)
// - TB: Triangular banded (upper/lower triangular banded matrices)
//
// Storage Format:
// - Band storage format stores only the diagonals within the band
// - For general banded: stores kl subdiagonals + main diagonal + ku superdiagonals
// - For symmetric/Hermitian: stores only upper or lower triangle bands
// - Leading dimension ldab ≥ kl + ku + 1 for general, ldab ≥ kd + 1 for symmetric
// ===================================================================================

// ===================================================================================
// BANDED MATRIX TYPE DEFINITION
// ===================================================================================

// Banded matrix storage - efficient for sparse banded systems
BandedMatrix :: struct($T: typeid) {
	data:      []T, // Band storage array, size ldab × cols
	rows:      Blas_Int, // Number of rows
	cols:      Blas_Int, // Number of columns
	kl:        Blas_Int, // Number of subdiagonals (lower bandwidth)
	ku:        Blas_Int, // Number of superdiagonals (upper bandwidth)
	ldab:      Blas_Int, // Leading dimension of band storage (>= kl + ku + 1)

	// For symmetric banded matrices
	symmetric: bool, // True if symmetric/Hermitian (only need ku or kl)
}

// ===================================================================================
// BANDED MATRIX CREATION
// ===================================================================================

// Create a general banded matrix
band_make :: proc(rows, cols, kl, ku: int, $T: typeid, allocator := context.allocator) -> BandedMatrix(T) {
	ldab := kl + ku + 1
	return BandedMatrix(T){rows = Blas_Int(rows), cols = Blas_Int(cols), kl = Blas_Int(kl), ku = Blas_Int(ku), ldab = Blas_Int(ldab), data = make([]T, ldab * cols, allocator)}
}

// Create symmetric banded matrix (only sets upper or lower band)
band_make_symmetric :: proc(n, k: int, uplo: UpLo, $T: typeid, allocator := context.allocator) -> BandedMatrix(T) {
	ldab := k + 1 // For symmetric, only need k+1 bands
	bm := BandedMatrix(T) {
		rows      = Blas_Int(n),
		cols      = Blas_Int(n),
		symmetric = true,
		ldab      = Blas_Int(ldab),
		data      = make([]T, ldab * n, allocator),
	}

	if uplo == .Upper {
		bm.ku = Blas_Int(k)
		bm.kl = 0
	} else {
		bm.kl = Blas_Int(k)
		bm.ku = 0
	}

	return bm
}

// Create a triangular banded matrix
band_make_triangular :: proc(
	n: int,
	k: int, // bandwidth
	uplo: UpLo,
	diag: Diag,
	$T: typeid,
	allocator := context.allocator,
) -> BandedMatrix(T) {
	ldab := k + 1
	data_size := ldab * n

	// For triangular banded, set appropriate kl/ku
	kl, ku: Blas_Int
	if uplo == .Upper {
		kl, ku = 0, Blas_Int(k)
	} else {
		kl, ku = Blas_Int(k), 0
	}

	return BandedMatrix(T){data = make([]T, data_size, allocator), rows = Blas_Int(n), cols = Blas_Int(n), kl = kl, ku = ku, ldab = Blas_Int(ldab)}
}

// ===================================================================================
// BANDED MATRIX INDEXING AND ACCESS
// ===================================================================================

// Access element (i,j) in banded storage
// Band storage format (column-major):
// AB(ku+1+i-j, j) = A(i,j) for max(0,j-ku) <= i <= min(m-1,j+kl)
band_index :: proc(bm: ^BandedMatrix($T), i, j: int) -> (idx: int, stored: bool) {
	assert(i >= 0 && i < int(bm.rows) && j >= 0 && j < int(bm.cols), "Index out of bounds")

	// Check if element is within the band
	if i > j + int(bm.kl) || i < j - int(bm.ku) {
		return -1, false
	}

	// Column-major band storage
	idx = int(bm.ku) + i - j + j * int(bm.ldab)
	return idx, true
}

// Get element (i,j) from banded matrix
band_get :: proc(bm: ^BandedMatrix($T), i, j: int) -> (value: T, stored: bool) {
	idx, is_stored := band_index(bm, i, j)
	if !is_stored {
		return T{}, false
	}
	return bm.data[idx], true
}

// Set element (i,j) in banded matrix
band_set :: proc(bm: ^BandedMatrix($T), i, j: int, value: T) -> bool {
	idx, is_stored := band_index(bm, i, j)
	if !is_stored {
		return false
	}
	bm.data[idx] = value
	return true
}

// ===================================================================================
// BANDED MATRIX PROPERTIES AND VALIDATION
// ===================================================================================

// Get banded storage parameters from matrix
band_get_params :: proc(bm: ^BandedMatrix($T)) -> (kl, ku, ldab: int) {
	return int(bm.kl), int(bm.ku), int(bm.ldab)
}

// Check if matrix has valid banded structure
band_validate :: proc(bm: ^BandedMatrix($T)) -> bool {
	if bm.ldab < bm.kl + bm.ku + 1 {
		return false
	}
	if len(bm.data) < int(bm.ldab) * int(bm.cols) {
		return false
	}
	return true
}

// Get the bandwidth of a banded matrix
band_get_bandwidth :: proc(bm: ^BandedMatrix($T)) -> (lower: int, upper: int) {
	return int(bm.kl), int(bm.ku) // TODO: flag to compute the bandwidth?? have in auxillery iirc
}

// ===================================================================================
// MEMORY MANAGEMENT
// ===================================================================================

// Delete banded matrix data
band_delete :: proc(bm: ^BandedMatrix($T), allocator := context.allocator) {
	delete(bm.data, allocator)
	bm.data = nil
}

// Clone a banded matrix
band_clone :: proc(bm: ^BandedMatrix($T), allocator := context.allocator) -> BandedMatrix(T) {
	clone := BandedMatrix(T) {
		rows      = bm.rows,
		cols      = bm.cols,
		kl        = bm.kl,
		ku        = bm.ku,
		ldab      = bm.ldab,
		symmetric = bm.symmetric,
		data      = make([]T, len(bm.data), allocator),
	}
	copy(clone.data, bm.data)
	return clone
}

// // ===================================================================================
// // WORKSPACE AND SIZE QUERIES
// // ===================================================================================
// // FIXME: no cstring; do we even need this?? have specific queries for ops..
// // Query workspace requirements for general banded operations
// query_workspace_band :: proc(operation: string, $T: typeid, n, kl, ku: int) -> (work_size, rwork_size, iwork_size: int) {
// 	// Default workspace requirements for common operations
// 	switch operation {
// 	case "factor":
// 		// LU factorization
// 		work_size = 0
// 		rwork_size = 0
// 		iwork_size = 0
// 	case "solve":
// 		// Linear solve
// 		work_size = 0
// 		rwork_size = 0
// 		iwork_size = 0
// 	case "condition":
// 		// Condition number estimation
// 		when is_float(T) {
// 			work_size = 3 * n
// 			iwork_size = n
// 		} else {
// 			work_size = 2 * n
// 			rwork_size = n
// 		}
// 	case "equilibrate":
// 		// matrix equilibration
// 		work_size = 0
// 		rwork_size = 0
// 		iwork_size = 0
// 	case:
// 		// Unknown operation, return minimal workspace
// 		work_size = n
// 		rwork_size = 0
// 		iwork_size = 0
// 	}

// 	return
// }

// // Allocate workspace for banded operations
// // FIXME: No cstrings.. do we even need?
// allocate_band_workspace :: proc(operation: string, $T: typeid, n, kl, ku: int, allocator := context.allocator) -> (work: []T, rwork: []f64, iwork: []Blas_Int) {
// 	work_size, rwork_size, iwork_size := query_workspace_band(operation, T, n, kl, ku)

// 	if work_size > 0 {
// 		work = make([]T, work_size, allocator)
// 	}
// 	if rwork_size > 0 {
// 		rwork = make([]f64, rwork_size, allocator)
// 	}
// 	if iwork_size > 0 {
// 		iwork = make([]Blas_Int, iwork_size, allocator)
// 	}

// 	return
// }
