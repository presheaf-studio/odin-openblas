package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// RFP (RECTANGULAR FULL PACKED) MATRIX TYPE DEFINITIONS AND BASIC UTILITIES
//
// This file provides the foundation for all RFP matrix operations.
// RFP (Rectangular Full Packed) format is a hybrid storage scheme that combines
// the benefits of packed and full storage formats, providing better cache performance
// than traditional packed storage while maintaining memory efficiency.
//
// RFP Format Details:
// - Stores only the upper or lower triangle of symmetric/Hermitian matrices
// - Uses a rectangular array layout for better cache performance
// - Supports both normal and transposed storage variants
// - Storage mapping depends on matrix dimension parity and transpose state
//
// References:
// - LAPACK Working Note 199: "The RFP Format for Packed Symmetric Matrices"
// - LAPACK Users' Guide, Section on RFP routines
// ===================================================================================

// ===================================================================================
// RFP MATRIX TYPE DEFINITION
// ===================================================================================

// RFP matrix storage - hybrid format for symmetric/Hermitian matrices
RFP :: struct($T: typeid) {
	// where is_float(T) || is_complex(T)
	data:        []T, // RFP storage array, size n*(n+1)/2
	n:           Blas_Int, // Matrix dimension (n×n)
	trans_state: TransposeState, // RFP format variant (Normal or Transpose/Conjugate)
	uplo:        UpLo, // Upper or Lower triangle of original matrix
}

// ===================================================================================
// RFP MATRIX CREATION
// ===================================================================================

// Create an RFP matrix
make_rfp :: proc(n: int, trans_state: TransposeState, uplo: UpLo, $T: typeid, allocator := context.allocator) -> RFP(T) {
	size := n * (n + 1) / 2
	return RFP(T){n = Blas_Int(n), data = make([]T, size, allocator), trans_state = trans_state, uplo = uplo}
}

// ===================================================================================
// RFP MATRIX INDEXING AND ACCESS
// ===================================================================================

// Access element (i,j) in RFP storage
// RFP storage is complex - the mapping depends on:
// - n even or odd
// - transr = 'N' or 'T'/'C'
// - uplo = 'U' or 'L'
// This creates 8 different storage schemes
rfp_index :: proc(rm: ^RFP($T), i, j: int) -> (idx: int, stored: bool) {
	assert(i >= 0 && i < int(rm.n) && j >= 0 && j < int(rm.n), "Index out of bounds")
	// FIXME
	n := int(rm.n)
	k := n / 2

	// RFP format is quite complex, this is a simplified version
	// Full implementation would need all 8 cases
	// See LAPACK working note 199 for complete details

	if rm.trans_state == .NoTrans {
		if rm.uplo == .Upper {
			// Normal, Upper case
			if n % 2 == 0 {
				// Even n, Normal, Upper
				// Complex indexing formula would go here
				// For now, return placeholder
				return -1, false
			} else {
				// Odd n, Normal, Upper
				// Complex indexing formula would go here
				return -1, false
			}
		} else {
			// Normal, Lower case
			// Similar complex logic
			return -1, false
		}
	} else {
		// Transposed cases
		// More complex logic
		return -1, false
	}

	// Placeholder - would need full implementation
	return -1, false
}

// Get element (i,j) from RFP matrix
rfp_get :: proc(rm: ^RFP($T), i, j: int) -> (value: T, stored: bool) {
	idx, is_stored := rfp_index(rm, i, j)
	if !is_stored {
		return T{}, false
	}
	return rm.data[idx], true
}

// Set element (i,j) in RFP matrix
rfp_set :: proc(rm: ^RFP($T), i, j: int, value: T) -> bool {
	idx, is_stored := rfp_index(rm, i, j)
	if !is_stored {
		return false
	}
	rm.data[idx] = value
	return true
}

// ===================================================================================
// RFP MATRIX PROPERTIES AND VALIDATION
// ===================================================================================

// Check if RFP matrix has valid structure
validate_rfp :: proc(rm: ^RFP($T)) -> bool {
	if rm.n <= 0 {
		return false
	}
	expected_size := int(rm.n) * (int(rm.n) + 1) / 2
	if len(rm.data) < expected_size {
		return false
	}
	return true
}

// Get RFP storage size for a given matrix dimension
rfp_storage_size :: proc(n: int) -> int {
	return n * (n + 1) / 2
}

// Check if RFP parameters are compatible
rfp_compatible :: proc(rm1: ^RFP($T), rm2: ^RFP(T)) -> bool {
	return rm1.n == rm2.n && rm1.trans_state == rm2.trans_state && rm1.uplo == rm2.uplo
}

// ===================================================================================
// MEMORY MANAGEMENT
// ===================================================================================

// Delete RFP matrix data
delete_rfp :: proc(rm: ^RFP($T), allocator := context.allocator) {
	delete(rm.data, allocator)
	rm.data = nil
}

// Clone an RFP matrix
clone_rfp :: proc(rm: ^RFP($T), allocator := context.allocator) -> RFP(T) {
	clone := RFP(T) {
		n           = rm.n,
		trans_state = rm.trans_state,
		uplo        = rm.uplo,
		data        = make([]T, len(rm.data), allocator),
	}
	copy(clone.data, rm.data)
	return clone
}


// ===================================================================================
// RFP FORMAT UTILITIES
// ===================================================================================

// Determine optimal RFP format for given matrix dimension
optimal_rfp_format :: proc(n: int) -> (trans_state: TransposeState, uplo: UpLo) {
	// Heuristics for optimal RFP format selection
	// These are simplified rules - actual optimal choice depends on access patterns

	if n % 2 == 0 {
		// Even dimension - normal format often performs better
		return .NoTrans, .Upper
	} else {
		// Odd dimension - transposed format may be better for some operations
		return .Trans, .Upper
	}
}

// ===================================================================================
// RFP CHOLESKY FACTORIZATION (PFTRF family)
// ===================================================================================

// Compute Cholesky factorization of a positive definite matrix in RFP format
// A = U**T * U or A = L * L**T (real)
// A = U**H * U or A = L * L**H (complex)
rfp_cholesky_factorize :: proc(
	A: ^RFP($T), // RFP matrix (input/output, overwritten with factor)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	n := A.n

	when T == f32 {
		lapack.spftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == f64 {
		lapack.dpftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == complex64 {
		lapack.cpftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == complex128 {
		lapack.zpftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
	}

	return info, info == 0
}

// ===================================================================================
// RFP CHOLESKY SOLVE (PFTRS family)
// ===================================================================================

// Solve linear system using RFP Cholesky factorization: A*X = B
rfp_cholesky_solve :: proc(
	A: ^RFP($T), // Cholesky factors in RFP format
	B: ^Matrix(T), // RHS matrix (input/output, overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")
	assert(B.rows == A.n, "B matrix must have same number of rows as A dimension")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	n := A.n
	nrhs := B.cols
	ldb := B.ld

	when T == f32 {
		lapack.spftrs_(&transr, &uplo, &n, &nrhs, raw_data(A.data), raw_data(B.data), &ldb, &info)
	} else when T == f64 {
		lapack.dpftrs_(&transr, &uplo, &n, &nrhs, raw_data(A.data), raw_data(B.data), &ldb, &info)
	} else when T == complex64 {
		lapack.cpftrs_(&transr, &uplo, &n, &nrhs, raw_data(A.data), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zpftrs_(&transr, &uplo, &n, &nrhs, raw_data(A.data), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// RFP CHOLESKY INVERSION (PFTRI family)
// ===================================================================================

// Compute inverse of positive definite matrix using RFP Cholesky factorization
rfp_cholesky_invert :: proc(
	A: ^RFP($T), // Cholesky factors (input/output, overwritten with inverse)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	n := A.n

	when T == f32 {
		lapack.spftri_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == f64 {
		lapack.dpftri_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == complex64 {
		lapack.cpftri_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == complex128 {
		lapack.zpftri_(&transr, &uplo, &n, raw_data(A.data), &info)
	}

	return info, info == 0
}

// ===================================================================================
// RFP TRIANGULAR SOLVE (TFSM family)
// ===================================================================================

// Solve triangular system with RFP format matrix
// Solves: op(A) * X = alpha * B  (side = Left)
//     or: X * op(A) = alpha * B  (side = Right)
// where A is stored in RFP format
rfp_triangular_solve :: proc(
	A: ^RFP($T), // Triangular matrix in RFP format
	B: ^Matrix(T), // Right-hand side matrix (solution on output)
	alpha: T, // Scalar multiplier
	side: Side = .Left,
	trans: TransposeState = .NoTrans,
	diag: Diag = .NonUnit,
) -> (
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	m := B.rows
	n := B.cols

	// Determine the size of A based on side
	a_size := side == .Left ? m : n
	assert(A.n == a_size, "RFP matrix dimension incompatible with B")

	transr := cast(u8)A.trans_state
	side_c := cast(u8)side
	uplo := cast(u8)A.uplo
	trans_c := cast(u8)trans
	diag_c := cast(u8)diag
	m_int := m
	n_int := n
	ldb := B.ld
	alpha := alpha

	when T == f32 {
		lapack.stfsm_(&transr, &side_c, &uplo, &trans_c, &diag_c, &m_int, &n_int, &alpha, raw_data(A.data), raw_data(B.data), &ldb)
	} else when T == f64 {
		lapack.dtfsm_(&transr, &side_c, &uplo, &trans_c, &diag_c, &m_int, &n_int, &alpha, raw_data(A.data), raw_data(B.data), &ldb)
	} else when T == complex64 {
		lapack.ctfsm_(&transr, &side_c, &uplo, &trans_c, &diag_c, &m_int, &n_int, &alpha, raw_data(A.data), raw_data(B.data), &ldb)
	} else when T == complex128 {
		lapack.ztfsm_(&transr, &side_c, &uplo, &trans_c, &diag_c, &m_int, &n_int, &alpha, raw_data(A.data), raw_data(B.data), &ldb)
	}

	return true
}

// ===================================================================================
// RFP TRIANGULAR INVERSION (TFTRI family)
// ===================================================================================

// Invert triangular matrix in RFP format
// Computes the inverse of a triangular matrix A stored in RFP format
rfp_triangular_invert :: proc(
	A: ^RFP($T), // Triangular matrix in RFP format (inverted in place)
	diag: Diag = .NonUnit,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) || is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	diag_c := cast(u8)diag
	n := A.n

	when T == f32 {
		lapack.stftri_(&transr, &uplo, &diag_c, &n, raw_data(A.data), &info)
	} else when T == f64 {
		lapack.dtftri_(&transr, &uplo, &diag_c, &n, raw_data(A.data), &info)
	} else when T == complex64 {
		lapack.ctftri_(&transr, &uplo, &diag_c, &n, raw_data(A.data), &info)
	} else when T == complex128 {
		lapack.ztftri_(&transr, &uplo, &diag_c, &n, raw_data(A.data), &info)
	}

	return info, info == 0
}

// ===================================================================================
// RFP SYMMETRIC RANK-K UPDATE (SFRK family - real only)
// ===================================================================================

// Symmetric rank-k update in RFP format: C := alpha*A*A^T + beta*C
rfp_symmetric_rank_k_update :: proc(
	C: ^RFP($T), // Symmetric matrix in RFP format (output)
	A: ^Matrix(T), // Input matrix
	alpha: T,
	beta: T,
	trans: TransposeState = .NoTrans,
) where is_float(T) {
	assert(validate_rfp(C), "Invalid RFP matrix structure")

	n := C.n
	k := (trans == .NoTrans) ? A.cols : A.rows
	lda := A.ld

	transr := cast(u8)C.trans_state
	uplo := cast(u8)C.uplo
	trans_c := cast(u8)trans

	alpha := alpha
	beta := beta

	when T == f32 {
		lapack.ssfrk_(&transr, &uplo, &trans_c, &n, &k, &alpha, raw_data(A.data), &lda, &beta, raw_data(C.data))
	} else when T == f64 {
		lapack.dsfrk_(&transr, &uplo, &trans_c, &n, &k, &alpha, raw_data(A.data), &lda, &beta, raw_data(C.data))
	}
}

// ===================================================================================
// RFP HERMITIAN RANK-K UPDATE (HFRK family - complex only)
// ===================================================================================


// Hermitian rank-k update in RFP format: C := alpha*A*A^H + beta*C
rfp_hermitian_rank_k_update :: proc(
	C: ^RFP($Cmplx), // Hermitian matrix in RFP format (output)
	A: ^Matrix(Cmplx), // Input matrix
	alpha: $Real,
	beta: Real,
	trans: TransposeState = .NoTrans,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	assert(validate_rfp(C), "Invalid RFP matrix structure")

	n := C.n
	k := (trans == .NoTrans) ? A.cols : A.rows
	lda := A.ld

	transr := cast(u8)C.trans_state
	uplo := cast(u8)C.uplo
	trans_c := cast(u8)trans

	alpha := alpha
	beta := beta

	when Cmplx == complex64 {
		lapack.chfrk_(&transr, &uplo, &trans_c, &n, &k, &alpha, raw_data(A.data), &lda, &beta, raw_data(C.data))
	} else when Cmplx == complex128 {
		lapack.zhfrk_(&transr, &uplo, &trans_c, &n, &k, &alpha, raw_data(A.data), &lda, &beta, raw_data(C.data))
	}
}
