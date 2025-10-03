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
RFP :: struct($T: typeid) where is_float(T) || is_complex(T) {
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

// Create RFP matrix with zero initialization
make_rfp_zero :: proc(n: int, trans_state: TransposeState, uplo: UpLo, $T: typeid, allocator := context.allocator) -> RFP(T) {
	rfp := make_rfp(n, trans_state, uplo, T, allocator)
	// Data is already zero-initialized by make
	return rfp
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
// WORKSPACE AND SIZE QUERIES
// ===================================================================================

// Query workspace requirements for RFP operations
query_workspace_rfp :: proc(operation: string, $T: typeid, n: int) -> (work_size, rwork_size, iwork_size: int) {
	// Default workspace requirements for common operations
	switch operation {
	case "factor":
		// Cholesky factorization for RFP
		work_size = 0
		rwork_size = 0
		iwork_size = 0
	case "solve":
		// Linear solve for RFP
		work_size = 0
		rwork_size = 0
		iwork_size = 0
	case "invert":
		// Matrix inversion for RFP
		work_size = n
		rwork_size = 0
		iwork_size = 0
	case "conversion":
		// Format conversion
		work_size = 0
		rwork_size = 0
		iwork_size = 0
	case:
		// Unknown operation, return minimal workspace
		work_size = n
		rwork_size = 0
		iwork_size = 0
	}

	return
}

// Allocate workspace for RFP operations
allocate_rfp_workspace :: proc(operation: string, $T: typeid, n: int, allocator := context.allocator) -> (work: []T, rwork: []f64, iwork: []Blas_Int) {
	work_size, rwork_size, iwork_size := query_workspace_rfp(operation, T, n)

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

// Get effective matrix dimension
get_rfp_dimension :: proc(rm: ^RFP($T)) -> int {
	return int(rm.n)
}

// Check if RFP uses transposed storage
is_rfp_transposed :: proc(rm: ^RFP($T)) -> bool {
	return rm.trans_state != .NoTrans
}

// ===================================================================================
// RFP CHOLESKY FACTORIZATION (PFTRF family)
// ===================================================================================

rfp_cholesky_factorize :: proc {
	rfp_cholesky_factorize_real,
	rfp_cholesky_factorize_complex,
}

// Compute Cholesky factorization of a positive definite matrix in RFP format
// A = U**T * U or A = L * L**T (real)
// A = U**H * U or A = L * L**H (complex)
rfp_cholesky_factorize_real :: proc(
	A: ^RFP($T), // RFP matrix (input/output, overwritten with factor)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	n := A.n

	when T == f32 {
		lapack.spftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == f64 {
		lapack.dpftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
	}

	return info, info == 0
}

rfp_cholesky_factorize_complex :: proc(
	A: ^RFP($T), // RFP matrix (input/output, overwritten with factor)
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	n := A.n

	when T == complex64 {
		lapack.cpftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == complex128 {
		lapack.zpftrf_(&transr, &uplo, &n, raw_data(A.data), &info)
	}

	return info, info == 0
}

// ===================================================================================
// RFP CHOLESKY SOLVE (PFTRS family)
// ===================================================================================

rfp_cholesky_solve :: proc {
	rfp_cholesky_solve_real,
	rfp_cholesky_solve_complex,
}

// Solve linear system using RFP Cholesky factorization: A*X = B
rfp_cholesky_solve_real :: proc(
	A: ^RFP($T), // Cholesky factors in RFP format
	B: ^Matrix(T), // RHS matrix (input/output, overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
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
	}

	return info, info == 0
}

rfp_cholesky_solve_complex :: proc(
	A: ^RFP($T), // Cholesky factors in RFP format
	B: ^Matrix(T), // RHS matrix (input/output, overwritten with solution)
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")
	assert(B.rows == A.n, "B matrix must have same number of rows as A dimension")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	n := A.n
	nrhs := B.cols
	ldb := B.ld

	when T == complex64 {
		lapack.cpftrs_(&transr, &uplo, &n, &nrhs, raw_data(A.data), raw_data(B.data), &ldb, &info)
	} else when T == complex128 {
		lapack.zpftrs_(&transr, &uplo, &n, &nrhs, raw_data(A.data), raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

// ===================================================================================
// RFP CHOLESKY INVERSION (PFTRI family)
// ===================================================================================

rfp_cholesky_invert :: proc {
	rfp_cholesky_invert_real,
	rfp_cholesky_invert_complex,
}

// Compute inverse of positive definite matrix using RFP Cholesky factorization
rfp_cholesky_invert_real :: proc(
	A: ^RFP($T), // Cholesky factors (input/output, overwritten with inverse)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	n := A.n

	when T == f32 {
		lapack.spftri_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == f64 {
		lapack.dpftri_(&transr, &uplo, &n, raw_data(A.data), &info)
	}

	return info, info == 0
}

rfp_cholesky_invert_complex :: proc(
	A: ^RFP($T), // Cholesky factors (input/output, overwritten with inverse)
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	n := A.n

	when T == complex64 {
		lapack.cpftri_(&transr, &uplo, &n, raw_data(A.data), &info)
	} else when T == complex128 {
		lapack.zpftri_(&transr, &uplo, &n, raw_data(A.data), &info)
	}

	return info, info == 0
}

// ===================================================================================
// RFP TRIANGULAR SOLVE (TFSM family)
// ===================================================================================

rfp_triangular_solve :: proc {
	rfp_triangular_solve_real,
	rfp_triangular_solve_complex,
}

// Solve triangular system with RFP format matrix
// Solves: op(A) * X = alpha * B  (side = Left)
//     or: X * op(A) = alpha * B  (side = Right)
// where A is stored in RFP format
rfp_triangular_solve_real :: proc(
	A: ^RFP($T), // Triangular matrix in RFP format
	B: ^Matrix(T), // Right-hand side matrix (solution on output)
	alpha: T, // Scalar multiplier
	side: Side = .Left,
	trans: TransposeState = .NoTrans,
	diag: Diag = .NonUnit,
) -> (
	ok: bool,
) where is_float(T) {
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
	alpha_copy := alpha

	when T == f32 {
		lapack.stfsm_(&transr, &side_c, &uplo, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(A.data), raw_data(B.data), &ldb)
	} else when T == f64 {
		lapack.dtfsm_(&transr, &side_c, &uplo, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(A.data), raw_data(B.data), &ldb)
	}

	return true
}

rfp_triangular_solve_complex :: proc(
	A: ^RFP($T), // Triangular matrix in RFP format
	B: ^Matrix(T), // Right-hand side matrix (solution on output)
	alpha: T, // Scalar multiplier
	side: Side = .Left,
	trans: TransposeState = .NoTrans,
	diag: Diag = .NonUnit,
) -> (
	ok: bool,
) where is_complex(T) {
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
	alpha_copy := alpha

	when T == complex64 {
		lapack.ctfsm_(&transr, &side_c, &uplo, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(A.data), raw_data(B.data), &ldb)
	} else when T == complex128 {
		lapack.ztfsm_(&transr, &side_c, &uplo, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(A.data), raw_data(B.data), &ldb)
	}

	return true
}

// ===================================================================================
// RFP TRIANGULAR INVERSION (TFTRI family)
// ===================================================================================

rfp_triangular_invert :: proc {
	rfp_triangular_invert_real,
	rfp_triangular_invert_complex,
}

// Invert triangular matrix in RFP format
// Computes the inverse of a triangular matrix A stored in RFP format
rfp_triangular_invert_real :: proc(
	A: ^RFP($T), // Triangular matrix in RFP format (inverted in place)
	diag: Diag = .NonUnit,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	diag_c := cast(u8)diag
	n := A.n

	when T == f32 {
		lapack.stftri_(&transr, &uplo, &diag_c, &n, raw_data(A.data), &info)
	} else when T == f64 {
		lapack.dtftri_(&transr, &uplo, &diag_c, &n, raw_data(A.data), &info)
	}

	return info, info == 0
}

rfp_triangular_invert_complex :: proc(
	A: ^RFP($T), // Triangular matrix in RFP format (inverted in place)
	diag: Diag = .NonUnit,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	assert(validate_rfp(A), "Invalid RFP matrix structure")

	transr := cast(u8)A.trans_state
	uplo := cast(u8)A.uplo
	diag_c := cast(u8)diag
	n := A.n

	when T == complex64 {
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
	alpha: T = 1.0,
	beta: T = 0.0,
	trans: TransposeState = .NoTrans,
) where is_float(T) {
	assert(validate_rfp(C), "Invalid RFP matrix structure")

	n := C.n
	k := (trans == .NoTrans) ? A.cols : A.rows
	lda := A.ld

	transr := cast(u8)C.trans_state
	uplo := cast(u8)C.uplo
	trans_c := cast(u8)trans

	alpha_copy := alpha
	beta_copy := beta

	when T == f32 {
		lapack.ssfrk_(&transr, &uplo, &trans_c, &n, &k, &alpha_copy, raw_data(A.data), &lda, &beta_copy, raw_data(C.data))
	} else when T == f64 {
		lapack.dsfrk_(&transr, &uplo, &trans_c, &n, &k, &alpha_copy, raw_data(A.data), &lda, &beta_copy, raw_data(C.data))
	}
	// Note: sfrk/dfrk don't have info parameter
}

// ===================================================================================
// RFP HERMITIAN RANK-K UPDATE (HFRK family - complex only)
// ===================================================================================

rfp_hermitian_rank_k_update :: proc {
	rfp_hermitian_rank_k_update_c64,
	rfp_hermitian_rank_k_update_c128,
}

// Hermitian rank-k update in RFP format: C := alpha*A*A^H + beta*C
rfp_hermitian_rank_k_update_c64 :: proc(
	C: ^RFP(complex64), // Hermitian matrix in RFP format (output)
	A: ^Matrix(complex64), // Input matrix
	alpha: f32 = 1.0,
	beta: f32 = 0.0,
	trans: TransposeState = .NoTrans,
) {
	assert(validate_rfp(C), "Invalid RFP matrix structure")

	n := C.n
	k := (trans == .NoTrans) ? A.cols : A.rows
	lda := A.ld

	transr := cast(u8)C.trans_state
	uplo := cast(u8)C.uplo
	trans_c := cast(u8)trans

	alpha_copy := alpha
	beta_copy := beta

	lapack.chfrk_(&transr, &uplo, &trans_c, &n, &k, &alpha_copy, raw_data(A.data), &lda, &beta_copy, raw_data(C.data))
}

rfp_hermitian_rank_k_update_c128 :: proc(
	C: ^RFP(complex128), // Hermitian matrix in RFP format (output)
	A: ^Matrix(complex128), // Input matrix
	alpha: f64 = 1.0,
	beta: f64 = 0.0,
	trans: TransposeState = .NoTrans,
) {
	assert(validate_rfp(C), "Invalid RFP matrix structure")

	n := C.n
	k := (trans == .NoTrans) ? A.cols : A.rows
	lda := A.ld

	transr := cast(u8)C.trans_state
	uplo := cast(u8)C.uplo
	trans_c := cast(u8)trans

	alpha_copy := alpha
	beta_copy := beta

	lapack.zhfrk_(&transr, &uplo, &trans_c, &n, &k, &alpha_copy, raw_data(A.data), &lda, &beta_copy, raw_data(C.data))
}
