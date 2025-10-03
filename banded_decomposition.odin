package openblas

import lapack "./f77"
import "base:builtin"

// ===================================================================================
// BANDED MATRIX DECOMPOSITION ROUTINES
//
// This file provides decomposition operations for banded matrices:
// - Bidiagonal reduction (GBBRD routines)
// - QR decomposition (GB-specific routines)
// - SVD preprocessing for banded matrices
//
// The routines in this file handle the conversion of general banded matrices
// to more structured forms suitable for eigenvalue computation or solving
// linear systems efficiently.
// ===================================================================================

// ===================================================================================
// BANDED TO BIDIAGONAL REDUCTION
// ===================================================================================

// Options for which orthogonal matrices to compute during bidiagonal reduction
BidiagVectorOption :: enum u8 {
	NONE    = 'N', // Compute neither Q nor P^T
	Q_ONLY  = 'Q', // Compute Q only
	PT_ONLY = 'P', // Compute P^T only
	BOTH    = 'B', // Compute both Q and P^T
}

// Reduce general banded matrix to bidiagonal form
banded_to_bidiag :: proc {
	banded_to_bidiag_real,
	banded_to_bidiag_complex,
}

// ===================================================================================
// WORKSPACE AND SIZE QUERIES
// ===================================================================================

// Query result sizes for banded to bidiagonal reduction
query_result_sizes_banded_to_bidiag :: proc(
	m, n: int,
	vect: BidiagVectorOption,
	apply_c: bool = false,
) -> (
	d_size: int,
	e_size: int,
	q_rows: int,
	q_cols: int,// Diagonal elements
	pt_rows: int,// Off-diagonal elements
	pt_cols: int,// Q matrix rows
	c_required: bool, // Q matrix cols// P^T matrix rows// P^T matrix cols// Whether C matrix workspace is needed
) {
	min_mn := min(m, n)
	d_size = min_mn
	e_size = max(0, min_mn - 1)

	// Q matrix dimensions
	if vect == .Q_ONLY || vect == .BOTH {
		q_rows = m
		q_cols = min_mn
	}

	// P^T matrix dimensions
	if vect == .PT_ONLY || vect == .BOTH {
		pt_rows = min_mn
		pt_cols = n
	}

	c_required = apply_c

	return
}

// Query workspace for banded to bidiagonal reduction
query_workspace_banded_to_bidiag :: proc($T: typeid, m, n: int) -> (work: int, rwork: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		work = 2 * max(m, n)
		rwork = 0
	} else when is_complex(T) {
		work = max(m, n)
		rwork = max(m, n)
	}
	return
}

// ===================================================================================
// IMPLEMENTATION - REAL TYPES
// ===================================================================================

// Reduce general banded matrix to bidiagonal form (real version)
banded_to_bidiag_real :: proc(
	vect: BidiagVectorOption,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - destroyed)
	d: []T, // Pre-allocated diagonal elements (size min(m,n))
	e: []T, // Pre-allocated off-diagonal elements (size min(m,n)-1)
	Q: ^Matrix(T) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(T) = nil, // Pre-allocated P^T matrix (optional)
	C: ^Matrix(T) = nil, // Matrix to apply transformations to (optional)
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	// Validate input dimensions
	min_mn := min(int(m), int(n))
	assert(len(d) >= min_mn, "Diagonal array too small")
	assert(len(e) >= max(0, min_mn - 1), "Off-diagonal array too small")
	assert(len(work) >= 2 * max(int(m), int(n)), "Work array too small")

	// Validate and set up Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^T = nil
	if (vect == .Q_ONLY || vect == .BOTH) && Q != nil {
		assert(int(Q.rows) >= int(m) && int(Q.cols) >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	// Validate and set up P^T matrix
	ldpt := Blas_Int(1)
	pt_ptr: ^T = nil
	if (vect == .PT_ONLY || vect == .BOTH) && PT != nil {
		assert(int(PT.rows) >= min_mn && int(PT.cols) >= int(n), "P^T matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle optional C matrix for applying transformations
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^T = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
	}

	// Convert enum to char for LAPACK
	vect_c := cast(u8)vect

	when T == f32 {
		lapack.sgbbrd_(&vect_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(d), raw_data(e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), &info)
	} else when T == f64 {
		lapack.dgbbrd_(&vect_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(d), raw_data(e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), &info)
	}

	return info, info == 0
}

// ===================================================================================
// IMPLEMENTATION - COMPLEX TYPES
// ===================================================================================

// Reduce general banded matrix to bidiagonal form (complex version)
banded_to_bidiag_complex :: proc(
	vect: BidiagVectorOption,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - destroyed)
	d: []$Real, // Pre-allocated real diagonal elements (size min(m,n))
	e: []Real, // Pre-allocated real off-diagonal elements (size min(m,n)-1)
	Q: ^Matrix(Cmplx) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(Cmplx) = nil, // Pre-allocated P^T matrix (optional)
	C: ^Matrix(Cmplx) = nil, // Matrix to apply transformations to (optional)
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	// Validate input dimensions
	min_mn := min(int(m), int(n))
	assert(len(d) >= min_mn, "Diagonal array too small")
	assert(len(e) >= max(0, min_mn - 1), "Off-diagonal array too small")
	assert(len(work) >= max(int(m), int(n)), "Work array too small")
	assert(len(rwork) >= max(int(m), int(n)), "Real work array too small")

	// Validate and set up Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^Cmplx = nil
	if (vect == .Q_ONLY || vect == .BOTH) && Q != nil {
		assert(int(Q.rows) >= int(m) && int(Q.cols) >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	// Validate and set up P^T matrix
	ldpt := Blas_Int(1)
	pt_ptr: ^Cmplx = nil
	if (vect == .PT_ONLY || vect == .BOTH) && PT != nil {
		assert(int(PT.rows) >= min_mn && int(PT.cols) >= int(n), "P^T matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle optional C matrix for applying transformations
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^Cmplx = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
	}

	// Convert enum to char for LAPACK
	vect_c := cast(u8)vect

	when Cmplx == complex64 {
		lapack.cgbbrd_(&vect_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(d), raw_data(e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgbbrd_(&vect_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(d), raw_data(e), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// CONVENIENCE WRAPPERS
// ===================================================================================

// Allocate workspace and perform banded to bidiagonal reduction
banded_to_bidiag_alloc :: proc {
	banded_to_bidiag_alloc_real,
	banded_to_bidiag_alloc_complex,
}

// Real version
banded_to_bidiag_alloc_real :: proc(
	vect: BidiagVectorOption,
	AB: ^BandedMatrix($T), // Banded matrix (input/output - destroyed)
	compute_q: bool = false,
	compute_pt: bool = false,
	allocator := context.allocator,
) -> (
	d: []T,
	e: []T,
	Q: Matrix(T),
	PT: Matrix(T),// Diagonal elements
	info: Info,// Off-diagonal elements
	ok: bool, // Q matrix (if requested)// P^T matrix (if requested)
) where is_float(T) {
	m, n := int(AB.rows), int(AB.cols)
	min_mn := min(m, n)

	// Allocate result arrays
	d = make([]T, min_mn, allocator)
	e = make([]T, max(0, min_mn - 1), allocator)

	// Allocate workspace
	work_size, rwork_size := query_workspace_banded_to_bidiag(T, m, n)
	work := make([]T, work_size, allocator)
	defer delete(work, allocator)

	// Allocate optional matrices
	q_ptr: ^Matrix(T)
	pt_ptr: ^Matrix(T)

	if compute_q && (vect == .Q_ONLY || vect == .BOTH) {
		Q = Matrix(T) {
			data   = make([]T, m * min_mn, allocator),
			rows   = Blas_Int(m),
			cols   = Blas_Int(min_mn),
			ld     = Blas_Int(m),
			format = .General,
		}
		q_ptr = &Q
	}

	if compute_pt && (vect == .PT_ONLY || vect == .BOTH) {
		PT = Matrix(T) {
			data   = make([]T, min_mn * n, allocator),
			rows   = Blas_Int(min_mn),
			cols   = Blas_Int(n),
			ld     = Blas_Int(min_mn),
			format = .General,
		}
		pt_ptr = &PT
	}

	// Call appropriate implementation
	info, ok = banded_to_bidiag_real(vect, AB, d, e, q_ptr, pt_ptr, nil, work)

	return
}

// Complex version
banded_to_bidiag_alloc_complex :: proc(
	vect: BidiagVectorOption,
	AB: ^BandedMatrix($Cmplx), // Banded matrix (input/output - destroyed)
	compute_q: bool = false,
	compute_pt: bool = false,
	allocator := context.allocator,
) -> (
	d: []$Real,
	e: []Real,
	Q: Matrix(Cmplx),
	PT: Matrix(Cmplx),// Diagonal elements
	info: Info,// Off-diagonal elements
	ok: bool, // Q matrix (if requested)// P^T matrix (if requested)
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	m, n := int(AB.rows), int(AB.cols)
	min_mn := min(m, n)

	// Allocate result arrays
	d = make([]Real, min_mn, allocator)
	e = make([]Real, max(0, min_mn - 1), allocator)

	// Allocate workspace
	work_size, rwork_size := query_workspace_banded_to_bidiag(Cmplx, m, n)
	work := make([]Cmplx, work_size, allocator)
	defer delete(work, allocator)

	rwork := make([]Real, rwork_size, allocator)
	defer delete(rwork, allocator)

	// Allocate optional matrices
	q_ptr: ^Matrix(Cmplx)
	pt_ptr: ^Matrix(Cmplx)

	if compute_q && (vect == .Q_ONLY || vect == .BOTH) {
		Q = Matrix(Cmplx) {
			data   = make([]Cmplx, m * min_mn, allocator),
			rows   = Blas_Int(m),
			cols   = Blas_Int(min_mn),
			ld     = Blas_Int(m),
			format = .General,
		}
		q_ptr = &Q
	}

	if compute_pt && (vect == .PT_ONLY || vect == .BOTH) {
		PT = Matrix(Cmplx) {
			data   = make([]Cmplx, min_mn * n, allocator),
			rows   = Blas_Int(min_mn),
			cols   = Blas_Int(n),
			ld     = Blas_Int(min_mn),
			format = .General,
		}
		pt_ptr = &PT
	}

	// Call appropriate implementation
	info, ok = banded_to_bidiag_complex(vect, AB, d, e, q_ptr, pt_ptr, nil, work, rwork)

	return
}
