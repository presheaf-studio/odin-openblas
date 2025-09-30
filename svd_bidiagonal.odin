package openblas

import lapack "./f77"

// ===================================================================================
// BIDIAGONAL MATRIX TYPE
// ===================================================================================

// Bidiagonal matrix representation for SVD computations
// Stores only the diagonal and one off-diagonal (2n-1 elements total)
Bidiagonal :: struct($T: typeid) {
	n:    Blas_Int, // Matrix dimension (n x n)
	uplo: UpLo, // Upper or Lower bidiagonal
	d:    []T, // Main diagonal (length n)
	e:    []T, // Off-diagonal (length n-1)
	// If uplo='U': superdiagonal
	// If uplo='L': subdiagonal
}

// Create a new bidiagonal matrix
make_bidiagonal :: proc($T: typeid, n: int, uplo: UpLo = .Upper, allocator := context.allocator) -> Bidiagonal(T) {
	return Bidiagonal(T){n = Blas_Int(n), uplo = uplo, d = make([]T, n, allocator), e = make([]T, max(0, n - 1), allocator)}
}

// Delete bidiagonal matrix
delete_bidiagonal :: proc(B: ^Bidiagonal($T)) {
	delete(B.d)
	delete(B.e)
}

// Create from existing diagonal arrays (takes ownership)
bidiagonal_from_arrays :: proc($T: typeid, d: []T, e: []T, uplo: UpLo = .Upper) -> Bidiagonal(T) {
	assert(len(d) > 0, "Main diagonal must have at least one element")
	assert(len(e) == len(d) - 1, "Off-diagonal must have n-1 elements")

	return Bidiagonal(T){n = Blas_Int(len(d)), uplo = uplo, d = d, e = e}
}

// ===================================================================================
// BIDIAGONAL SVD OPERATIONS
// ===================================================================================

// SVD of bidiagonal matrix using bdsqr
// Computes the SVD of a real n-by-n (upper or lower) bidiagonal matrix B
bidiagonal_svd :: proc {
	bidiagonal_svd_real,
	bidiagonal_svd_complex,
}

// Query workspace size for bidiagonal SVD
query_workspace_bidiagonal_svd :: proc(B: ^Bidiagonal($T), compute_u: bool = true, compute_vt: bool = true, compute_c: bool = false) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	n := int(B.n)

	when is_float(T) {
		work_size = 4 * n // Real types need work array
		rwork_size = 0
	} else {
		work_size = 0
		rwork_size = 4 * n // Complex types need real work array
	}

	return
}

// Real bidiagonal SVD (f32/f64)
bidiagonal_svd_real :: proc(
	B: ^Bidiagonal($T),
	U: ^Matrix(T) = nil, // Left singular vectors (optional)
	VT: ^Matrix(T) = nil, // Right singular vectors transposed (optional)
	C: ^Matrix(T) = nil, // Additional matrix to transform (optional)
	work: []T, // Workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := B.n

	ncvt := Blas_Int(0) // Number of columns of VT
	nru := Blas_Int(0) // Number of rows of U
	ncc := Blas_Int(0) // Number of columns of C

	ldu := Blas_Int(1)
	ldvt := Blas_Int(1)
	ldc := Blas_Int(1)

	u_ptr: ^T = nil
	vt_ptr: ^T = nil
	c_ptr: ^T = nil

	if VT != nil {
		ncvt = VT.cols
		ldvt = VT.ld
		vt_ptr = raw_data(VT.data)
		assert(VT.rows >= n, "VT must have at least n rows")
	}

	if U != nil {
		nru = U.rows
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.cols >= n, "U must have at least n columns")
	}

	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C must have at least n rows")
	}

	assert(len(work) >= 4 * int(n), "Work array too small")

	uplo_c := cast(u8)B.uplo

	when T == f32 {
		lapack.sbdsqr_(&uplo_c, &n, &ncvt, &nru, &ncc, raw_data(B.d), raw_data(B.e), vt_ptr, &ldvt, u_ptr, &ldu, c_ptr, &ldc, raw_data(work), &info)
	} else when T == f64 {
		lapack.dbdsqr_(&uplo_c, &n, &ncvt, &nru, &ncc, raw_data(B.d), raw_data(B.e), vt_ptr, &ldvt, u_ptr, &ldu, c_ptr, &ldc, raw_data(work), &info)
	}

	return info, info == 0
}

// Complex bidiagonal SVD (complex64/complex128)
bidiagonal_svd_complex :: proc(
	B: ^Bidiagonal($Real), // Bidiagonal matrix (real)
	U: ^Matrix($T) = nil, // Left singular vectors (optional, complex)
	VT: ^Matrix(T) = nil, // Right singular vectors transposed (optional, complex)
	C: ^Matrix(T) = nil, // Additional matrix to transform (optional, complex)
	rwork: []Real, // Real workspace (pre-allocated)
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	n := B.n

	// Determine what to compute
	ncvt := Blas_Int(0)
	nru := Blas_Int(0)
	ncc := Blas_Int(0)

	ldu := Blas_Int(1)
	ldvt := Blas_Int(1)
	ldc := Blas_Int(1)

	u_ptr: ^T = nil
	vt_ptr: ^T = nil
	c_ptr: ^T = nil

	if VT != nil {
		ncvt = VT.cols
		ldvt = VT.ld
		vt_ptr = raw_data(VT.data)
		assert(VT.rows >= n, "VT must have at least n rows")
	}

	if U != nil {
		nru = U.rows
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.cols >= n, "U must have at least n columns")
	}

	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C must have at least n rows")
	}

	assert(len(rwork) >= 4 * int(n), "Real work array too small")

	uplo_c := cast(u8)B.uplo

	when T == complex64 {
		lapack.cbdsqr_(&uplo_c, &n, &ncvt, &nru, &ncc, raw_data(B.d), raw_data(B.e), vt_ptr, &ldvt, u_ptr, &ldu, c_ptr, &ldc, raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zbdsqr_(&uplo_c, &n, &ncvt, &nru, &ncc, raw_data(B.d), raw_data(B.e), vt_ptr, &ldvt, u_ptr, &ldu, c_ptr, &ldc, raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL DIVIDE AND CONQUER SVD
// ===================================================================================

// Bidiagonal SVD using divide-and-conquer (bdsdc)
bidiagonal_svd_dc :: proc {
	bidiagonal_svd_dc_real,
}

// Job options for bidiagonal divide-and-conquer
Bidiagonal_DC_Job :: enum u8 {
	None     = 'N', // Compute singular values only
	Compact  = 'P', // Compute singular values and compact singular vectors
	Implicit = 'I', // Compute singular values and singular vectors
}

// Real bidiagonal divide-and-conquer SVD (f32/f64)
bidiagonal_svd_dc_real :: proc(B: ^Bidiagonal($T), U: ^Matrix(T) = nil, VT: ^Matrix(T) = nil, work: []T, iwork: []Blas_Int, compq: Bidiagonal_DC_Job = .Implicit) -> (info: Info, ok: bool) where is_float(T) {
	n := B.n
	uplo_c := cast(u8)B.uplo
	compq_c := cast(u8)compq

	ldu := Blas_Int(1)
	ldvt := Blas_Int(1)
	u_ptr: ^T = nil
	vt_ptr: ^T = nil

	if compq == .Implicit {
		if U != nil {
			ldu = U.ld
			u_ptr = raw_data(U.data)
			assert(U.rows == n && U.cols == n, "U must be n x n")
		}
		if VT != nil {
			ldvt = VT.ld
			vt_ptr = raw_data(VT.data)
			assert(VT.rows == n && VT.cols == n, "VT must be n x n")
		}
	}

	assert(compq != .Compact, ".Compact Not Implemented")
	// Q and IQ are only used for compact storage (not implemented here)
	q_dummy: T
	iq_dummy: Blas_Int

	when T == f32 {
		lapack.sbdsdc_(&uplo_c, &compq_c, &n, raw_data(B.d), raw_data(B.e), u_ptr, &ldu, vt_ptr, &ldvt, &q_dummy, &iq_dummy, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dbdsdc_(&uplo_c, &compq_c, &n, raw_data(B.d), raw_data(B.e), u_ptr, &ldu, vt_ptr, &ldvt, &q_dummy, &iq_dummy, raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL SELECTIVE SVD
// ===================================================================================

// Selective SVD of bidiagonal matrix (bdsvdx)
bidiagonal_svd_select :: proc {
	bidiagonal_svd_select_real,
}

// Real selective bidiagonal SVD (f32/f64)
bidiagonal_svd_select_real :: proc(
	B: ^Bidiagonal($T),
	S: []T, // Output singular values
	Z: ^Matrix(T) = nil, // Output singular vectors
	work: []T,
	iwork: []Blas_Int,
	jobz: SVD_Select_Job = .Vectors,
	range: SVD_Range_Option = .All,
	vl: T,
	vu: T,
	il: Blas_Int = 1,
	iu: Blas_Int = -1,
) -> (
	ns: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := B.n
	uplo_c := cast(u8)B.uplo
	jobz_c := cast(u8)jobz
	range_c := cast(u8)range

	vl := vl
	vu := vu
	il := il
	iu := iu

	if iu < 0 {iu = n}

	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if Z != nil && jobz == .Vectors {
		ldz = Z.ld
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		lapack.sbdsvdx_(&uplo_c, &jobz_c, &range_c, &n, raw_data(B.d), raw_data(B.e), &vl, &vu, &il, &iu, &ns, raw_data(S), z_ptr, &ldz, raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dbdsvdx_(&uplo_c, &jobz_c, &range_c, &n, raw_data(B.d), raw_data(B.e), &vl, &vu, &il, &iu, &ns, raw_data(S), z_ptr, &ldz, raw_data(work), raw_data(iwork), &info)
	}

	return ns, info, info == 0
}


// ===================================================================================
// BIDIAGONAL REDUCTION
// Reduce general matrix to bidiagonal form for SVD computation
// ===================================================================================

// Reduce general matrix to bidiagonal form using Householder reflections
// A = Q * B * P^H where B is bidiagonal
m_bidiagonalize :: proc {
	m_bidiagonalize_real,
	m_bidiagonalize_c64,
	m_bidiagonalize_c128,
}

m_bidiagonalize_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with bidiagonal form)
	allocator := context.allocator,
) -> (
	D: []T,
	E: []T,
	tauq: []T,
	taup: []T,
	info: Info, // Diagonal elements of B// Off-diagonal elements of B// Scalar factors for Q// Scalar factors for P
) where T == f32 || T == f64 {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	min_mn := min(m, n)

	// Allocate output arrays
	D = make([]T, min_mn, allocator)
	E = make([]T, min_mn - 1, allocator)
	tauq = make([]T, min_mn, allocator)
	taup = make([]T, min_mn, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: T

	when T == f32 {
		lapack.sgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), &work_query, &lwork, &info)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := make([]T, lwork, allocator)
		defer delete(work)

		// Perform reduction
		lapack.sgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), &work_query, &lwork, &info)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := make([]T, lwork, allocator)
		defer delete(work)

		// Perform reduction
		lapack.dgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	}

	return D, E, tauq, taup, info
}

m_bidiagonalize_c64 :: proc(
	A: ^Matrix(complex64),
	allocator := context.allocator,
) -> (
	D: []f32,
	E: []f32,
	tauq: []complex64,
	taup: []complex64,
	info: Info, // Real diagonal elements// Real off-diagonal elements// Scalar factors for Q// Scalar factors for P
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	min_mn := min(m, n)

	// Allocate output arrays
	D = make([]f32, min_mn, allocator)
	E = make([]f32, min_mn - 1, allocator)
	tauq = make([]complex64, min_mn, allocator)
	taup = make([]complex64, min_mn, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex64

	lapack.cgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), &work_query, &lwork, &info)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := make([]complex64, lwork, allocator)
	defer delete(work)

	// Perform reduction
	lapack.cgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)

	return D, E, tauq, taup, info
}

m_bidiagonalize_c128 :: proc(
	A: ^Matrix(complex128),
	allocator := context.allocator,
) -> (
	D: []f64,
	E: []f64,
	tauq: []complex128,
	taup: []complex128,
	info: Info, // Real diagonal elements// Real off-diagonal elements// Scalar factors for Q// Scalar factors for P
) {
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	min_mn := min(m, n)

	// Allocate output arrays
	D = make([]f64, min_mn, allocator)
	E = make([]f64, min_mn - 1, allocator)
	tauq = make([]complex128, min_mn, allocator)
	taup = make([]complex128, min_mn, allocator)

	// Query for optimal workspace
	lwork := Blas_Int(-1)
	work_query: complex128

	lapack.zgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), &work_query, &lwork, &info)

	// Allocate workspace
	lwork = Blas_Int(real(work_query))
	work := make([]complex128, lwork, allocator)
	defer delete(work)

	// Perform reduction
	lapack.zgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)

	return D, E, tauq, taup, info
}
