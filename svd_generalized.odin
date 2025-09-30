package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"

// Job options for Generalized SVD matrix computation
GSVD_Job :: enum u8 {
	Compute = 'U', // Compute matrix (U for left, V for middle, Q for right)
	Full    = 'V', // Alternative name for compute (used for V)
	Right   = 'Q', // Alternative name for compute (used for Q)
	None    = 'N', // Do not compute matrix
}

// Query workspace size for blocked generalized SVD
query_workspace_gsvd_blocked :: proc(
	$T: typeid,
	m, n, p: int,
	jobu := GSVD_Job.Compute,
	jobv := GSVD_Job.Full,
	jobq := GSVD_Job.Right,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
) where is_float(T) ||
	is_complex(T) {
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	p_int := Blas_Int(p)

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	// Integer workspace is always n
	iwork_size = n

	// Real workspace depends on type
	when is_float(T) {
		rwork_size = 0 // No real workspace for real types
	} else {
		rwork_size = 2 * n // Complex types need real workspace
	}

	// Query for optimal workspace
	lwork := QUERY_WORKSPACE
	work_query: T
	k, l: Blas_Int
	info: Info

	when T == f32 {
		lapack.sggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &n_int, &p_int, &k, &l, nil, &m_int, nil, &p_int, nil, nil, nil, &m_int, nil, &p_int, nil, &n_int, &work_query, &lwork, nil, &info)
	} else when T == f64 {
		lapack.dggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &n_int, &p_int, &k, &l, nil, &m_int, nil, &p_int, nil, nil, nil, &m_int, nil, &p_int, nil, &n_int, &work_query, &lwork, nil, &info)
	} else when T == complex64 {
		lapack.cggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &n_int, &p_int, &k, &l, nil, &m_int, nil, &p_int, nil, nil, nil, &m_int, nil, &p_int, nil, &n_int, &work_query, &lwork, nil, nil, &info)
	} else when T == complex128 {
		lapack.zggsvd3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &n_int, &p_int, &k, &l, nil, &m_int, nil, &p_int, nil, nil, nil, &m_int, nil, &p_int, nil, &n_int, &work_query, &lwork, nil, nil, &info)
	}

	// Convert work query result
	when is_float(T) {
		work_size = int(work_query)
	} else {
		work_size = int(real(work_query))
	}

	return
}

// Blocked/improved version of generalized SVD
m_gsvd_blocked :: proc {
	m_gsvd_blocked_real,
	m_gsvd_blocked_complex,
}

m_gsvd_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	alpha: []T, // Pre-allocated output singular values (size n)
	beta: []T, // Pre-allocated output singular values (size n)
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	p := B.rows
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	assert(len(alpha) >= int(n), "alpha array too small")
	assert(len(beta) >= int(n), "beta array too small")
	assert(len(work) > 0, "workspace required")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^T = nil
	v_ptr: ^T = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sggsvd3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dggsvd3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
		)
	}

	return k, l, info, info == 0
}

// Complex generalized SVD (complex64/complex128)
m_gsvd_blocked_complex :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	alpha: []$Real, // Pre-allocated output singular values (size n)
	beta: []Real, // Pre-allocated output singular values (size n)
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	work: []T, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace (size 2*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	p := B.rows
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	assert(len(alpha) >= int(n), "alpha array too small")
	assert(len(beta) >= int(n), "beta array too small")
	assert(len(work) > 0, "workspace required")
	assert(len(rwork) >= 2 * int(n), "real workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^T = nil
	v_ptr: ^T = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	lwork := Blas_Int(len(work))

	when T == complex64 {
		lapack.cggsvd3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
		)
	} else when T == complex128 {
		lapack.zggsvd3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
		)
	}

	return k, l, info, info == 0
}


// ===================================================================================
// GENERALIZED SINGULAR VALUE DECOMPOSITION
// ===================================================================================
// Query workspace size for standard generalized SVD
query_workspace_gsvd :: proc($T: typeid, m, n, p: int) -> (work_size: int, rwork_size: int, iwork_size: int) where is_float(T) || is_complex(T) {
	// Integer workspace is always n
	iwork_size = n

	// Workspace estimates for non-blocked algorithm
	when is_float(T) {
		work_size = max(3 * n, m, p) + n
		rwork_size = 0 // No real workspace for real types
	} else {
		work_size = max(3 * n, m, p) + n
		rwork_size = 2 * n // Complex types need real workspace
	}

	return
}

// Compute generalized SVD: U^H*A*Q = D1*[0 R], V^H*B*Q = D2*[0 R]
m_gsvd :: proc {
	m_gsvd_real,
	m_gsvd_complex,
}

m_gsvd_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	alpha: []T, // Pre-allocated output singular values (size n)
	beta: []T, // Pre-allocated output singular values (size n)
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	p := B.rows

	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	assert(len(alpha) >= int(n), "alpha array too small")
	assert(len(beta) >= int(n), "beta array too small")
	assert(len(work) >= max(3 * int(n), int(m), int(p)) + int(n), "workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^T = nil
	v_ptr: ^T = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	when T == f32 {
		lapack.sggsvd_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dggsvd_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	}

	return k, l, info, info == 0
}

// Complex generalized SVD (complex64/complex128)
m_gsvd_complex :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	alpha: []$Real, // Pre-allocated output singular values (size n)
	beta: []Real, // Pre-allocated output singular values (size n)
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	work: []T, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace (size 2*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	p := B.rows

	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	assert(len(alpha) >= int(n), "alpha array too small")
	assert(len(beta) >= int(n), "beta array too small")
	assert(len(work) >= max(3 * int(n), int(m), int(p)) + int(n), "workspace too small")
	assert(len(rwork) >= 2 * int(n), "real workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^T = nil
	v_ptr: ^T = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	when T == complex64 {
		lapack.cggsvd_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(work),
			raw_data(rwork),
			raw_data(iwork),
			&info,
		)
	} else when T == complex128 {
		lapack.zggsvd_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&n,
			&p,
			&k,
			&l,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			raw_data(alpha),
			raw_data(beta),
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(work),
			raw_data(rwork),
			raw_data(iwork),
			&info,
		)
	}

	return k, l, info, info == 0
}


// ===================================================================================
// GENERALIZED SVD PREPROCESSING
// ===================================================================================

// Query workspace size for generalized SVD preprocessing
query_workspace_gsvd_preprocess :: proc($T: typeid, m, p, n: int) -> (work_size: int, rwork_size: int, iwork_size: int, tau_size: int) where is_float(T) || is_complex(T) {
	tau_size = n
	iwork_size = n

	when is_float(T) {
		work_size = max(3 * n, m, p)
		rwork_size = 0 // No real workspace for real types
	} else {
		work_size = max(3 * n, m, p)
		rwork_size = 2 * n // Complex types need real workspace
	}

	return
}

// Preprocessing for generalized SVD - reduce to standard form
m_gsvd_preprocess :: proc {
	m_gsvd_preprocess_real,
	m_gsvd_preprocess_complex,
}

m_gsvd_preprocess_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tola: T,
	tolb: T,
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	tau: []T, // Pre-allocated tau array (size n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	p := A.cols
	n := B.cols
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	tola_actual := tola
	tolb_actual := tolb

	assert(len(tau) >= int(n), "tau array too small")
	assert(len(work) >= max(3 * int(n), int(m), int(p)), "workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^T = nil
	v_ptr: ^T = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	when T == f32 {
		lapack.sggsvp_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			raw_data(work),
			&info,
		)
	} else when T == f64 {
		lapack.dggsvp_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			raw_data(work),
			&info,
		)
	}

	return k, l, info, info == 0
}

// Complex generalized SVD preprocessing
m_gsvd_preprocess_complex :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tola: $Real,
	tolb: Real,
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	tau: []T, // Pre-allocated tau array (size n)
	work: []T, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace (size 2*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
	m := A.rows
	p := A.cols
	n := B.cols
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	tola_actual := tola
	tolb_actual := tolb

	assert(len(tau) >= int(n), "tau array too small")
	assert(len(work) >= max(3 * int(n), int(m), int(p)), "workspace too small")
	assert(len(rwork) >= 2 * int(n), "real workspace too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^T = nil
	v_ptr: ^T = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	when T == complex64 {
		lapack.cggsvp_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(iwork),
			raw_data(rwork),
			raw_data(tau),
			raw_data(work),
			&info,
		)
	} else when T == complex128 {
		lapack.zggsvp_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(iwork),
			raw_data(rwork),
			raw_data(tau),
			raw_data(work),
			&info,
		)
	}

	return k, l, info, info == 0
}


// Query workspace size for blocked generalized SVD preprocessing
query_workspace_gsvd_preprocess_blocked :: proc(
	$T: typeid,
	m, p, n: int,
	jobu := GSVD_Job.Compute,
	jobv := GSVD_Job.Full,
	jobq := GSVD_Job.Right,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
	tau_size: int,
) where is_float(T) ||
	is_complex(T) {
	m_int := Blas_Int(m)
	p_int := Blas_Int(p)
	n_int := Blas_Int(n)

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	tau_size = n
	iwork_size = n

	when is_float(T) {
		rwork_size = 0 // No real workspace for real types
	} else {
		rwork_size = 2 * n
	}

	lwork := QUERY_WORKSPACE
	work_query: T
	k, l: Blas_Int
	info: Info
	tola_default := T(1e-8) when is_float(T) else 1e-8
	tolb_default := T(1e-8) when is_float(T) else 1e-8

	when T == f32 {
		lapack.sggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &p_int, &n_int, nil, &m_int, nil, &p_int, &tola_default, &tolb_default, &k, &l, nil, &m_int, nil, &p_int, nil, &n_int, nil, nil, &work_query, &lwork, &info)
	} else when T == f64 {
		lapack.dggsvp3_(&jobu_c, &jobv_c, &jobq_c, &m_int, &p_int, &n_int, nil, &m_int, nil, &p_int, &tola_default, &tolb_default, &k, &l, nil, &m_int, nil, &p_int, nil, &n_int, nil, nil, &work_query, &lwork, &info)
	} else when T == complex64 {
		lapack.cggsvp3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m_int,
			&p_int,
			&n_int,
			nil,
			&m_int,
			nil,
			&p_int,
			&tola_default,
			&tolb_default,
			&k,
			&l,
			nil,
			&m_int,
			nil,
			&p_int,
			nil,
			&n_int,
			nil,
			nil,
			nil,
			&work_query,
			&lwork,
			&info,
		)
	} else when T == complex128 {
		lapack.zggsvp3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m_int,
			&p_int,
			&n_int,
			nil,
			&m_int,
			nil,
			&p_int,
			&tola_default,
			&tolb_default,
			&k,
			&l,
			nil,
			&m_int,
			nil,
			&p_int,
			nil,
			&n_int,
			nil,
			nil,
			nil,
			&work_query,
			&lwork,
			&info,
		)
	}

	// Convert work query result
	when is_float(T) {
		work_size = int(work_query)
	} else {
		work_size = int(real(work_query))
	}

	return
}

// Improved/blocked version of generalized SVD preprocessing
m_gsvd_preprocess_blocked :: proc {
	m_gsvd_preprocess_blocked_real,
	m_gsvd_preprocess_blocked_complex,
}

m_gsvd_preprocess_blocked_real :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tola: T,
	tolb: T,
	U: ^Matrix(T) = nil, // Pre-allocated left transformation matrix (m x m)
	V: ^Matrix(T) = nil, // Pre-allocated middle transformation matrix (p x p)
	Q: ^Matrix(T) = nil, // Pre-allocated right transformation matrix (n x n)
	tau: []T, // Pre-allocated tau array (size n)
	work: []T, // Pre-allocated workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	p := A.cols
	n := B.cols
	lda := A.ld
	ldb := B.ld

	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	tola_actual := tola
	tolb_actual := tolb

	assert(len(tau) >= int(n), "tau array too small")
	assert(len(iwork) >= int(n), "integer workspace too small")

	ldu := Blas_Int(1)
	ldv := Blas_Int(1)
	ldq := Blas_Int(1)
	u_ptr: ^T = nil
	v_ptr: ^T = nil
	q_ptr: ^T = nil

	if U != nil && jobu != .None {
		ldu = U.ld
		u_ptr = raw_data(U.data)
		assert(U.rows >= m && U.cols >= m, "U must be at least m x m")
	}
	if V != nil && jobv != .None {
		ldv = V.ld
		v_ptr = raw_data(V.data)
		assert(V.rows >= p && V.cols >= p, "V must be at least p x p")
	}
	if Q != nil && jobq != .None {
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
		assert(Q.rows >= n && Q.cols >= n, "Q must be at least n x n")
	}

	lwork := Blas_Int(len(work))
	if lwork == 0 {
		lwork = QUERY_WORKSPACE
		work_query: T

		when T == f32 {
			lapack.sggsvp3_(
				&jobu_c,
				&jobv_c,
				&jobq_c,
				&m,
				&p,
				&n,
				raw_data(A.data),
				&lda,
				raw_data(B.data),
				&ldb,
				&tola_actual,
				&tolb_actual,
				&k,
				&l,
				u_ptr,
				&ldu,
				v_ptr,
				&ldv,
				q_ptr,
				&ldq,
				raw_data(iwork),
				raw_data(tau),
				&work_query,
				&lwork,
				&info,
			)
		} else when T == f64 {
			lapack.dggsvp3_(
				&jobu_c,
				&jobv_c,
				&jobq_c,
				&m,
				&p,
				&n,
				raw_data(A.data),
				&lda,
				raw_data(B.data),
				&ldb,
				&tola_actual,
				&tolb_actual,
				&k,
				&l,
				u_ptr,
				&ldu,
				v_ptr,
				&ldv,
				q_ptr,
				&ldq,
				raw_data(iwork),
				raw_data(tau),
				&work_query,
				&lwork,
				&info,
			)
		}
		// Return with required workspace size in k
		k = Blas_Int(work_query)
		return k, 0, -1, false
	}

	// Perform preprocessing
	when T == f32 {
		lapack.sggsvp3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dggsvp3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(iwork),
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return k, l, info, info == 0
}

m_gsvd_preprocess_blocked_complex :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tola: $R,
	tolb: R,
	U: ^Matrix(T) = nil,
	V: ^Matrix(T) = nil,
	Q: ^Matrix(T) = nil,
	tau: []T,
	work: []T,
	rwork: []R,
	iwork: []Blas_Int,
	jobu: GSVD_Job = .Compute,
	jobv: GSVD_Job = .Full,
	jobq: GSVD_Job = .Right,
) -> (
	k, l: Blas_Int,
	info: Info,
	ok: bool,
) where is_complex(T),
	R ==
	real_type_of(T) {
	m := Blas_Int(A.rows)
	p := Blas_Int(A.cols)
	n := Blas_Int(B.cols)

	lda := A.ld
	ldb := B.ld

	// Set job parameters
	jobu_c := cast(u8)jobu
	jobv_c := cast(u8)jobv
	jobq_c := cast(u8)jobq

	// Use machine precision if tolerances not specified
	tola_actual := tola == 0 ? (T == complex64 ? R(1.2e-7) : R(2.2e-16)) : tola
	tolb_actual := tolb == 0 ? (T == complex64 ? R(1.2e-7) : R(2.2e-16)) : tolb

	// Matrix leading dimensions
	ldu := U != nil ? U.ld : Blas_Int(1)
	ldv := V != nil ? V.ld : Blas_Int(1)
	ldq := Q != nil ? Q.ld : Blas_Int(1)

	// Matrix data pointers
	u_ptr := U != nil ? raw_data(U.data) : nil
	v_ptr := V != nil ? raw_data(V.data) : nil
	q_ptr := Q != nil ? raw_data(Q.data) : nil

	lwork := Blas_Int(len(work))

	when T == complex64 {
		lapack.cggsvp3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(iwork),
			raw_data(rwork),
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == complex128 {
		lapack.zggsvp3_(
			&jobu_c,
			&jobv_c,
			&jobq_c,
			&m,
			&p,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(B.data),
			&ldb,
			&tola_actual,
			&tolb_actual,
			&k,
			&l,
			u_ptr,
			&ldu,
			v_ptr,
			&ldv,
			q_ptr,
			&ldq,
			raw_data(iwork),
			raw_data(rwork),
			raw_data(tau),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return k, l, info, info == 0
}
