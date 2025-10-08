package openblas

import lapack "./f77"

// Import types from other modules (assuming they're defined there)
// Matrix, BandedMatrix, UpLo, Info, Blas_Int should be available in the package

// ===================================================================================
// BIDIAGONAL CONVERSION FUNCTIONS
// Functions to convert matrices TO bidiagonal form
// Note: Bidiagonal type is defined in svd_bidiagonal.odin
// ===================================================================================

// ===================================================================================
// CONVERSION TO BIDIAGONAL FORM
// ===================================================================================

// Reduce general matrix to bidiagonal form using Householder reflections
// A = Q * B * P^H where B is bidiagonal

// Query workspace size for bidiagonal reduction
query_workspace_bidi_reduce_from_dns :: proc($T: typeid, m, n: int) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	// FIXME: needs workspace query!
	switch T {
	case (f32):
	case (f64):
	case (complex64):
	case (complex128):
	case:
		panic("Invalid typeid provided")
	}
	return -1, -1 // Indicates workspace query needed
}

// Real bidiagonal reduction from dense (general matrix to bidiagonal reduction)
bidi_reduce_from_dns :: proc(
	A: ^Matrix($Real_Or_Complex), // General matrix (overwritten with bidiagonal form)
	D: []$Real, // Pre-allocated diagonal elements (length min(m,n))
	E: []Real, // Pre-allocated off-diagonal elements (length min(m,n)-1)
	tauq: []Real_Or_Complex, // Pre-allocated tau for Q (length min(m,n))
	taup: []Real_Or_Complex, // Pre-allocated tau for P (length min(m,n))
	work: []Real_Or_Complex, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where (Real_Or_Complex == f32 && Real == f32) || (Real_Or_Complex == f64 && Real == f32) || (Real_Or_Complex == complex64 && Real == f32) || (Real_Or_Complex == complex128 && Real == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(tauq) >= int(min_mn), "tauq array too small")
	assert(len(taup) >= int(min_mn), "taup array too small")
	assert(len(work) > 0, "no workspace provided")

	lwork := Blas_Int(len(work))
	when T == f32 {
		lapack.sgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	} else when Cmplx == complex64 {
		lapack.cgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// ===================================================================================
// BANDED TO BIDIAGONAL CONVERSION
// ===================================================================================

// Banded to bidiagonal computation type
BandedToBidiagJob :: enum u8 {
	None = 'N', // No orthogonal matrices computed
	Q    = 'Q', // Compute Q only
	P    = 'P', // Compute P^T only
	Both = 'B', // Compute both Q and P^T
}

// CS decomposition signs convention
CSDecompositionSigns :: enum u8 {
	Positive = '+', // Use positive signs convention
	Negative = '-', // Use negative signs convention
}

// Reduce banded matrix to bidiagonal form
bidi_reduce_from_band :: proc {
	bidi_reduce_from_band_real,
	bidi_reduce_from_band_complex,
}

// Query workspace size for banded to bidiagonal reduction
query_workspace_bidi_reduce_from_band :: proc($T: typeid, m, n, kl, ku: int, is_complex := false) -> (work_size: int, rwork_size: int) {
	if !is_complex {
		return 2 * max(m, n), 0
	} else {
		return max(m, n), max(m, n)
	}
}

// Real banded to bidiagonal reduction
bidi_reduce_from_band_real :: proc(
	AB: ^BandedMatrix($T), // Banded matrix (modified on output)
	D: []T, // Pre-allocated diagonal elements (length min(m,n))
	E: []T, // Pre-allocated off-diagonal elements (length min(m,n)-1)
	Q: ^Matrix(T), // Pre-allocated Q matrix (optional)
	PT: ^Matrix(T), // Pre-allocated P^T matrix (optional)
	C: ^Matrix(T), // Apply transformation to C (optional)
	work: []T, // Pre-allocated workspace
	job: BandedToBidiagJob = .None, // Which matrices to compute
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	min_mn := min(m, n)
	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(work) >= 2 * int(max(m, n)), "Work array too small")

	// Handle Q and PT
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	q_ptr: ^T = nil
	pt_ptr: ^T = nil

	if Q != nil && (job == .Q || job == .Both) {
		assert(Q.rows >= m && Q.cols >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	if PT != nil && (job == .P || job == .Both) {
		assert(PT.rows >= min_mn && PT.cols >= n, "PT matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^T = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C matrix too small")
	}

	job_c := cast(u8)job

	when T == f32 {
		lapack.sgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), &info)
	} else when T == f64 {
		lapack.dgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), &info)
	}

	return info, info == 0
}

// Complex banded to bidiagonal reduction
bidi_reduce_from_band_complex :: proc(
	AB: ^BandedMatrix($Cmplx), // Banded matrix (modified on output)
	D: []$Real, // Pre-allocated real diagonal elements
	E: []Real, // Pre-allocated real off-diagonal elements
	Q: ^Matrix(Cmplx) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(Cmplx) = nil, // Pre-allocated P^T matrix (optional)
	C: ^Matrix(Cmplx) = nil, // Apply transformation to C (optional)
	work: []Cmplx, // Pre-allocated workspace
	rwork: []Real, // Pre-allocated real workspace
	job: BandedToBidiagJob = .None, // Which matrices to compute
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	m := AB.rows
	n := AB.cols
	kl := AB.kl
	ku := AB.ku
	ldab := AB.ldab

	// Validate sizes
	min_mn := min(m, n)
	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(work) >= int(max(m, n)), "Work array too small")
	assert(len(rwork) >= int(max(m, n)), "Real work array too small")

	// Handle Q and PT
	ldq := Blas_Int(1)
	ldpt := Blas_Int(1)
	q_ptr: ^Cmplx = nil
	pt_ptr: ^Cmplx = nil

	if Q != nil && (job == .Q || job == .Both) {
		assert(Q.rows >= m && Q.cols >= min_mn, "Q matrix too small")
		ldq = Q.ld
		q_ptr = raw_data(Q.data)
	}

	if PT != nil && (job == .P || job == .Both) {
		assert(PT.rows >= min_mn && PT.cols >= n, "PT matrix too small")
		ldpt = PT.ld
		pt_ptr = raw_data(PT.data)
	}

	// Handle C matrix
	ncc := Blas_Int(0)
	ldc := Blas_Int(1)
	c_ptr: ^Cmplx = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C matrix too small")
	}

	job_c := cast(u8)job

	when Cmplx == complex64 {
		lapack.cgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL BLOCK DIAGONALIZATION (CS Decomposition)
// ===================================================================================


// Query workspace size for bidiagonal block diagonalization
query_workspace_bidi_block_diagonalize :: proc($T: typeid, m, p, q: int) -> (work_size: int) where is_float(T) || is_complex(T) {
	// Workspace query returns size via work_query
	// FIXME: needs workspace query!
	switch T {
	case (f32):
	case (f64):
	case (complex64):
	case (complex128):
	case:
		panic("Invalid typeid provided")
	}
	return -1 // Indicates workspace query needed
}
// Compute bidiagonal block diagonalization (real variants)
// Simultaneously bidiagonalizes the blocks of a partitioned matrix X
bidi_block_diagonalize_real :: proc(
	X11: ^Matrix($T), // (p x q) upper-left block (overwritten)
	X12: ^Matrix(T), // (p x (m-q)) upper-right block (overwritten)
	X21: ^Matrix(T), // ((m-p) x q) lower-left block (overwritten)
	X22: ^Matrix(T), // ((m-p) x (m-q)) lower-right block (overwritten)
	theta: []T, // Pre-allocated angles (length min(p,q))
	phi: []T, // Pre-allocated angles (length min(p,q))
	taup1: []T, // Pre-allocated tau for P1 (length p)
	taup2: []T, // Pre-allocated tau for P2 (length m-p)
	tauq1: []T, // Pre-allocated tau for Q1 (length q)
	tauq2: []T, // Pre-allocated tau for Q2 (length m-q)
	work: []T, // Pre-allocated workspace
	trans: TransposeMode = .None, // Transpose option
	signs: CSDecompositionSigns = .Positive, // Signs convention
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	// Extract dimensions
	p := X11.rows
	q := X11.cols
	m_p := X21.rows
	m := p + m_p
	m_q := X22.cols

	assert(X12.rows == p && X12.cols == m_q, "X12 has inconsistent dimensions")
	assert(X21.rows == m_p && X21.cols == q, "X21 has inconsistent dimensions")
	assert(X22.rows == m_p && X22.cols == m_q, "X22 has inconsistent dimensions")

	min_pq := min(p, q)

	assert(len(theta) >= int(min_pq), "theta array too small")
	assert(len(phi) >= int(min_pq), "phi array too small")
	assert(len(taup1) >= int(p), "taup1 array too small")
	assert(len(taup2) >= int(m_p), "taup2 array too small")
	assert(len(tauq1) >= int(q), "tauq1 array too small")
	assert(len(tauq2) >= int(m_q), "tauq2 array too small")

	trans_c := cast(u8)trans
	signs_c := cast(u8)signs

	m_val := m
	p_val := p
	q_val := q
	ldx11 := X11.ld
	ldx12 := X12.ld
	ldx21 := X21.ld
	ldx22 := X22.ld
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.sorbdb_(
			&trans_c,
			&signs_c,
			&m_val,
			&p_val,
			&q_val,
			raw_data(X11.data),
			&ldx11,
			raw_data(X12.data),
			&ldx12,
			raw_data(X21.data),
			&ldx21,
			raw_data(X22.data),
			&ldx22,
			raw_data(theta),
			raw_data(phi),
			raw_data(taup1),
			raw_data(taup2),
			raw_data(tauq1),
			raw_data(tauq2),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when T == f64 {
		lapack.dorbdb_(
			&trans_c,
			&signs_c,
			&m_val,
			&p_val,
			&q_val,
			raw_data(X11.data),
			&ldx11,
			raw_data(X12.data),
			&ldx12,
			raw_data(X21.data),
			&ldx21,
			raw_data(X22.data),
			&ldx22,
			raw_data(theta),
			raw_data(phi),
			raw_data(taup1),
			raw_data(taup2),
			raw_data(tauq1),
			raw_data(tauq2),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return info, info == 0
}

// Complex bidiagonal block diagonalization (simultaneous bidiagonalization of matrix blocks)
bidi_block_diagonalize_complex :: proc(
	X11: ^Matrix($Cmplx), // (p x q) upper-left block (overwritten)
	X12: ^Matrix(Cmplx), // (p x (m-q)) upper-right block (overwritten)
	X21: ^Matrix(Cmplx), // ((m-p) x q) lower-left block (overwritten)
	X22: ^Matrix(Cmplx), // ((m-p) x (m-q)) lower-right block (overwritten)
	theta: []$Real, // Pre-allocated real angles (length min(p,q))
	phi: []Real, // Pre-allocated real angles (length min(p,q))
	taup1: []Cmplx, // Pre-allocated tau for P1 (length p)
	taup2: []Cmplx, // Pre-allocated tau for P2 (length m-p)
	tauq1: []Cmplx, // Pre-allocated tau for Q1 (length q)
	tauq2: []Cmplx, // Pre-allocated tau for Q2 (length m-q)
	work: []Cmplx, // Pre-allocated complex workspace
	trans: TransposeMode = .None, // Transpose option
	signs: CSDecompositionSigns = .Positive, // Signs convention
) -> (
	info: Info,
	ok: bool,
) where (Cmplx == complex64 && Real == f32) || (Cmplx == complex128 && Real == f64) {
	// Extract dimensions
	p := X11.rows
	q := X11.cols
	m_p := X21.rows
	m := p + m_p
	m_q := X22.cols

	assert(X12.rows == p && X12.cols == m_q, "X12 has inconsistent dimensions")
	assert(X21.rows == m_p && X21.cols == q, "X21 has inconsistent dimensions")
	assert(X22.rows == m_p && X22.cols == m_q, "X22 has inconsistent dimensions")

	min_pq := min(p, q)

	assert(len(theta) >= int(min_pq), "theta array too small")
	assert(len(phi) >= int(min_pq), "phi array too small")
	assert(len(taup1) >= int(p), "taup1 array too small")
	assert(len(taup2) >= int(m_p), "taup2 array too small")
	assert(len(tauq1) >= int(q), "tauq1 array too small")
	assert(len(tauq2) >= int(m_q), "tauq2 array too small")

	trans_c := cast(u8)trans
	signs_c := cast(u8)signs

	m_val := m
	p_val := p
	q_val := q
	ldx11 := X11.ld
	ldx12 := X12.ld
	ldx21 := X21.ld
	ldx22 := X22.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cunbdb_(
			&trans_c,
			&signs_c,
			&m_val,
			&p_val,
			&q_val,
			raw_data(X11.data),
			&ldx11,
			raw_data(X12.data),
			&ldx12,
			raw_data(X21.data),
			&ldx21,
			raw_data(X22.data),
			&ldx22,
			raw_data(theta),
			raw_data(phi),
			raw_data(taup1),
			raw_data(taup2),
			raw_data(tauq1),
			raw_data(tauq2),
			raw_data(work),
			&lwork,
			&info,
		)
	} else when Cmplx == complex128 {
		lapack.zunbdb_(
			&trans_c,
			&signs_c,
			&m_val,
			&p_val,
			&q_val,
			raw_data(X11.data),
			&ldx11,
			raw_data(X12.data),
			&ldx12,
			raw_data(X21.data),
			&ldx21,
			raw_data(X22.data),
			&ldx22,
			raw_data(theta),
			raw_data(phi),
			raw_data(taup1),
			raw_data(taup2),
			raw_data(tauq1),
			raw_data(tauq2),
			raw_data(work),
			&lwork,
			&info,
		)
	}

	return info, info == 0
}
