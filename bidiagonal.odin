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
gebrd :: proc {
	gebrd_real,
	gebrd_c64,
	gebrd_c128,
}

// Query workspace size for GEBRD
query_workspace_gebrd :: proc($T: typeid, m, n: int) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		// For real types, query the actual workspace
		return -1, 0 // Use -1 to indicate workspace query needed
	} else {
		// For complex types
		return -1, 0 // Use -1 to indicate workspace query needed
	}
}

// Real GEBRD (general matrix to bidiagonal reduction)
gebrd_real :: proc(
	A: ^Matrix($T), // General matrix (overwritten with bidiagonal form)
	D: []T, // Pre-allocated diagonal elements (length min(m,n))
	E: []T, // Pre-allocated off-diagonal elements (length min(m,n)-1)
	tauq: []T, // Pre-allocated tau for Q (length min(m,n))
	taup: []T, // Pre-allocated tau for P (length min(m,n))
	work: []T, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Validate input sizes
	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(tauq) >= int(min_mn), "tauq array too small")
	assert(len(taup) >= int(min_mn), "taup array too small")

	// Workspace query if needed
	if len(work) == 0 {
		return Info(-1), false
	}

	when T == f32 {
		lwork := Blas_Int(len(work))
		lapack.sgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	} else when T == f64 {
		lwork := Blas_Int(len(work))
		lapack.dgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Complex64 GEBRD
gebrd_c64 :: proc(
	A: ^Matrix(complex64), // General matrix (overwritten with bidiagonal form)
	D: []f32, // Pre-allocated real diagonal elements (length min(m,n))
	E: []f32, // Pre-allocated real off-diagonal elements (length min(m,n)-1)
	tauq: []complex64, // Pre-allocated tau for Q (length min(m,n))
	taup: []complex64, // Pre-allocated tau for P (length min(m,n))
	work: []complex64, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Validate input sizes
	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(tauq) >= int(min_mn), "tauq array too small")
	assert(len(taup) >= int(min_mn), "taup array too small")

	if len(work) == 0 {
		return Info(-1), false
	}

	lwork := Blas_Int(len(work))
	lapack.cgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)

	return info, info == 0
}

// Complex128 GEBRD
gebrd_c128 :: proc(
	A: ^Matrix(complex128), // General matrix (overwritten with bidiagonal form)
	D: []f64, // Pre-allocated real diagonal elements (length min(m,n))
	E: []f64, // Pre-allocated real off-diagonal elements (length min(m,n)-1)
	tauq: []complex128, // Pre-allocated tau for Q (length min(m,n))
	taup: []complex128, // Pre-allocated tau for P (length min(m,n))
	work: []complex128, // Pre-allocated workspace
) -> (
	info: Info,
	ok: bool,
) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Validate input sizes
	assert(len(D) >= int(min_mn), "D array too small")
	assert(len(E) >= int(max(0, min_mn - 1)), "E array too small")
	assert(len(tauq) >= int(min_mn), "tauq array too small")
	assert(len(taup) >= int(min_mn), "taup array too small")

	if len(work) == 0 {
		return Info(-1), false
	}

	lwork := Blas_Int(len(work))
	lapack.zgebrd_(&m, &n, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tauq), raw_data(taup), raw_data(work), &lwork, &info)

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
gbbrd :: proc {
	gbbrd_real,
	gbbrd_c64,
	gbbrd_c128,
}

// Query workspace size for GBBRD
query_workspace_gbbrd :: proc($T: typeid, m, n, kl, ku: int) -> (work_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		return 2 * max(m, n), 0
	} else {
		return max(m, n), max(m, n)
	}
}

// Real GBBRD (banded matrix to bidiagonal reduction)
gbbrd_real :: proc(
	AB: ^BandedMatrix($T), // Banded matrix (modified on output)
	D: []T, // Pre-allocated diagonal elements (length min(m,n))
	E: []T, // Pre-allocated off-diagonal elements (length min(m,n)-1)
	Q: ^Matrix(T) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(T) = nil, // Pre-allocated P^T matrix (optional)
	C: ^Matrix(T) = nil, // Apply transformation to C (optional)
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

	// Validate sizes
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

// Complex64 GBBRD
gbbrd_c64 :: proc(
	AB: ^BandedMatrix(complex64), // Banded matrix (modified on output)
	D: []f32, // Pre-allocated real diagonal elements
	E: []f32, // Pre-allocated real off-diagonal elements
	Q: ^Matrix(complex64) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(complex64) = nil, // Pre-allocated P^T matrix (optional)
	C: ^Matrix(complex64) = nil, // Apply transformation to C (optional)
	work: []complex64, // Pre-allocated workspace
	rwork: []f32, // Pre-allocated real workspace
	job: BandedToBidiagJob = .None, // Which matrices to compute
) -> (
	info: Info,
	ok: bool,
) {
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
	q_ptr: ^complex64 = nil
	pt_ptr: ^complex64 = nil

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
	c_ptr: ^complex64 = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C matrix too small")
	}

	job_c := cast(u8)job

	lapack.cgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)

	return info, info == 0
}

// Complex128 GBBRD
gbbrd_c128 :: proc(
	AB: ^BandedMatrix(complex128), // Banded matrix (modified on output)
	D: []f64, // Pre-allocated real diagonal elements
	E: []f64, // Pre-allocated real off-diagonal elements
	Q: ^Matrix(complex128) = nil, // Pre-allocated Q matrix (optional)
	PT: ^Matrix(complex128) = nil, // Pre-allocated P^T matrix (optional)
	C: ^Matrix(complex128) = nil, // Apply transformation to C (optional)
	work: []complex128, // Pre-allocated workspace
	rwork: []f64, // Pre-allocated real workspace
	job: BandedToBidiagJob = .None, // Which matrices to compute
) -> (
	info: Info,
	ok: bool,
) {
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
	q_ptr: ^complex128 = nil
	pt_ptr: ^complex128 = nil

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
	c_ptr: ^complex128 = nil
	if C != nil {
		ncc = C.cols
		ldc = C.ld
		c_ptr = raw_data(C.data)
		assert(C.rows >= n, "C matrix too small")
	}

	job_c := cast(u8)job

	lapack.zgbbrd_(&job_c, &m, &n, &ncc, &kl, &ku, raw_data(AB.data), &ldab, raw_data(D), raw_data(E), q_ptr, &ldq, pt_ptr, &ldpt, c_ptr, &ldc, raw_data(work), raw_data(rwork), &info)

	return info, info == 0
}

// ===================================================================================
// BIDIAGONAL BLOCK DIAGONALIZATION (CS Decomposition)
// ===================================================================================

// Compute bidiagonal block diagonalization (real variants)
// Simultaneously bidiagonalizes the blocks of a partitioned matrix X
orbdb :: proc {
	orbdb_real,
}

// Query workspace size for ORBDB
query_workspace_orbdb :: proc($T: typeid, m, p, q: int) -> (work_size: int) where is_float(T) {
	// Workspace query returns size via work_query
	return -1 // Indicates workspace query needed
}

// Real ORBDB (simultaneous bidiagonalization of matrix blocks)
orbdb_real :: proc(
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

	// Validate block dimensions
	assert(X12.rows == p && X12.cols == m_q, "X12 has inconsistent dimensions")
	assert(X21.rows == m_p && X21.cols == q, "X21 has inconsistent dimensions")
	assert(X22.rows == m_p && X22.cols == m_q, "X22 has inconsistent dimensions")

	min_pq := min(p, q)

	// Validate output array sizes
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
			1,
			1,
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
			1,
			1,
		)
	}

	return info, info == 0
}

// Compute bidiagonal block diagonalization (complex variants)
// Simultaneously bidiagonalizes the blocks of a partitioned complex matrix X
unbdb :: proc {
	unbdb_complex,
}

// Query workspace size for UNBDB
query_workspace_unbdb :: proc($Cmplx: typeid, m, p, q: int) -> (work_size: int) where is_complex(Cmplx) {
	// Workspace query returns size via work_query
	return -1 // Indicates workspace query needed
}

// Complex UNBDB (simultaneous bidiagonalization of matrix blocks)
unbdb_complex :: proc(
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
	trans: MatrixTranspose = .None, // Transpose option
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

	// Validate block dimensions
	assert(X12.rows == p && X12.cols == m_q, "X12 has inconsistent dimensions")
	assert(X21.rows == m_p && X21.cols == q, "X21 has inconsistent dimensions")
	assert(X22.rows == m_p && X22.cols == m_q, "X22 has inconsistent dimensions")

	min_pq := min(p, q)

	// Validate output array sizes
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
			1,
			1,
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
			1,
			1,
		)
	}

	return info, info == 0
}
