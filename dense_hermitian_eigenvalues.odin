package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// HERMITIAN EIGENVALUE SOLVERS - NON-ALLOCATING API
// ============================================================================
// Complex Hermitian matrices: all eigenvalues are real
// Using standard QR algorithm and more advanced methods

// Query workspace for Hermitian eigenvalue computation (QR algorithm)
query_workspace_compute_hermitian_eigenvalues :: proc {
	query_workspace_compute_hermitian_eigenvalues_complex,
}

query_workspace_compute_hermitian_eigenvalues_complex :: proc($Cmplx: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, rwork_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		rwork := make([]f32, max(1, 3 * n - 2)) // Real workspace
		defer delete(rwork)
		lapack.cheev_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			raw_data(rwork),
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork := make([]f64, max(1, 3 * n - 2)) // Real workspace
		defer delete(rwork)
		lapack.zheev_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			raw_data(rwork),
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
	}

	return work_size, rwork_size
}

// Compute eigenvalues and eigenvectors of Hermitian matrix using QR algorithm
compute_hermitian_eigenvalues :: proc {
	compute_hermitian_eigenvalues_complex,
}

compute_hermitian_eigenvalues_complex :: proc(
	A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
	W: []$Real, // Output eigenvalues (length n, real values)
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
	jobz: EigenJobOption,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Complex workspace required")
	assert(len(rwork) >= max(1, 3 * n - 2), "Real workspace too small")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cheev_(&jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zheev_(&jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return info, info == 0
}

// ============================================================================
// DIVIDE AND CONQUER ALGORITHM - More efficient for large matrices
// ============================================================================

// Query workspace for Hermitian eigenvalue computation (Divide and Conquer)
query_workspace_compute_hermitian_eigenvalues_dc :: proc {
	query_workspace_compute_hermitian_eigenvalues_dc_complex,
}

query_workspace_compute_hermitian_eigenvalues_dc_complex :: proc($Cmplx: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, rwork_size: int, iwork_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	lrwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		rwork_query: f32
		iwork_query: Blas_Int
		lapack.cheevd_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
		iwork_size = int(iwork_query)
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork_query: f64
		iwork_query: Blas_Int
		lapack.zheevd_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
		iwork_size = int(iwork_query)
	}

	return work_size, rwork_size, iwork_size
}

// Compute eigenvalues and eigenvectors using Divide and Conquer algorithm
compute_hermitian_eigenvalues_dc :: proc {
	compute_hermitian_eigenvalues_dc_complex,
}

compute_hermitian_eigenvalues_dc_complex :: proc(
	A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
	W: []$Real, // Output eigenvalues (length n, real values)
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	jobz: EigenJobOption,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Complex workspace required")
	assert(len(rwork) > 0, "Real workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))

	when Cmplx == complex64 {
		lapack.cheevd_(&jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	} else when Cmplx == complex128 {
		lapack.zheevd_(&jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// RELATIVELY ROBUST REPRESENTATIONS (RRR) - Most advanced algorithm
// ============================================================================

// Query workspace for Hermitian eigenvalue computation (RRR algorithm)
query_workspace_compute_hermitian_eigenvalues_rrr :: proc {
	query_workspace_compute_hermitian_eigenvalues_rrr_complex,
}

query_workspace_compute_hermitian_eigenvalues_rrr_complex :: proc($Cmplx: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, rwork_size: int, iwork_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	range_c: u8 = 'A' // All eigenvalues
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	ldz := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	lrwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info

	// Dummy values for query
	vl: f64 = 0
	vu: f64 = 0
	il: Blas_Int = 1
	iu: Blas_Int = n_int
	abstol: f64 = 0
	m: Blas_Int = 0

	when Cmplx == complex64 {
		work_query: complex64
		rwork_query: f32
		iwork_query: Blas_Int
		lapack.cheevr_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil,
			&lda, // A
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz,
			nil, // W, Z, ISUPPZ
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
		iwork_size = int(iwork_query)
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork_query: f64
		iwork_query: Blas_Int
		lapack.zheevr_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil,
			&lda, // A
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz,
			nil, // W, Z, ISUPPZ
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
		iwork_size = int(iwork_query)
	}

	return work_size, rwork_size, iwork_size
}

// Compute eigenvalues and eigenvectors using RRR algorithm (most robust)
compute_hermitian_eigenvalues_rrr :: proc {
	compute_hermitian_eigenvalues_rrr_complex,
}

compute_hermitian_eigenvalues_rrr_complex :: proc(
	A: ^Matrix($Cmplx), // Input Hermitian matrix (preserved)
	W: []$Real, // Output eigenvalues (length n, but m eigenvalues found)
	Z: ^Matrix(Cmplx), // Output eigenvectors (n x m)
	ISUPPZ: []Blas_Int, // Support indices (length 2*max(1,m))
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	vl: Real, // Value range (if range == 'V')
	vu: Real,
	il: int, // Index range (if range == 'I'), 1-based
	iu: int,
	abstol: Real, // Absolute tolerance for eigenvalues
	range: EigenRangeOption,
	jobz: EigenJobOption,
	uplo := MatrixRegion.Upper,
) -> (
	m: int,
	info: Info,// Number of eigenvalues found
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Complex workspace required")
	assert(len(rwork) > 0, "Real workspace required")
	assert(len(iwork) > 0, "Integer workspace required")

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	ldz := Z.ld
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int

	when Cmplx == complex64 {
		lapack.cheevr_(&jobz_c, &range_c, &uplo_c, &n_int, raw_data(A.data), &lda, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(ISUPPZ), raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	} else when Cmplx == complex128 {
		lapack.zheevr_(&jobz_c, &range_c, &uplo_c, &n_int, raw_data(A.data), &lda, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(ISUPPZ), raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	}

	return int(m_int), info, info == 0
}

// ============================================================================
// SELECTED EIGENVALUES - Legacy Expert Driver (HEEVX)
// ============================================================================

// Query workspace for Hermitian eigenvalue computation (Expert driver)
query_workspace_compute_hermitian_eigenvalues_expert :: proc {
	query_workspace_compute_hermitian_eigenvalues_expert_complex,
}

query_workspace_compute_hermitian_eigenvalues_expert_complex :: proc($Cmplx: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, rwork_size: int, iwork_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	range_c: u8 = 'A' // All eigenvalues
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	ldz := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	// Dummy values for query
	vl: f64 = 0
	vu: f64 = 0
	il: Blas_Int = 1
	iu: Blas_Int = n_int
	abstol: f64 = 0
	m: Blas_Int = 0

	when Cmplx == complex64 {
		work_query: complex64
		rwork := make([]f32, 7 * n) // Fixed size for HEEVX
		defer delete(rwork)
		lapack.cheevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil,
			&lda, // A
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz, // W, Z
			&work_query,
			&lwork,
			raw_data(rwork),
			nil,
			nil, // rwork, iwork, IFAIL
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
		iwork_size = 5 * n // Fixed size for HEEVX
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork := make([]f64, 7 * n) // Fixed size for HEEVX
		defer delete(rwork)
		lapack.zheevx_(
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil,
			&lda, // A
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz, // W, Z
			&work_query,
			&lwork,
			raw_data(rwork),
			nil,
			nil, // rwork, iwork, IFAIL
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
		iwork_size = 5 * n // Fixed size for HEEVX
	}

	return work_size, rwork_size, iwork_size
}

// Compute selected eigenvalues and eigenvectors using expert driver
compute_hermitian_eigenvalues_expert :: proc {
	compute_hermitian_eigenvalues_expert_complex,
}

compute_hermitian_eigenvalues_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed)
	W: []$Real, // Output eigenvalues (length n, but m eigenvalues found)
	Z: ^Matrix(Cmplx), // Output eigenvectors (n x m)
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	IFAIL: []Blas_Int, // Failed convergence indices (length n)
	vl: Real, // Value range (if range == 'V')
	vu: Real,
	il: int, // Index range (if range == 'I'), 1-based
	iu: int,
	abstol: Real, // Absolute tolerance for eigenvalues
	range: EigenRangeOption,
	jobz: EigenJobOption,
	uplo := MatrixRegion.Upper,
) -> (
	m: int,
	info: Info,// Number of eigenvalues found
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Complex workspace required")
	assert(len(rwork) >= 7 * n, "Real workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(IFAIL) >= n, "IFAIL array too small")

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	ldz := Z.ld
	lwork := Blas_Int(len(work))
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	m_int: Blas_Int

	when Cmplx == complex64 {
		lapack.cheevx_(&jobz_c, &range_c, &uplo_c, &n_int, raw_data(A.data), &lda, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), raw_data(IFAIL), &info)
	} else when Cmplx == complex128 {
		lapack.zheevx_(&jobz_c, &range_c, &uplo_c, &n_int, raw_data(A.data), &lda, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), raw_data(IFAIL), &info)
	}

	return int(m_int), info, info == 0
}

// ============================================================================
// TWO-STAGE ALGORITHMS - For very large matrices
// ============================================================================

// Query workspace for two-stage Hermitian eigenvalue computation
query_workspace_compute_hermitian_eigenvalues_2stage :: proc {
	query_workspace_compute_hermitian_eigenvalues_2stage_complex,
}

query_workspace_compute_hermitian_eigenvalues_2stage_complex :: proc($Cmplx: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, rwork_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		rwork := make([]f32, max(1, 3 * n - 2)) // Real workspace
		defer delete(rwork)
		lapack.cheev_2stage_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			raw_data(rwork),
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork := make([]f64, max(1, 3 * n - 2)) // Real workspace
		defer delete(rwork)
		lapack.zheev_2stage_(
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // a
			&lda,
			nil, // w
			&work_query,
			&lwork,
			raw_data(rwork),
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
	}

	return work_size, rwork_size
}

// Compute eigenvalues and eigenvectors using two-stage algorithm
compute_hermitian_eigenvalues_2stage :: proc {
	compute_hermitian_eigenvalues_2stage_complex,
}

compute_hermitian_eigenvalues_2stage_complex :: proc(
	A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
	W: []$Real, // Output eigenvalues (length n, real values)
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
	jobz: EigenJobOption,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Complex workspace required")
	assert(len(rwork) >= max(1, 3 * n - 2), "Real workspace too small")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.cheev_2stage_(&jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zheev_2stage_(&jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return info, info == 0
}

// ============================================================================
// GENERALIZED HERMITIAN EIGENVALUE PROBLEMS
// ============================================================================
// Solve generalized eigenvalue problems of the form:
//   itype=1: A*x = lambda*B*x
//   itype=2: A*B*x = lambda*x
//   itype=3: B*A*x = lambda*x
// where A is Hermitian and B is Hermitian positive definite

// Query workspace for generalized Hermitian eigenvalue computation (QR algorithm)
query_workspace_compute_generalized_hermitian_eigenvalues :: proc {
	query_workspace_compute_generalized_hermitian_eigenvalues_complex,
}

query_workspace_compute_generalized_hermitian_eigenvalues_complex :: proc($Cmplx: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, rwork_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info
	itype := Blas_Int(1)

	when Cmplx == complex64 {
		work_query: complex64
		rwork := make([]f32, max(1, 3 * n - 2)) // Real workspace
		defer delete(rwork)
		lapack.chegv_(
			&itype,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // A
			&lda,
			nil, // B
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			raw_data(rwork),
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork := make([]f64, max(1, 3 * n - 2)) // Real workspace
		defer delete(rwork)
		lapack.zhegv_(
			&itype,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // A
			&lda,
			nil, // B
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			raw_data(rwork),
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
	}

	return work_size, rwork_size
}

// Compute generalized eigenvalues and eigenvectors using QR algorithm
compute_generalized_hermitian_eigenvalues :: proc {
	compute_generalized_hermitian_eigenvalues_complex,
}

compute_generalized_hermitian_eigenvalues_complex :: proc(
	A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
	B: ^Matrix(Cmplx), // Input Hermitian positive definite matrix (destroyed)
	W: []$Real, // Output eigenvalues (length n, real values)
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	jobz: EigenJobOption,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == B.cols, "Matrix B must be square")
	assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Complex workspace required")
	assert(len(rwork) >= max(1, 3 * n - 2), "Real workspace too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))
	itype_blas := Blas_Int(itype)

	when Cmplx == complex64 {
		lapack.chegv_(&itype_blas, &jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &info)
	} else when Cmplx == complex128 {
		lapack.zhegv_(&itype_blas, &jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &info)
	}

	return info, info == 0
}

// ============================================================================
// GENERALIZED DIVIDE AND CONQUER ALGORITHM
// ============================================================================

// Query workspace for generalized Hermitian eigenvalue computation (Divide and Conquer)
query_workspace_compute_generalized_hermitian_eigenvalues_dc :: proc {
	query_workspace_compute_generalized_hermitian_eigenvalues_dc_complex,
}

query_workspace_compute_generalized_hermitian_eigenvalues_dc_complex :: proc($Cmplx: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, rwork_size: int, iwork_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	lrwork := QUERY_WORKSPACE
	liwork := QUERY_WORKSPACE
	info: Info
	itype := Blas_Int(1)

	when Cmplx == complex64 {
		work_query: complex64
		rwork_query: f32
		iwork_query: Blas_Int
		lapack.chegvd_(
			&itype,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // A
			&lda,
			nil, // B
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
		iwork_size = int(iwork_query)
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork_query: f64
		iwork_query: Blas_Int
		lapack.zhegvd_(
			&itype,
			&jobz_c,
			&uplo_c,
			&n_int,
			nil, // A
			&lda,
			nil, // B
			&ldb,
			nil, // w
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
		iwork_size = int(iwork_query)
	}

	return work_size, rwork_size, iwork_size
}

// Compute generalized eigenvalues and eigenvectors using Divide and Conquer algorithm
compute_generalized_hermitian_eigenvalues_dc :: proc {
	compute_generalized_hermitian_eigenvalues_dc_complex,
}

compute_generalized_hermitian_eigenvalues_dc_complex :: proc(
	A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed/overwritten with eigenvectors if requested)
	B: ^Matrix(Cmplx), // Input Hermitian positive definite matrix (destroyed)
	W: []$Real, // Output eigenvalues (length n, real values)
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	jobz: EigenJobOption,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == B.cols, "Matrix B must be square")
	assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Complex workspace required")
	assert(len(rwork) > 0, "Real workspace required")
	assert(len(iwork) > 0, "Integer workspace required")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	jobz_c := cast(u8)jobz
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	ldb := B.ld
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))
	liwork := Blas_Int(len(iwork))
	itype_blas := Blas_Int(itype)

	when Cmplx == complex64 {
		lapack.chegvd_(&itype_blas, &jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	} else when Cmplx == complex128 {
		lapack.zhegvd_(&itype_blas, &jobz_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, raw_data(W), raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info)
	}

	return info, info == 0
}

// ============================================================================
// GENERALIZED EXPERT DRIVER (HEGVX)
// ============================================================================

// Query workspace for generalized Hermitian eigenvalue computation (Expert driver)
query_workspace_compute_generalized_hermitian_eigenvalues_expert :: proc {
	query_workspace_compute_generalized_hermitian_eigenvalues_expert_complex,
}

query_workspace_compute_generalized_hermitian_eigenvalues_expert_complex :: proc($Cmplx: typeid, n: int, jobz: EigenJobOption) -> (work_size: int, rwork_size: int, iwork_size: int) where is_complex(Cmplx) {
	// Query LAPACK for optimal workspace size
	jobz_c := cast(u8)jobz
	range_c: u8 = 'A' // All eigenvalues
	uplo_c: u8 = 'U' // Default to upper
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	ldz := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info
	itype := Blas_Int(1)

	// Dummy values for query
	vl: f64 = 0
	vu: f64 = 0
	il: Blas_Int = 1
	iu: Blas_Int = n_int
	abstol: f64 = 0
	m: Blas_Int = 0

	when Cmplx == complex64 {
		work_query: complex64
		rwork := make([]f32, 7 * n) // Fixed size for HEGVX
		defer delete(rwork)
		lapack.chegvx_(
			&itype,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil,
			&lda, // A
			nil,
			&ldb, // B
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz, // W, Z
			&work_query,
			&lwork,
			raw_data(rwork),
			nil,
			nil, // rwork, iwork, IFAIL
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
		iwork_size = 5 * n // Fixed size for HEGVX
	} else when Cmplx == complex128 {
		work_query: complex128
		rwork := make([]f64, 7 * n) // Fixed size for HEGVX
		defer delete(rwork)
		lapack.zhegvx_(
			&itype,
			&jobz_c,
			&range_c,
			&uplo_c,
			&n_int,
			nil,
			&lda, // A
			nil,
			&ldb, // B
			&vl,
			&vu,
			&il,
			&iu,
			&abstol,
			&m,
			nil,
			nil,
			&ldz, // W, Z
			&work_query,
			&lwork,
			raw_data(rwork),
			nil,
			nil, // rwork, iwork, IFAIL
			&info,
		)
		work_size = int(real(work_query))
		rwork_size = len(rwork)
		iwork_size = 5 * n // Fixed size for HEGVX
	}

	return work_size, rwork_size, iwork_size
}

// Compute selected generalized eigenvalues and eigenvectors using expert driver
compute_generalized_hermitian_eigenvalues_expert :: proc {
	compute_generalized_hermitian_eigenvalues_expert_complex,
}

compute_generalized_hermitian_eigenvalues_expert_complex :: proc(
	A: ^Matrix($Cmplx), // Input Hermitian matrix (destroyed)
	B: ^Matrix(Cmplx), // Input Hermitian positive definite matrix (destroyed)
	W: []$Real, // Output eigenvalues (length n, but m eigenvalues found)
	Z: ^Matrix(Cmplx), // Output eigenvectors (n x m)
	work: []Cmplx, // Pre-allocated complex workspace
	rwork: []Real, // Pre-allocated real workspace
	iwork: []Blas_Int, // Pre-allocated integer workspace
	IFAIL: []Blas_Int, // Failed convergence indices (length n)
	vl: Real, // Value range lower bound (if range == 'V')
	vu: Real, // Value range upper bound (if range == 'V')
	il: int = 1, // Index range lower bound (if range == 'I'), 1-based
	iu: int = 1, // Index range upper bound (if range == 'I'), 1-based
	abstol: Real, // Absolute tolerance for eigenvalues
	itype: int = 1, // Problem type (1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x)
	range: EigenRangeOption,
	jobz: EigenJobOption,
	uplo := MatrixRegion.Upper,
) -> (
	m: int,
	info: Info,// Number of eigenvalues found
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == B.cols, "Matrix B must be square")
	assert(A.rows == B.rows, "Matrices A and B must have same dimensions")
	assert(len(W) >= n, "Eigenvalue array too small")
	assert(len(work) > 0, "Complex workspace required")
	assert(len(rwork) >= 7 * n, "Real workspace too small")
	assert(len(iwork) >= 5 * n, "Integer workspace too small")
	assert(len(IFAIL) >= n, "IFAIL array too small")
	assert(itype >= 1 && itype <= 3, "Invalid problem type")

	jobz_c := cast(u8)jobz
	range_c := cast(u8)range
	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	ldb := B.ld
	ldz := Z.ld
	lwork := Blas_Int(len(work))
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	itype_blas := Blas_Int(itype)
	m_int: Blas_Int

	when Cmplx == complex64 {
		lapack.chegvx_(&itype_blas, &jobz_c, &range_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), raw_data(IFAIL), &info)
	} else when Cmplx == complex128 {
		lapack.zhegvx_(&itype_blas, &jobz_c, &range_c, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, &vl, &vu, &il_int, &iu_int, &abstol, &m_int, raw_data(W), raw_data(Z.data), &ldz, raw_data(work), &lwork, raw_data(rwork), raw_data(iwork), raw_data(IFAIL), &info)
	}

	return int(m_int), info, info == 0
}
// ============================================================================
// HERMITIAN GENERALIZED EIGENVALUE REDUCTION (HEGST)
// ============================================================================
// Reduce generalized Hermitian eigenvalue problem to standard form
// Transforms A*x = lambda*B*x into C*y = lambda*y where C = inv(U^H)*A*inv(U) or inv(L)*A*inv(L^H)

// Reduce generalized Hermitian eigenvalue problem to standard form (complex)
reduce_generalized_hermitian_to_standard_form_complex :: proc(
	A: ^Matrix($Cmplx), // Input/output: Hermitian matrix A, overwritten with transformed matrix
	B: ^Matrix(Cmplx), // Input: Cholesky factor of B (from potrf)
	itype: int = 1, // Problem type: 1: A*x=lambda*B*x, 2: A*B*x=lambda*x, 3: B*A*x=lambda*x
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix A must be square")
	assert(B.rows == B.cols, "Matrix B must be square")
	assert(A.rows == B.rows, "Matrices A and B must have same dimension")
	assert(itype >= 1 && itype <= 3, "Invalid problem type (must be 1, 2, or 3)")

	uplo_c := cast(u8)uplo
	itype_blas := Blas_Int(itype)
	n_int := Blas_Int(n)
	lda := A.ld
	ldb := B.ld

	when Cmplx == complex64 {
		lapack.chegst_(&itype_blas, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	} else when Cmplx == complex128 {
		lapack.zhegst_(&itype_blas, &uplo_c, &n_int, raw_data(A.data), &lda, raw_data(B.data), &ldb, &info)
	}

	return info, info == 0
}

reduce_generalized_hermitian_to_standard_form :: proc {
	reduce_generalized_hermitian_to_standard_form_complex,
}

// ============================================================================
// HERMITIAN TRIDIAGONAL REDUCTION (HETRD)
// ============================================================================
// Reduce Hermitian matrix to tridiagonal form using unitary similarity transformations

// Query workspace for Hermitian tridiagonal reduction
query_workspace_reduce_hermitian_to_tridiagonal :: proc {
	query_workspace_reduce_hermitian_to_tridiagonal_complex,
}

query_workspace_reduce_hermitian_to_tridiagonal_complex :: proc($Cmplx: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_complex(Cmplx) {
	n_int := Blas_Int(n)
	uplo_c := cast(u8)uplo
	lda := Blas_Int(max(1, n))
	lwork := QUERY_WORKSPACE
	info: Info

	when Cmplx == complex64 {
		work_query: complex64
		lapack.chetrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	} else when Cmplx == complex128 {
		work_query: complex128
		lapack.zhetrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info)
		work_size = int(real(work_query))
	}

	return work_size
}

// Reduce Hermitian matrix to tridiagonal form (complex)
reduce_hermitian_to_tridiagonal_complex :: proc(
	A: ^Matrix($Cmplx), // Input/output: Hermitian matrix, overwritten with Q
	D: []$Real, // Output: Diagonal elements (length n, real values)
	E: []Real, // Output: Off-diagonal elements (length n-1, real values)
	tau: []Cmplx, // Output: Scalar factors of elementary reflectors (length n-1)
	work: []Cmplx, // Pre-allocated workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(Cmplx),
	Real == real_type_of(Cmplx) {
	n := A.rows
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(D) >= n, "Diagonal array too small")
	assert(len(E) >= n - 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1, "Tau array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	lda := A.ld
	lwork := Blas_Int(len(work))

	when Cmplx == complex64 {
		lapack.chetrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tau), raw_data(work), &lwork, &info)
	} else when Cmplx == complex128 {
		lapack.zhetrd_(&uplo_c, &n_int, raw_data(A.data), &lda, raw_data(D), raw_data(E), raw_data(tau), raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

reduce_hermitian_to_tridiagonal :: proc {
	reduce_hermitian_to_tridiagonal_complex,
}
