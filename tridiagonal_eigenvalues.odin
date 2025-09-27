package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC SOLVE WITH FACTORIZATION
// ============================================================================
// Solves A*X = B using the factorization from sptrf
m_solve_packed_symmetric_factorized :: proc {
	m_solve_packed_symmetric_factorized_f32_c64,
	m_solve_packed_symmetric_factorized_f64_c128,
}

m_compute_tridiagonal_eigenvalues_dc :: proc {
	m_compute_tridiagonal_eigenvalues_dc_f32_c64,
	m_compute_tridiagonal_eigenvalues_dc_f64_c128,
}


// Solve packed symmetric system using factorization for f32/c64
m_solve_packed_symmetric_factorized_f32_c64 :: proc(
	ap: []$T, // Factored packed matrix from sptrf
	ipiv: []Blas_Int, // Pivot indices from sptrf
	B: ^Matrix(T), // Right-hand side (overwritten with solution)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	n := B.rows
	nrhs := B.cols
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(B.ld)

	when T == f32 {
		lapack.ssptrs_(uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(ipiv), raw_data(B.data), &ldb, &info, len(uplo_c))
	} else when T == complex64 {
		lapack.csptrs_(uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(ipiv), raw_data(B.data), &ldb, &info, len(uplo_c))
	}

	ok = info == 0
	return info, ok
}

// Solve packed symmetric system using factorization for f64/c128
m_solve_packed_symmetric_factorized_f64_c128 :: proc(
	ap: []$T, // Factored packed matrix from sptrf
	ipiv: []Blas_Int, // Pivot indices from sptrf
	B: ^Matrix(T), // Right-hand side (overwritten with solution)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	n := B.rows
	nrhs := B.cols
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_cstring(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(B.ld)

	when T == f64 {
		lapack.dsptrs_(uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(ipiv), raw_data(B.data), &ldb, &info, len(uplo_c))
	} else when T == complex128 {
		lapack.zsptrs_(uplo_c, &n_int, &nrhs_int, raw_data(ap), raw_data(ipiv), raw_data(B.data), &ldb, &info, len(uplo_c))
	}

	ok = info == 0
	return info, ok
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE COMPUTATION - BISECTION
// ============================================================================
// Computes selected eigenvalues of a tridiagonal matrix using bisection

// Eigenvalue ordering option
EigenvalueOrder :: enum {
	BLOCKS, // 'B' - Eigenvalues ordered by blocks
	ENTIRE, // 'E' - Entire matrix eigenvalues in order
}

// Query workspace for bisection eigenvalue computation (STEBZ)
query_workspace_tridiagonal_bisection_eigenvalues :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) {
	// STEBZ requires:
	// work: 4*n
	// iwork: 3*n
	return 4 * n, 3 * n
}

// Compute eigenvalues using bisection for f32/f64 (STEBZ)
m_compute_eigenvalues_tridiagonal_eigenvalues_bisection :: proc(
	d: []$T, // Diagonal elements
	e: []T, // Off-diagonal elements
	w: []T, // Pre-allocated eigenvalues output
	iblock: []Blas_Int, // Pre-allocated block indices
	isplit: []Blas_Int, // Pre-allocated split points
	work: []T, // Pre-allocated workspace (4*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (3*n)
	range := EigenRangeOption.ALL,
	order := EigenvalueOrder.ENTIRE,
	vl: T, // Upper bound (if range == VALUE)
	il: int = 1, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: T, // Absolute tolerance
) -> (
	num_found: int,
	num_splits: int,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= n, "Eigenvalue array too small")
	assert(len(iblock) >= n, "Block indices array too small")
	assert(len(isplit) >= n, "Split points array too small")
	assert(len(work) >= 4 * n, "Insufficient workspace")
	assert(len(iwork) >= 3 * n, "Insufficient integer workspace")

	range_c := eigen_range_to_cstring(range)
	order_c := order == .ENTIRE ? cstring("E") : cstring("B")

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	nsplit: Blas_Int

	// Call LAPACK
	when T == f32 {
		lapack.sstebz_(
			range_c,
			order_c,
			&n_int,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			raw_data(d),
			raw_data(e),
			&m,
			&nsplit,
			raw_data(w),
			raw_data(iblock),
			raw_data(isplit),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(range_c),
			len(order_c),
		)
	} else when T == f64 {
		lapack.dstebz_(
			range_c,
			order_c,
			&n_int,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			raw_data(d),
			raw_data(e),
			&m,
			&nsplit,
			raw_data(w),
			raw_data(iblock),
			raw_data(isplit),
			raw_data(work),
			raw_data(iwork),
			&info,
			len(range_c),
			len(order_c),
		)
	}

	num_found = int(m)
	num_splits = int(nsplit)
	ok = info == 0
	return num_found, num_splits, info, ok
}


// ============================================================================
// TRIDIAGONAL EIGENVALUE COMPUTATION - DIVIDE AND CONQUER
// ============================================================================

// Query workspace for divide-and-conquer eigenvalue computation (STEDC)
query_workspace_tridiagonal_eigenvalues_dc :: proc($T: typeid, n: int, compz: CompzOption) -> (work_size: int, rwork_size: int, iwork_size: int) {
	n_int := Blas_Int(n)
	compz_c := compz_to_char(compz)

	when T == f32 {
		work_query: f32
		iwork_query: Blas_Int
		lwork := Blas_Int(-1)
		liwork := Blas_Int(-1)
		info: Info

		lapack.sstedc_(
			compz_c,
			&n_int,
			nil, // d
			nil, // e
			nil, // z
			&n_int, // ldz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			len(compz_c),
		)

		return int(work_query), 0, int(iwork_query)
	} else when T == f64 {
		work_query: f64
		iwork_query: Blas_Int
		lwork := Blas_Int(-1)
		liwork := Blas_Int(-1)
		info: Info

		lapack.dstedc_(
			compz_c,
			&n_int,
			nil, // d
			nil, // e
			nil, // z
			&n_int, // ldz
			&work_query,
			&lwork,
			&iwork_query,
			&liwork,
			&info,
			len(compz_c),
		)

		return int(work_query), 0, int(iwork_query)
	} else when T == complex64 {
		work_query: complex64
		rwork_query: f32
		iwork_query: Blas_Int
		lwork := Blas_Int(-1)
		lrwork := Blas_Int(-1)
		liwork := Blas_Int(-1)
		info: Info

		lapack.cstedc_(
			compz_c,
			&n_int,
			nil, // d
			nil, // e
			nil, // z
			&n_int, // ldz
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
			len(compz_c),
		)

		return int(real(work_query)), int(rwork_query), int(iwork_query)
	} else when T == complex128 {
		work_query: complex128
		rwork_query: f64
		iwork_query: Blas_Int
		lwork := Blas_Int(-1)
		lrwork := Blas_Int(-1)
		liwork := Blas_Int(-1)
		info: Info

		lapack.zstedc_(
			compz_c,
			&n_int,
			nil, // d
			nil, // e
			nil, // z
			&n_int, // ldz
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&iwork_query,
			&liwork,
			&info,
			len(compz_c),
		)

		return int(real(work_query)), int(rwork_query), int(iwork_query)
	}
}

// Compute eigenvalues/eigenvectors using divide-and-conquer for f32/c64
m_compute_tridiagonal_eigenvalues_dc_f32_c64 :: proc(
	d: []f32, // Diagonal (modified to eigenvalues on output)
	e: []f32, // Off-diagonal (destroyed)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []T, // Pre-allocated workspace
	rwork: []f32 = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	compz := CompzOption.None,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_c := compz_to_char(compz)
	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: rawptr = nil
	if compz != .None && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f32 {
		assert(len(work) > 0, "Workspace required")
		assert(len(iwork) > 0, "Integer workspace required")

		lapack.sstedc_(compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, len(compz_c))
	} else when T == complex64 {
		assert(len(work) > 0, "Workspace required")
		assert(len(rwork) > 0, "Real workspace required")
		assert(len(iwork) > 0, "Integer workspace required")

		lrwork := Blas_Int(len(rwork))

		lapack.cstedc_(compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info, len(compz_c))
	}

	ok = info == 0
	return info, ok
}

// Compute eigenvalues/eigenvectors using divide-and-conquer for f64/c128
m_compute_tridiagonal_eigenvalues_dc_f64_c128 :: proc(
	d: []f64, // Diagonal (modified to eigenvalues on output)
	e: []f64, // Off-diagonal (destroyed)
	Z: ^Matrix($T) = nil, // Eigenvector matrix (optional)
	work: []T, // Pre-allocated workspace
	rwork: []f64 = nil, // Pre-allocated real workspace (complex only)
	iwork: []Blas_Int, // Pre-allocated integer workspace
	compz := CompzOption.None,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	n := len(d)
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_c := compz_to_char(compz)
	n_int := Blas_Int(n)
	lwork := Blas_Int(len(work))
	liwork := Blas_Int(len(iwork))

	// Handle eigenvector matrix
	ldz := Blas_Int(1)
	z_ptr: rawptr = nil
	if compz != .None && Z != nil {
		assert(Z.rows >= n && Z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(Z.ld)
		z_ptr = raw_data(Z.data)
	}

	when T == f64 {
		assert(len(work) > 0, "Workspace required")
		assert(len(iwork) > 0, "Integer workspace required")

		lapack.dstedc_(compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(iwork), &liwork, &info, len(compz_c))
	} else when T == complex128 {
		assert(len(work) > 0, "Workspace required")
		assert(len(rwork) > 0, "Real workspace required")
		assert(len(iwork) > 0, "Integer workspace required")

		lrwork := Blas_Int(len(rwork))

		lapack.zstedc_(compz_c, &n_int, raw_data(d), raw_data(e), z_ptr, &ldz, raw_data(work), &lwork, raw_data(rwork), &lrwork, raw_data(iwork), &liwork, &info, len(compz_c))
	}

	ok = info == 0
	return info, ok
}
