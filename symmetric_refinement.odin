package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC ITERATIVE REFINEMENT
// ============================================================================
// Improves the computed solution to a symmetric system of linear equations

// Query workspace size for symmetric iterative refinement
query_workspace_refine_symmetric :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		// Real types: work = 3*n, iwork = n
		work_size = 3 * n
		iwork_size = n
		rwork_size = 0
	} else {
		// Complex types: work = 2*n, rwork = n
		work_size = 2 * n
		iwork_size = 0
		rwork_size = n
	}
	return
}

// Query result sizes for symmetric iterative refinement
query_result_sizes_refine_symmetric :: proc(nrhs: int) -> (ferr_size: int, berr_size: int) {
	return nrhs, nrhs
}

// Symmetric iterative refinement for f32/f64
m_refine_symmetric_f32_f64 :: proc(
	a: ^Matrix($T), // Original matrix A
	af: ^Matrix(T), // Factored matrix from factorization
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: ^Matrix(T), // Right-hand side matrix
	x: ^Matrix(T), // Solution matrix (refined on output)
	ferr: []T, // Forward error bounds (size nrhs)
	berr: []T, // Backward error bounds (size nrhs)
	work: []T, // Workspace (size 3*n)
	iwork: []Blas_Int, // Integer workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := a.ld
	ldaf := af.ld
	ldb := b.ld
	ldx := x.ld

	when T == f32 {
		lapack.ssyrfs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, af.data, &ldaf, raw_data(ipiv), b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	} else when T == f64 {
		lapack.dsyrfs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, af.data, &ldaf, raw_data(ipiv), b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info)
	}

	return info, info == 0
}

// Symmetric iterative refinement for complex64/complex128
m_refine_symmetric_c64_c128 :: proc(
	a: ^Matrix($T), // Original matrix A
	af: ^Matrix(T), // Factored matrix from factorization
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: ^Matrix(T), // Right-hand side matrix
	x: ^Matrix(T), // Solution matrix (refined on output)
	ferr: []$R, // Forward error bounds (size nrhs)
	berr: []R, // Backward error bounds (size nrhs)
	work: []T, // Complex workspace (size 2*n)
	rwork: []R, // Real workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := cast(u8)uplo
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := a.ld
	ldaf := af.ld
	ldb := b.ld
	ldx := x.ld

	when T == complex64 {
		lapack.csyrfs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, af.data, &ldaf, raw_data(ipiv), b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	} else when T == complex128 {
		lapack.zsyrfs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, af.data, &ldaf, raw_data(ipiv), b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info)
	}

	return info, info == 0
}

// Procedure group for symmetric iterative refinement
m_refine_symmetric :: proc {
	m_refine_symmetric_f32_f64,
	m_refine_symmetric_c64_c128,
}


// ============================================================================
// EXTENDED SYMMETRIC ITERATIVE REFINEMENT
// ============================================================================
// Extended version with equilibration and advanced error bounds

// Query workspace size for extended symmetric iterative refinement
query_workspace_refine_symmetric_extended :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int, rwork_size: int) where is_float(T) || is_complex(T) {
	when is_float(T) {
		// Real types: work = 4*n, iwork = n
		work_size = 4 * n
		iwork_size = n
		rwork_size = 0
	} else {
		// Complex types: work = 2*n, rwork = 3*n
		work_size = 2 * n
		iwork_size = 0
		rwork_size = 3 * n
	}
	return
}

// Query result sizes for extended symmetric iterative refinement
query_result_sizes_refine_symmetric_extended :: proc(nrhs: int, n_err_bnds: int = ERROR_BOUND_TYPES) -> (berr_size: int, err_bounds_norm_size: int, err_bounds_comp_size: int) {
	return nrhs, nrhs * n_err_bnds, nrhs * n_err_bnds
}

// Extended symmetric iterative refinement for f32/f64
m_refine_symmetric_extended_f32_f64 :: proc(
	a: ^Matrix($T), // Original matrix A
	af: ^Matrix(T), // Factored matrix from factorization
	ipiv: []Blas_Int, // Pivot indices from factorization
	s: []T, // Scale factors (if equed == APPLIED)
	b: ^Matrix(T), // Right-hand side matrix
	x: ^Matrix(T), // Solution matrix (refined on output)
	berr: []T, // Backward error bounds (size nrhs)
	err_bounds_norm: []T, // Normwise error bounds (nrhs * n_err_bnds)
	err_bounds_comp: []T, // Componentwise error bounds (nrhs * n_err_bnds)
	work: []T, // Workspace (size 4*n)
	iwork: []Blas_Int, // Integer workspace (size n)
	equed := EquilibrationState.None,
	uplo := MatrixRegion.Upper,
	n_err_bnds: int = ERROR_BOUND_TYPES,
	nparams: int = 0,
	params: []T = nil,
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where is_float(T) {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bounds_norm) >= nrhs * n_err_bnds, "Normwise error bounds array too small")
	assert(len(err_bounds_comp) >= nrhs * n_err_bnds, "Componentwise error bounds array too small")
	assert(len(work) >= 4 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := cast(u8)uplo
	equed_c := cast(u8)(equed)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := a.ld
	ldaf := af.ld
	ldb := b.ld
	ldx := x.ld
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)

	when T == f32 {
		lapack.ssyrfsx_(
			&uplo_c,
			&equed_c,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			af.data,
			&ldaf,
			raw_data(ipiv),
			raw_data(s) if s != nil else nil,
			b.data,
			&ldb,
			x.data,
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bounds_norm),
			raw_data(err_bounds_comp),
			&nparams_int,
			raw_data(params) if params != nil else nil,
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	} else when T == f64 {
		lapack.dsyrfsx_(
			&uplo_c,
			&equed_c,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			af.data,
			&ldaf,
			raw_data(ipiv),
			raw_data(s) if s != nil else nil,
			b.data,
			&ldb,
			x.data,
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bounds_norm),
			raw_data(err_bounds_comp),
			&nparams_int,
			raw_data(params) if params != nil else nil,
			raw_data(work),
			raw_data(iwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

// Extended symmetric iterative refinement for complex64/complex128
m_refine_symmetric_extended_c64_c128 :: proc(
	a: ^Matrix($T), // Original matrix A
	af: ^Matrix(T), // Factored matrix from factorization
	ipiv: []Blas_Int, // Pivot indices from factorization
	s: []$R, // Scale factors (if equed == APPLIED)
	b: ^Matrix(T), // Right-hand side matrix
	x: ^Matrix(T), // Solution matrix (refined on output)
	berr: []R, // Backward error bounds (size nrhs)
	err_bounds_norm: []R, // Normwise error bounds (nrhs * n_err_bnds)
	err_bounds_comp: []R, // Componentwise error bounds (nrhs * n_err_bnds)
	work: []T, // Complex workspace (size 2*n)
	rwork: []R, // Real workspace (size 3*n)
	equed := EquilibrationState.None,
	uplo := MatrixRegion.Upper,
	n_err_bnds: int = ERROR_BOUND_TYPES,
	nparams: int = 0,
	params: []R = nil,
) -> (
	rcond: R,
	info: Info,
	ok: bool,
) where is_complex(T) {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(err_bounds_norm) >= nrhs * n_err_bnds, "Normwise error bounds array too small")
	assert(len(err_bounds_comp) >= nrhs * n_err_bnds, "Componentwise error bounds array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= 3 * n, "Real workspace too small")

	uplo_c := cast(u8)uplo
	equed_c := cast(u8)(equed)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := a.ld
	ldaf := af.ld
	ldb := b.ld
	ldx := x.ld
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)

	when T == complex64 {
		lapack.csyrfsx_(
			&uplo_c,
			&equed_c,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			af.data,
			&ldaf,
			raw_data(ipiv),
			raw_data(s) if s != nil else nil,
			b.data,
			&ldb,
			x.data,
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bounds_norm),
			raw_data(err_bounds_comp),
			&nparams_int,
			raw_data(params) if params != nil else nil,
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	} else when T == complex128 {
		lapack.zsyrfsx_(
			&uplo_c,
			&equed_c,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			af.data,
			&ldaf,
			raw_data(ipiv),
			raw_data(s) if s != nil else nil,
			b.data,
			&ldb,
			x.data,
			&ldx,
			&rcond,
			raw_data(berr),
			&n_err_bnds_int,
			raw_data(err_bounds_norm),
			raw_data(err_bounds_comp),
			&nparams_int,
			raw_data(params) if params != nil else nil,
			raw_data(work),
			raw_data(rwork),
			&info,
		)
	}

	return rcond, info, info == 0
}

// Procedure group for extended symmetric iterative refinement
m_refine_symmetric_extended :: proc {
	m_refine_symmetric_extended_f32_f64,
	m_refine_symmetric_extended_c64_c128,
}
