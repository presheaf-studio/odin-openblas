package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// GENERALIZED SYMMETRIC BANDED EIGENVALUE PROBLEMS
// ============================================================================
// Solves the generalized eigenvalue problem A*x = λ*B*x where A and B are
// symmetric banded matrices and B is positive definite

m_reduce_generalized_banded :: proc {
	m_reduce_generalized_banded_f32_c64,
	m_reduce_generalized_banded_f64_c128,
}

// ============================================================================
// SYMMETRIC BANDED GENERALIZED EIGENVALUE REDUCTION
// ============================================================================
// Reduces a real symmetric-definite banded generalized eigenproblem
// A*x = λ*B*x to standard form C*y = λ*y

// Reduce generalized banded eigenvalue problem to standard form (f32/complex64)
// For real symmetric matrices (f32) and complex Hermitian matrices (complex64)
m_reduce_generalized_banded_f32_c64 :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: Blas_Int,
	ka: Blas_Int, // Number of superdiagonals of A
	kb: Blas_Int, // Number of superdiagonals of B
	ab: ^Matrix($T), // Band matrix A (modified on output)
	bb: ^Matrix(T), // Band matrix B (must contain Cholesky factor)
	x: ^Matrix(T) = nil, // Transformation matrix (if vect == FORM_VECTORS)
	work: []T = nil, // Workspace (size 2*n for real, n for complex)
	rwork: []f32 = nil, // Real workspace for complex64 only
	allocator := context.allocator,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	vect_cstring := vector_option_to_cstring(vect)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n := n
	ka := ka
	kb := kb
	ldab := ab.ld
	ldbb := bb.ld
	info_int: Info

	// Handle transformation matrix
	ldx: Blas_Int = 1
	x_ptr: ^T = nil
	if vect == .FORM_VECTORS && x != nil {
		assert(x.rows >= n && x.cols >= n, "Transformation matrix too small")
		ldx = x.ld
		x_ptr = raw_data(x.data)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	allocated_rwork := false

	when T == f32 {
		work_size := 2 * n
		if allocated_work {
			work = make([]T, work_size, allocator)
		}
	} else when T == complex64 {
		work_size := n
		if allocated_work {
			work = make([]T, work_size, allocator)
		}
		allocated_rwork = rwork == nil
		if allocated_rwork {
			rwork = make([]f32, n, allocator)
		}
	}

	defer {
		if allocated_work do delete(work)
		if allocated_rwork do delete(rwork)
	}

	// Call LAPACK
	when T == f32 {
		lapack.ssbgst_(
			vect_cstring,
			uplo_cstring,
			&n,
			&ka,
			&kb,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			&info_int,
			1,
			1,
		)
	} else when T == complex64 {
		lapack.chbgst_(
			vect_cstring,
			uplo_cstring,
			&n,
			&ka,
			&kb,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			raw_data(rwork),
			&info_int,
			1,
			1,
		)
	}

	return info_int, info_int == 0
}

// Reduce generalized banded eigenvalue problem to standard form (f64/complex128)
// For real symmetric matrices (f64) and complex Hermitian matrices (complex128)
m_reduce_generalized_banded_f64_c128 :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: Blas_Int,
	ka: Blas_Int, // Number of superdiagonals of A
	kb: Blas_Int, // Number of superdiagonals of B
	ab: ^Matrix($T), // Band matrix A (modified on output)
	bb: ^Matrix(T), // Band matrix B (must contain Cholesky factor)
	x: ^Matrix(T) = nil, // Transformation matrix (if vect == FORM_VECTORS)
	work: []T = nil, // Workspace (size 2*n for real, n for complex)
	rwork: []f64 = nil, // Real workspace for complex128 only
	allocator := context.allocator,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	vect_cstring := vector_option_to_cstring(vect)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n := n
	ka := ka
	kb := kb
	ldab := ab.ld
	ldbb := bb.ld
	info_int: Info

	// Handle transformation matrix
	ldx: Blas_Int = 1
	x_ptr: ^T = nil
	if vect == .FORM_VECTORS && x != nil {
		assert(x.rows >= n && x.cols >= n, "Transformation matrix too small")
		ldx = x.ld
		x_ptr = raw_data(x.data)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	allocated_rwork := false

	when T == f64 {
		work_size := 2 * n
		if allocated_work {
			work = make([]T, work_size, allocator)
		}
	} else when T == complex128 {
		work_size := n
		if allocated_work {
			work = make([]T, work_size, allocator)
		}
		allocated_rwork = rwork == nil
		if allocated_rwork {
			rwork = make([]f64, n, allocator)
		}
	}

	defer {
		if allocated_work do delete(work)
		if allocated_rwork do delete(rwork)
	}

	// Call LAPACK
	when T == f64 {
		lapack.dsbgst_(
			vect_cstring,
			uplo_cstring,
			&n,
			&ka,
			&kb,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			&info_int,
			1,
			1,
		)
	} else when T == complex128 {
		lapack.zhbgst_(
			vect_cstring,
			uplo_cstring,
			&n,
			&ka,
			&kb,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			x_ptr,
			&ldx,
			raw_data(work),
			raw_data(rwork),
			&info_int,
			1,
			1,
		)
	}

	return info_int, info_int == 0
}

// ============================================================================
// SYMMETRIC BANDED GENERALIZED EIGENVALUE COMPUTATION
// ============================================================================

// Generalized eigenvalue result
GeneralizedBandedEigenResult :: struct($T: typeid) {
	eigenvalues:            []T, // Computed eigenvalues (sorted)
	eigenvectors:           Matrix(T), // Eigenvector matrix (if requested)
	b_is_positive_definite: bool, // True if B was successfully factored
	all_positive:           bool, // True if all eigenvalues > 0
	min_eigenvalue:         f64, // Smallest eigenvalue
	max_eigenvalue:         f64, // Largest eigenvalue
	condition_number:       f64, // max|λ|/min|λ|
}

// Solve generalized banded eigenvalue problem
m_solve_generalized_banded :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	n: int,
	ka: int, // Number of superdiagonals of A
	kb: int, // Number of superdiagonals of B
	ab: ^Matrix($T), // Band matrix A (modified on output)
	bb: ^Matrix(T), // Band matrix B (modified to Cholesky factor)
	w: []T = nil, // Eigenvalues (size n)
	z: ^Matrix(T) = nil, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []T = nil, // Workspace (size 3*n)
	allocator := context.allocator,
) -> (
	result: GeneralizedBandedEigenResult(T),
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_cstring := eigen_job_to_cstring(jobz)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int: Blas_Int = n
	ka_int: Blas_Int = ka
	kb_int: Blas_Int = kb
	ldab := ab.ld
	ldbb := bb.ld
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]T, n, allocator)
	}
	result.eigenvalues = w

	// Handle eigenvectors
	ldz: Blas_Int = 1
	z_ptr: ^T = nil
	if jobz == .VALUES_VECTORS && z != nil {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = z.ld
		z_ptr = raw_data(z.data)
		result.eigenvectors = z^
	}

	// Allocate workspace if not provided
	work_size := 3 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]T, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	when T == f32 {
		lapack.ssbgv_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&info_int,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dsbgv_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&info_int,
			1,
			1,
		)
	}

	info = info_int
	ok = info == 0 || info > n

	// Check if B was positive definite
	result.b_is_positive_definite = ok

	// Analyze eigenvalues
	if ok && n > 0 {
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		when T == f32 {
			if abs(w[0]) > machine_parameter(f32, .Epsilon) {
				result.condition_number = f64(abs(w[n - 1] / w[0]))
			} else {
				result.condition_number = math.INF_F64
			}
		} else {
			if abs(result.min_eigenvalue) > machine_parameter(f64, .Epsilon) {
				result.condition_number = abs(result.max_eigenvalue / result.min_eigenvalue)
			} else {
				result.condition_number = math.INF_F64
			}
		}
	}

	return result, info, ok
}

// ============================================================================
// DIVIDE-AND-CONQUER GENERALIZED EIGENVALUE
// ============================================================================

// Solve generalized banded eigenvalue problem using divide-and-conquer (f32/complex64)
m_solve_generalized_banded_dc_f32_c64 :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: ^Matrix($T),
	bb: ^Matrix(T),
	w: []f32 = nil,
	z: ^Matrix(T) = nil,
	work: []T = nil,
	lwork: int = -1,
	rwork: []f32 = nil,
	lrwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedBandedEigenResult(f32),
	info: Info,
	work_size: int,
	rwork_size: int,
	iwork_size: int,
) where T == f32 || T == complex64 {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_cstring := eigen_job_to_cstring(jobz)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.ld)
	ldbb := Blas_Int(bb.ld)
	lwork_int := Blas_Int(lwork)
	lrwork_int := Blas_Int(lrwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 || (T == complex64 && lrwork == -1) {
		ldz := Blas_Int(1)

		when T == f32 {
			work_query: f32
			iwork_query: Blas_Int

			lapack.ssbgvd_(
				jobz_cstring,
				uplo_cstring,
				&n_int,
				&ka_int,
				&kb_int,
				raw_data(ab.data),
				&ldab,
				raw_data(bb.data),
				&ldbb,
				nil,
				nil,
				&ldz,
				&work_query,
				&lwork_int,
				&iwork_query,
				&liwork_int,
				&info_int,
				1,
				1,
			)

			work_size = int(work_query)
			iwork_size = int(iwork_query)
			rwork_size = 0
		} else when T == complex64 {
			work_query: complex64
			rwork_query: f32
			iwork_query: Blas_Int

			lapack.chbgvd_(
				jobz_cstring,
				uplo_cstring,
				&n_int,
				&ka_int,
				&kb_int,
				raw_data(ab.data),
				&ldab,
				raw_data(bb.data),
				&ldbb,
				nil,
				nil,
				&ldz,
				&work_query,
				&lwork_int,
				&rwork_query,
				&lrwork_int,
				&iwork_query,
				&liwork_int,
				&info_int,
				1,
				1,
			)

			work_size = int(real(work_query))
			rwork_size = int(rwork_query)
			iwork_size = int(iwork_query)
		}

		return result, Info(info_int), work_size, rwork_size, iwork_size
	}

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}
	result.eigenvalues = w

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^T = nil
	if jobz == .VALUES_VECTORS && z != nil {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.ld)
		z_ptr = raw_data(z.data)
		result.eigenvectors = z^
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	allocated_rwork := false
	allocated_iwork := iwork == nil

	when T == complex64 {
		allocated_rwork = rwork == nil
	}

	if allocated_work || allocated_rwork || allocated_iwork {
		// Query for optimal workspace
		lwork_query := Blas_Int(-1)
		lrwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		when T == f32 {
			work_query: f32
			iwork_query: Blas_Int

			lapack.ssbgvd_(
				jobz_cstring,
				uplo_cstring,
				&n_int,
				&ka_int,
				&kb_int,
				raw_data(ab.data),
				&ldab,
				raw_data(bb.data),
				&ldbb,
				raw_data(w),
				z_ptr,
				&ldz,
				&work_query,
				&lwork_query,
				&iwork_query,
				&liwork_query,
				&info_int,
				1,
				1,
			)

			if allocated_work {
				lwork = int(work_query)
				work = make([]T, lwork, allocator)
			}
			if allocated_iwork {
				liwork = int(iwork_query)
				iwork = make([]Blas_Int, liwork, allocator)
			}
		} else when T == complex64 {
			work_query: complex64
			rwork_query: f32
			iwork_query: Blas_Int

			lapack.chbgvd_(
				jobz_cstring,
				uplo_cstring,
				&n_int,
				&ka_int,
				&kb_int,
				raw_data(ab.data),
				&ldab,
				raw_data(bb.data),
				&ldbb,
				raw_data(w),
				z_ptr,
				&ldz,
				&work_query,
				&lwork_query,
				&rwork_query,
				&lrwork_query,
				&iwork_query,
				&liwork_query,
				&info_int,
				1,
				1,
			)

			if allocated_work {
				lwork = int(real(work_query))
				work = make([]T, lwork, allocator)
			}
			if allocated_rwork {
				lrwork = int(rwork_query)
				rwork = make([]f32, lrwork, allocator)
			}
			if allocated_iwork {
				liwork = int(iwork_query)
				iwork = make([]Blas_Int, liwork, allocator)
			}
		}
	}
	defer {
		if allocated_work do delete(work)
		if allocated_rwork do delete(rwork)
		if allocated_iwork do delete(iwork)
	}

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))
	when T == complex64 {
		lrwork_int = Blas_Int(len(rwork))
	}

	// Call LAPACK
	when T == f32 {
		lapack.ssbgvd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
			1,
			1,
		)
	} else when T == complex64 {
		lapack.chbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			raw_data(ab.data),
			&ldab,
			raw_data(bb.data),
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(work),
			&lwork_int,
			raw_data(rwork),
			&lrwork_int,
			raw_data(iwork),
			&liwork_int,
			&info_int,
			1,
			1,
		)
	}

	info = info_int

	// Check if B was positive definite
	result.b_is_positive_definite = info == 0 || info > Blas_Int(n)

	// Analyze eigenvalues
	if (info == 0 || info > Blas_Int(n)) && n > 0 {
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_parameter(f32, .Epsilon) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	when T == complex64 {
		rwork_size = len(rwork)
	} else {
		rwork_size = 0
	}
	iwork_size = len(iwork)
	return
}

// Solve generalized banded eigenvalue problem using divide-and-conquer (f64/complex128)
m_solve_generalized_banded_dc_f64_c128 :: proc(
	jobz: EigenJobOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: ^Matrix($T),
	bb: ^Matrix(T),
	w: []f64 = nil,
	z: ^Matrix(T) = nil,
	work: []T = nil,
	lwork: int = -1,
	rwork: []f64 = nil,
	lrwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedBandedEigenResult(f64),
	info: Info,
	work_size: int,
	rwork_size: int,
	iwork_size: int,
) where T == f64 || T == complex128 {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_cstring := eigen_job_to_cstring(jobz)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.ld)
	ldbb := Blas_Int(bb.ld)
	lwork_int := Blas_Int(lwork)
	lrwork_int := Blas_Int(lrwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 || (T == complex128 && lrwork == -1) {
		ldz := Blas_Int(1)

		when T == f64 {
			work_query: f64
			iwork_query: Blas_Int

			lapack.dsbgvd_(
				jobz_cstring,
				uplo_cstring,
				&n_int,
				&ka_int,
				&kb_int,
				raw_data(ab.data),
				&ldab,
				raw_data(bb.data),
				&ldbb,
				nil,
				nil,
				&ldz,
				&work_query,
				&lwork_int,
				&iwork_query,
				&liwork_int,
				&info_int,
				1,
				1,
			)

			work_size = int(work_query)
			iwork_size = int(iwork_query)
			rwork_size = 0
		} else when T == complex128 {
			work_query: complex128
			rwork_query: f64
			iwork_query: Blas_Int

			lapack.zhbgvd_(
				jobz_cstring,
				uplo_cstring,
				&n_int,
				&ka_int,
				&kb_int,
				raw_data(ab.data),
				&ldab,
				raw_data(bb.data),
				&ldbb,
				nil,
				nil,
				&ldz,
				&work_query,
				&lwork_int,
				&rwork_query,
				&lrwork_int,
				&iwork_query,
				&liwork_int,
				&info_int,
				1,
				1,
			)

			work_size = int(real(work_query))
			rwork_size = int(rwork_query)
			iwork_size = int(iwork_query)
		}

		return result, Info(info_int), work_size, rwork_size, iwork_size
	}

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}
	result.eigenvalues = w

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	allocated_iwork := iwork == nil

	if allocated_work || allocated_iwork {
		// Query for optimal workspace
		work_query: f32
		iwork_query: Blas_Int
		lwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		lapack.ssbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			ab.data,
			&ldab,
			bb.data,
			&ldbb,
			raw_data(w),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_query,
			&iwork_query,
			&liwork_query,
			&info_int,
			1,
			1,
		)

		if allocated_work {
			lwork = int(work_query)
			work = make([]f32, lwork, allocator)
		}
		if allocated_iwork {
			liwork = int(iwork_query)
			iwork = make([]Blas_Int, liwork, allocator)
		}
	}
	defer {
		if allocated_work do delete(work)
		if allocated_iwork do delete(iwork)
	}

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.ssbgvd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Check if B was positive definite
	result.b_is_positive_definite = info == .OK || info > Info(n)

	// Analyze eigenvalues
	if (info == .OK || info > Info(n)) && n > 0 {
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_parameter(f32, .Epsilon) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

sbgvd :: proc {
	dsbgvd,
	ssbgvd,
}

// ============================================================================
// SELECTIVE GENERALIZED EIGENVALUE
// ============================================================================

// Selective generalized eigenvalue result
SelectiveGeneralizedResult :: struct($T: typeid) {
	eigenvalues:            []T,
	eigenvectors:           Matrix(T),
	num_found:              int,
	failed_indices:         []Blas_Int,
	b_is_positive_definite: bool,
}

// Double precision selective generalized eigenvalue
dsbgvx :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: Matrix(f64),
	bb: Matrix(f64),
	q: Matrix(f64) = {},
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f64 = 0,
	w: []f64 = nil,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: SelectiveGeneralizedResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_cstring := eigen_job_to_cstring(jobz)
	range_cstring := eigen_range_to_cstring(range)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.ld)
	ldbb := Blas_Int(bb.ld)
	ldq := Blas_Int(max(1, n))
	q_ptr: ^T = nil
	if q != nil && q.data != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.ld)
		q_ptr = raw_data(q.data)
	}

	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol

	m: Blas_Int
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}

	// Allocate eigenvector storage
	ldz := Blas_Int(max(1, n))
	z_ptr: ^f64 = nil
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 7 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Allocate failure array
	ifail := make([]Blas_Int, n, allocator)

	// Call LAPACK
	lapack.dsbgvx_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
		q_ptr,
		&ldq,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		raw_data(iwork),
		raw_data(ifail),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.num_found = int(m)
	if result.num_found > 0 {
		result.eigenvalues = w[:result.num_found]
	}
	if jobz == .VALUES_VECTORS {
		result.eigenvectors = z
	}
	result.failed_indices = ifail
	result.b_is_positive_definite = info == .OK || info > Info(n)

	return
}

// Single precision selective generalized eigenvalue
ssbgvx :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: MatrixRegion,
	n: int,
	ka: int,
	kb: int,
	ab: Matrix(f32),
	bb: Matrix(f32),
	q: Matrix(f32) = {},
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f32 = 0,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: SelectiveGeneralizedResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_cstring := eigen_job_to_cstring(jobz)
	range_cstring := eigen_range_to_cstring(range)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
	ldq := Blas_Int(max(1, n))
	q_ptr: ^f32 = nil
	if q.data != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
	}

	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol

	m: Blas_Int
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}

	// Allocate eigenvector storage
	ldz := Blas_Int(max(1, n))
	z_ptr: ^f32 = nil
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 7 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Allocate failure array
	ifail := make([]Blas_Int, n, allocator)

	// Call LAPACK
	lapack.ssbgvx_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
		q_ptr,
		&ldq,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		raw_data(iwork),
		raw_data(ifail),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.num_found = int(m)
	if result.num_found > 0 {
		result.eigenvalues = w[:result.num_found]
	}
	if jobz == .VALUES_VECTORS {
		result.eigenvectors = z
	}
	result.failed_indices = ifail
	result.b_is_positive_definite = info == .OK || info > Info(n)

	return
}

sbgvx :: proc {
	dsbgvx,
	ssbgvx,
}

// ============================================================================
// SYMMETRIC BAND TO TRIDIAGONAL REDUCTION
// ============================================================================

// Tridiagonal reduction result
TridiagonalReductionResult :: struct($T: typeid) {
	diagonal:     []T, // Diagonal elements of tridiagonal matrix
	off_diagonal: []T, // Off-diagonal elements
	q_matrix:     Matrix(T), // Orthogonal transformation matrix (if requested)
}

// Double precision band to tridiagonal reduction
dsbtrd :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int, // Number of superdiagonals/subdiagonals
	ab: Matrix(f64), // Band matrix (modified on output)
	d: []f64 = nil, // Diagonal of tridiagonal (size n)
	e: []f64 = nil, // Off-diagonal of tridiagonal (size n-1)
	q: Matrix(f64) = {}, // Orthogonal matrix (if vect == FORM_VECTORS)
	work: []f64 = nil, // Workspace (size n)
	allocator := context.allocator,
) -> (
	result: TridiagonalReductionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	vect_cstring := vector_option_to_cstring(vect)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	info_int: Info

	// Allocate diagonal if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f64, n, allocator)
	}
	result.diagonal = d

	// Allocate off-diagonal if not provided
	allocated_e := e == nil
	if allocated_e && n > 0 {
		e = make([]f64, n - 1, allocator)
	}
	result.off_diagonal = e

	// Handle Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^f64 = nil
	if vect == .FORM_VECTORS {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
		result.q_matrix = q
	}

	// Allocate workspace if not provided
	work_size := n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsbtrd_(
		vect_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		raw_data(d),
		raw_data(e),
		q_ptr,
		&ldq,
		raw_data(work),
		&info_int,
		1,
		1,
	)

	return result, Info(info_int)
}

// Single precision band to tridiagonal reduction
ssbtrd :: proc(
	vect: VectorOption,
	uplo: MatrixRegion,
	n: int,
	kd: int,
	ab: Matrix(f32),
	d: []f32 = nil,
	e: []f32 = nil,
	q: Matrix(f32) = {},
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: TridiagonalReductionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	vect_cstring := vector_option_to_cstring(vect)
	uplo_cstring := matrix_region_to_cstring(uplo)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	info_int: Info

	// Allocate diagonal if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f32, n, allocator)
	}
	result.diagonal = d

	// Allocate off-diagonal if not provided
	allocated_e := e == nil
	if allocated_e && n > 0 {
		e = make([]f32, n - 1, allocator)
	}
	result.off_diagonal = e

	// Handle Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^f32 = nil
	if vect == .FORM_VECTORS {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
		result.q_matrix = q
	}

	// Allocate workspace if not provided
	work_size := n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssbtrd_(
		vect_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		raw_data(d),
		raw_data(e),
		q_ptr,
		&ldq,
		raw_data(work),
		&info_int,
		1,
		1,
	)

	return result, Info(info_int)
}

sbtrd :: proc {
	dsbtrd,
	ssbtrd,
}

// ============================================================================
// SYMMETRIC RANK-K UPDATE IN RFP FORMAT
// ============================================================================

// RFP (Rectangular Full Packed) format transpose options
RFPTranspose :: enum {
	NORMAL, // 'N' - Normal form
	TRANSPOSE, // 'T' - Transpose form
	CONJUGATE, // 'C' - Conjugate transpose (complex only)
}

rfp_transpose_to_cstring :: proc(trans: RFPTranspose) -> cstring {
	switch trans {
	case .NORMAL:
		return "N"
	case .TRANSPOSE:
		return "T"
	case .CONJUGATE:
		return "C"
	}
	unreachable()
}

// Double precision symmetric rank-k update in RFP format
dsfrk :: proc(
	transr: RFPTranspose,
	uplo: MatrixRegion,
	trans: TransposeMode,
	n: int,
	k: int,
	alpha: f64,
	a: Matrix(f64),
	beta: f64,
	c: []f64, // RFP format array
) {
	assert(n >= 0 && k >= 0, "Dimensions must be non-negative")

	transr_cstring := rfp_transpose_to_cstring(transr)

	uplo_cstring := matrix_region_to_cstring(uplo)

	trans_cstring := transpose_to_cstring(trans)

	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	alpha_val := alpha
	beta_val := beta
	lda := Blas_Int(a.stride)

	// Call LAPACK
	lapack.dsfrk_(
		transr_cstring,
		uplo_cstring,
		trans_cstring,
		&n_int,
		&k_int,
		&alpha_val,
		a.data,
		&lda,
		&beta_val,
		raw_data(c),
		1,
		1,
		1,
	)
}

// Single precision symmetric rank-k update in RFP format
ssfrk :: proc(
	transr: RFPTranspose,
	uplo: MatrixRegion,
	trans: TransposeMode,
	n: int,
	k: int,
	alpha: f32,
	a: Matrix(f32),
	beta: f32,
	c: []f32,
) {
	assert(n >= 0 && k >= 0, "Dimensions must be non-negative")

	transr_cstring := rfp_transpose_to_cstring(transr)

	uplo_cstring := matrix_region_to_cstring(uplo)

	trans_cstring := transpose_to_cstring(trans)

	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	alpha_val := alpha
	beta_val := beta
	lda := Blas_Int(a.stride)

	// Call LAPACK
	lapack.ssfrk_(
		transr_cstring,
		uplo_cstring,
		trans_cstring,
		&n_int,
		&k_int,
		&alpha_val,
		a.data,
		&lda,
		&beta_val,
		raw_data(c),
		1,
		1,
		1,
	)
}

sfrk :: proc {
	dsfrk,
	ssfrk,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Solve generalized eigenvalue problem for banded matrices
solve_generalized_banded :: proc(
	a: Matrix($T),
	b: Matrix(T),
	ka: int,
	kb: int,
	compute_vectors := false,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	info: Info,
) {
	n := a.cols
	jobz := compute_vectors ? EigenJobOption.VALUES_VECTORS : EigenJobOption.VALUES_ONLY

	// Make copies since they get modified
	a_copy := matrix_clone(&a, allocator)
	defer matrix_delete(&a_copy)
	b_copy := matrix_clone(&b, allocator)
	defer matrix_delete(&b_copy)

	if compute_vectors {
		eigenvectors = create_matrix(T, n, n, allocator)
	}

	when T == f64 {
		result, info_val := dsbgv(
			jobz,
			.Lower,
			n,
			ka,
			kb,
			a_copy,
			b_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else when T == f32 {
		result, info_val := ssbgv(
			jobz,
			.Lower,
			n,
			ka,
			kb,
			a_copy,
			b_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else {
		#panic("Unsupported type for generalized banded eigenvalue")
	}
}
