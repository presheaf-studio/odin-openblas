package openblas

import lapack "./f77"
import "base:builtin"
import "core:c"
import "core:math"
import "core:mem"

// ===================================================================================
// HOUSEHOLDER REFLECTOR OPERATIONS
// ===================================================================================

// Apply single Householder reflector proc groups
solve_apply_householder_reflector :: proc {
	solve_apply_householder_reflector_f32,
	solve_apply_householder_reflector_f64,
	solve_apply_householder_reflector_c64,
	solve_apply_householder_reflector_c128,
}

// Apply block Householder reflector proc groups
solve_apply_block_householder_reflector :: proc {
	solve_apply_block_householder_reflector_f32,
	solve_apply_block_householder_reflector_f64,
	solve_apply_block_householder_reflector_c64,
	solve_apply_block_householder_reflector_c128,
}

// Generate Householder reflector proc groups
v_generate_householder_reflector :: proc {
	v_generate_householder_reflector_f32,
	v_generate_householder_reflector_f64,
	v_generate_householder_reflector_c64,
	v_generate_householder_reflector_c128,
}

// Form triangular factor of block Householder reflector proc groups
form_triangular_block_householder :: proc {
	form_triangular_block_householder_f32,
	form_triangular_block_householder_f64,
	form_triangular_block_householder_c64,
	form_triangular_block_householder_c128,
}

// Apply elementary Householder reflector proc groups
apply_elementary_householder :: proc {
	apply_elementary_householder_f32,
	apply_elementary_householder_f64,
	apply_elementary_householder_c64,
	apply_elementary_householder_c128,
}

// ===================================================================================
// SINGLE HOUSEHOLDER REFLECTOR APPLICATION
// ===================================================================================

// Query workspace for applying Householder reflector
query_workspace_householder_reflector :: proc($T: typeid, m: int, n: int, side: ReflectorSide) -> (work: Blas_Int) where is_float(T) || is_complex(T) {
	return side == .Left ? Blas_Int(n) : Blas_Int(m)
}

// Apply single Householder reflector (f32)
solve_apply_householder_reflector_f32 :: proc(side: ReflectorSide, m: int, n: int, V: ^Vector(f32), tau: f32, C: ^Matrix(f32), work: []f32) {
	// Validate workspace
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	incv := Blas_Int(V.incr)
	ldc := Blas_Int(C.ld)
	side_c := cast(u8)side
	tau_val := tau

	lapack.slarf_(&side_c, &m_int, &n_int, &V.data[V.offset], &incv, &tau_val, raw_data(C.data), &ldc, raw_data(work))
}

// Apply single Householder reflector (f64)
solve_apply_householder_reflector_f64 :: proc(side: ReflectorSide, m: int, n: int, V: ^Vector(f64), tau: f64, C: ^Matrix(f64), work: []f64) {
	// Validate workspace
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	incv := Blas_Int(V.incr)
	ldc := Blas_Int(C.ld)
	side_c := cast(u8)side
	tau_val := tau

	lapack.dlarf_(&side_c, &m_int, &n_int, &V.data[V.offset], &incv, &tau_val, raw_data(C.data), &ldc, raw_data(work))
}

// Apply single Householder reflector (c64)
solve_apply_householder_reflector_c64 :: proc(side: ReflectorSide, m: int, n: int, V: ^Vector(complex64), tau: complex64, C: ^Matrix(complex64), work: []complex64) {
	// Validate workspace
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	incv := Blas_Int(V.incr)
	ldc := Blas_Int(C.ld)
	side_c := cast(u8)side
	tau_val := tau

	lapack.clarf_(&side_c, &m_int, &n_int, cast(^complex64)&V.data[V.offset], &incv, cast(^complex64)&tau_val, cast(^complex64)raw_data(C.data), &ldc, cast(^complex64)raw_data(work))
}

// Apply single Householder reflector (c128)
solve_apply_householder_reflector_c128 :: proc(side: ReflectorSide, m: int, n: int, V: ^Vector(complex128), tau: complex128, C: ^Matrix(complex128), work: []complex128) {
	// Validate workspace
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	incv := Blas_Int(V.incr)
	ldc := Blas_Int(C.ld)
	side_c := cast(u8)side
	tau_val := tau

	lapack.zlarf_(&side_c, &m_int, &n_int, cast(^complex128)&V.data[V.offset], &incv, cast(^complex128)&tau_val, cast(^complex128)raw_data(C.data), &ldc, cast(^complex128)raw_data(work))
}

// ===================================================================================
// BLOCK HOUSEHOLDER REFLECTOR APPLICATION
// ===================================================================================

// Query workspace for applying block Householder reflector
query_workspace_block_householder_reflector :: proc($T: typeid, m: int, n: int, k: int, side: ReflectorSide) -> (work: Blas_Int) where is_float(T) || is_complex(T) {
	ldwork := side == .Left ? n : m
	return Blas_Int(ldwork * k)
}

// Apply block Householder reflector (f32)
solve_apply_block_householder_reflector_f32 :: proc(
	side: ReflectorSide,
	trans: ReflectorTranspose,
	direction: ReflectorDirection,
	storage: ReflectorStorage,
	m: int,
	n: int,
	k: int,
	V: ^Matrix(f32), // Matrix of Householder vectors
	T_matrix: ^Matrix(f32), // Block reflector T matrix
	C: ^Matrix(f32), // Matrix to transform
	work: []f32, // Workspace array
) {
	// Validate workspace
	ldwork := side == .Left ? n : m
	assert(len(work) >= ldwork * k, "Work array too small")

	side_c := cast(u8)side
	trans_c := cast(u8)trans
	direct_c := cast(u8)direction
	storev_c := cast(u8)storage

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)
	ldc := Blas_Int(C.ld)
	ldwork_int := Blas_Int(ldwork)

	lapack.slarfb_(&side_c, &trans_c, &direct_c, &storev_c, &m_int, &n_int, &k_int, raw_data(V.data), &ldv, raw_data(T_matrix.data), &ldt, raw_data(C.data), &ldc, raw_data(work), &ldwork_int)
}

// Apply block Householder reflector (f64)
solve_apply_block_householder_reflector_f64 :: proc(
	side: ReflectorSide,
	trans: ReflectorTranspose,
	direction: ReflectorDirection,
	storage: ReflectorStorage,
	m: int,
	n: int,
	k: int,
	V: ^Matrix(f64), // Matrix of Householder vectors
	T_matrix: ^Matrix(f64), // Block reflector T matrix
	C: ^Matrix(f64), // Matrix to transform
	work: []f64, // Workspace array
) {
	// Validate workspace
	ldwork := side == .Left ? n : m
	assert(len(work) >= ldwork * k, "Work array too small")

	side_c := cast(u8)side
	trans_c := cast(u8)trans
	direct_c := cast(u8)direction
	storev_c := cast(u8)storage

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)
	ldc := Blas_Int(C.ld)
	ldwork_int := Blas_Int(ldwork)

	lapack.dlarfb_(&side_c, &trans_c, &direct_c, &storev_c, &m_int, &n_int, &k_int, raw_data(V.data), &ldv, raw_data(T_matrix.data), &ldt, raw_data(C.data), &ldc, raw_data(work), &ldwork_int)
}

// Apply block Householder reflector (c64)
solve_apply_block_householder_reflector_c64 :: proc(
	side: ReflectorSide,
	trans: ReflectorTranspose,
	direction: ReflectorDirection,
	storage: ReflectorStorage,
	m: int,
	n: int,
	k: int,
	V: ^Matrix(complex64), // Matrix of Householder vectors
	T_matrix: ^Matrix(complex64), // Block reflector T matrix
	C: ^Matrix(complex64), // Matrix to transform
	work: []complex64, // Workspace array
) {
	// Validate workspace
	ldwork := side == .Left ? n : m
	assert(len(work) >= ldwork * k, "Work array too small")

	side_c := cast(u8)side
	trans_c := cast(u8)trans
	direct_c := cast(u8)direction
	storev_c := cast(u8)storage

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)
	ldc := Blas_Int(C.ld)
	ldwork_int := Blas_Int(ldwork)

	lapack.clarfb_(&side_c, &trans_c, &direct_c, &storev_c, &m_int, &n_int, &k_int, cast(^complex64)raw_data(V.data), &ldv, cast(^complex64)raw_data(T_matrix.data), &ldt, cast(^complex64)raw_data(C.data), &ldc, cast(^complex64)raw_data(work), &ldwork_int)
}

// Apply block Householder reflector (c128)
solve_apply_block_householder_reflector_c128 :: proc(
	side: ReflectorSide,
	trans: ReflectorTranspose,
	direction: ReflectorDirection,
	storage: ReflectorStorage,
	m: int,
	n: int,
	k: int,
	V: ^Matrix(complex128), // Matrix of Householder vectors
	T_matrix: ^Matrix(complex128), // Block reflector T matrix
	C: ^Matrix(complex128), // Matrix to transform
	work: []complex128, // Workspace array
) {
	// Validate workspace
	ldwork := side == .Left ? n : m
	assert(len(work) >= ldwork * k, "Work array too small")

	side_c := cast(u8)side
	trans_c := cast(u8)trans
	direct_c := cast(u8)direction
	storev_c := cast(u8)storage

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)
	ldc := Blas_Int(C.ld)
	ldwork_int := Blas_Int(ldwork)

	lapack.zlarfb_(&side_c, &trans_c, &direct_c, &storev_c, &m_int, &n_int, &k_int, cast(^complex128)raw_data(V.data), &ldv, cast(^complex128)raw_data(T_matrix.data), &ldt, cast(^complex128)raw_data(C.data), &ldc, cast(^complex128)raw_data(work), &ldwork_int)
}

// ===================================================================================
// HOUSEHOLDER REFLECTOR GENERATION
// ===================================================================================

// Generate Householder reflector (f32)
v_generate_householder_reflector_f32 :: proc(
	X: ^Vector(f32), // Vector to reflect (modified in-place)
) -> (
	tau: f32,
) {
	// Generate Householder reflector to zero out all but first element
	n := Blas_Int(X.size)
	incx := Blas_Int(X.incr)

	lapack.slarfg_(&n, &X.data[X.offset], &X.data[X.offset + incx], &incx, &tau)

	return tau
}

// Generate Householder reflector (f64)
v_generate_householder_reflector_f64 :: proc(
	X: ^Vector(f64), // Vector to reflect (modified in-place)
) -> (
	tau: f64,
) {
	// Generate Householder reflector to zero out all but first element
	n := Blas_Int(X.size)
	incx := Blas_Int(X.incr)

	lapack.dlarfg_(&n, &X.data[X.offset], &X.data[X.offset + incx], &incx, &tau)

	return tau
}

// Generate Householder reflector (c64)
v_generate_householder_reflector_c64 :: proc(
	X: ^Vector(complex64), // Vector to reflect (modified in-place)
) -> (
	tau: complex64,
) {
	// Generate Householder reflector to zero out all but first element
	n := Blas_Int(X.size)
	incx := Blas_Int(X.incr)

	lapack.clarfg_(&n, cast(^complex64)&X.data[X.offset], cast(^complex64)&X.data[X.offset + incx], &incx, cast(^complex64)&tau)

	return tau
}

// Generate Householder reflector (c128)
v_generate_householder_reflector_c128 :: proc(
	X: ^Vector(complex128), // Vector to reflect (modified in-place)
) -> (
	tau: complex128,
) {
	// Generate Householder reflector to zero out all but first element
	n := Blas_Int(X.size)
	incx := Blas_Int(X.incr)

	lapack.zlarfg_(&n, cast(^complex128)&X.data[X.offset], cast(^complex128)&X.data[X.offset + incx], &incx, cast(^complex128)&tau)

	return tau
}

// ===================================================================================
// TRIANGULAR BLOCK HOUSEHOLDER FACTOR FORMATION (LARFT)
// ===================================================================================

// Form triangular factor of block Householder reflector (f32)
// Computes the triangular T matrix used in block Householder transformations
form_triangular_block_householder_f32 :: proc(
	direction: ReflectorDirection, // Forward or Backward
	storage: ReflectorStorage, // Column-wise or Row-wise storage of V
	n: int, // Order of the matrix V
	k: int, // Number of reflectors
	V: ^Matrix(f32), // Householder vectors (n x k)
	tau: []f32, // Scalar factors of reflectors (size k)
	T_matrix: ^Matrix(f32), // Triangular factor T (k x k, output)
) {
	assert(len(tau) >= k, "tau array too small")
	assert(T_matrix.rows >= k && T_matrix.cols >= k, "T matrix too small")

	direct_c := cast(u8)direction
	storev_c := cast(u8)storage
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)

	lapack.slarft_(&direct_c, &storev_c, &n_int, &k_int, raw_data(V.data), &ldv, raw_data(tau), raw_data(T_matrix.data), &ldt, 1, 1)
}

// Form triangular factor of block Householder reflector (f64)
// Computes the triangular T matrix used in block Householder transformations
form_triangular_block_householder_f64 :: proc(
	direction: ReflectorDirection, // Forward or Backward
	storage: ReflectorStorage, // Column-wise or Row-wise storage of V
	n: int, // Order of the matrix V
	k: int, // Number of reflectors
	V: ^Matrix(f64), // Householder vectors (n x k)
	tau: []f64, // Scalar factors of reflectors (size k)
	T_matrix: ^Matrix(f64), // Triangular factor T (k x k, output)
) {
	assert(len(tau) >= k, "tau array too small")
	assert(T_matrix.rows >= k && T_matrix.cols >= k, "T matrix too small")

	direct_c := cast(u8)direction
	storev_c := cast(u8)storage
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)

	lapack.dlarft_(&direct_c, &storev_c, &n_int, &k_int, raw_data(V.data), &ldv, raw_data(tau), raw_data(T_matrix.data), &ldt, 1, 1)
}

// Form triangular factor of block Householder reflector (c64)
// Computes the triangular T matrix used in block Householder transformations
form_triangular_block_householder_c64 :: proc(
	direction: ReflectorDirection, // Forward or Backward
	storage: ReflectorStorage, // Column-wise or Row-wise storage of V
	n: int, // Order of the matrix V
	k: int, // Number of reflectors
	V: ^Matrix(complex64), // Householder vectors (n x k)
	tau: []complex64, // Scalar factors of reflectors (size k)
	T_matrix: ^Matrix(complex64), // Triangular factor T (k x k, output)
) {
	assert(len(tau) >= k, "tau array too small")
	assert(T_matrix.rows >= k && T_matrix.cols >= k, "T matrix too small")

	direct_c := cast(u8)direction
	storev_c := cast(u8)storage
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)

	lapack.clarft_(&direct_c, &storev_c, &n_int, &k_int, cast(^complex64)raw_data(V.data), &ldv, cast(^complex64)raw_data(tau), cast(^complex64)raw_data(T_matrix.data), &ldt, 1, 1)
}

// Form triangular factor of block Householder reflector (c128)
// Computes the triangular T matrix used in block Householder transformations
form_triangular_block_householder_c128 :: proc(
	direction: ReflectorDirection, // Forward or Backward
	storage: ReflectorStorage, // Column-wise or Row-wise storage of V
	n: int, // Order of the matrix V
	k: int, // Number of reflectors
	V: ^Matrix(complex128), // Householder vectors (n x k)
	tau: []complex128, // Scalar factors of reflectors (size k)
	T_matrix: ^Matrix(complex128), // Triangular factor T (k x k, output)
) {
	assert(len(tau) >= k, "tau array too small")
	assert(T_matrix.rows >= k && T_matrix.cols >= k, "T matrix too small")

	direct_c := cast(u8)direction
	storev_c := cast(u8)storage
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := Blas_Int(V.ld)
	ldt := Blas_Int(T_matrix.ld)

	lapack.zlarft_(&direct_c, &storev_c, &n_int, &k_int, cast(^complex128)raw_data(V.data), &ldv, cast(^complex128)raw_data(tau), cast(^complex128)raw_data(T_matrix.data), &ldt, 1, 1)
}

// ===================================================================================
// ELEMENTARY HOUSEHOLDER REFLECTOR APPLICATION (LARFX)
// ===================================================================================

// Query workspace for elementary Householder reflector
query_workspace_elementary_householder :: proc($T: typeid, m: int, n: int, side: ReflectorSide) -> (work_size: int) where is_float(T) || is_complex(T) {
	// larfx requires workspace of size n (if side=Left) or m (if side=Right)
	return side == .Left ? n : m
}

// Apply elementary Householder reflector (f32)
// Specialized version of larf for small-order reflectors (faster for n <= 10)
apply_elementary_householder_f32 :: proc(
	side: ReflectorSide, // Left or Right
	m: int, // Number of rows of C
	n: int, // Number of columns of C
	V: []f32, // Householder vector (size m if side=Left, n if side=Right)
	tau: f32, // Scalar factor tau
	C: ^Matrix(f32), // Matrix to transform (m x n)
	work: []f32, // Workspace (size n if side=Left, m if side=Right)
) {
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	v_size := side == .Left ? m : n
	assert(len(V) >= v_size, "V vector too small")

	side_c := cast(u8)side
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	ldc := Blas_Int(C.ld)
	tau_val := tau

	lapack.slarfx_(&side_c, &m_int, &n_int, raw_data(V), &tau_val, raw_data(C.data), &ldc, raw_data(work), 1)
}

// Apply elementary Householder reflector (f64)
// Specialized version of larf for small-order reflectors (faster for n <= 10)
apply_elementary_householder_f64 :: proc(
	side: ReflectorSide, // Left or Right
	m: int, // Number of rows of C
	n: int, // Number of columns of C
	V: []f64, // Householder vector (size m if side=Left, n if side=Right)
	tau: f64, // Scalar factor tau
	C: ^Matrix(f64), // Matrix to transform (m x n)
	work: []f64, // Workspace (size n if side=Left, m if side=Right)
) {
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	v_size := side == .Left ? m : n
	assert(len(V) >= v_size, "V vector too small")

	side_c := cast(u8)side
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	ldc := Blas_Int(C.ld)
	tau_val := tau

	lapack.dlarfx_(&side_c, &m_int, &n_int, raw_data(V), &tau_val, raw_data(C.data), &ldc, raw_data(work), 1)
}

// Apply elementary Householder reflector (c64)
// Specialized version of larf for small-order reflectors (faster for n <= 10)
apply_elementary_householder_c64 :: proc(
	side: ReflectorSide, // Left or Right
	m: int, // Number of rows of C
	n: int, // Number of columns of C
	V: []complex64, // Householder vector (size m if side=Left, n if side=Right)
	tau: complex64, // Scalar factor tau
	C: ^Matrix(complex64), // Matrix to transform (m x n)
	work: []complex64, // Workspace (size n if side=Left, m if side=Right)
) {
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	v_size := side == .Left ? m : n
	assert(len(V) >= v_size, "V vector too small")

	side_c := cast(u8)side
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	ldc := Blas_Int(C.ld)
	tau_val := tau

	lapack.clarfx_(&side_c, &m_int, &n_int, cast(^complex64)raw_data(V), cast(^complex64)&tau_val, cast(^complex64)raw_data(C.data), &ldc, cast(^complex64)raw_data(work), 1)
}

// Apply elementary Householder reflector (c128)
// Specialized version of larf for small-order reflectors (faster for n <= 10)
apply_elementary_householder_c128 :: proc(
	side: ReflectorSide, // Left or Right
	m: int, // Number of rows of C
	n: int, // Number of columns of C
	V: []complex128, // Householder vector (size m if side=Left, n if side=Right)
	tau: complex128, // Scalar factor tau
	C: ^Matrix(complex128), // Matrix to transform (m x n)
	work: []complex128, // Workspace (size n if side=Left, m if side=Right)
) {
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	v_size := side == .Left ? m : n
	assert(len(V) >= v_size, "V vector too small")

	side_c := cast(u8)side
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	ldc := Blas_Int(C.ld)
	tau_val := tau

	lapack.zlarfx_(&side_c, &m_int, &n_int, cast(^complex128)raw_data(V), cast(^complex128)&tau_val, cast(^complex128)raw_data(C.data), &ldc, cast(^complex128)raw_data(work), 1)
}

// ===================================================================================
// UTILITY FUNCTIONS
// ===================================================================================

// Convert ReflectorSide to cstring
side_to_cstring :: proc(side: ReflectorSide) -> cstring {
	switch side {
	case .Left:
		return "L"
	case .Right:
		return "R"
	}
	unreachable()
}

// Convert ReflectorTranspose to cstring
transpose_to_cstring :: proc(trans: ReflectorTranspose) -> cstring {
	switch trans {
	case .None:
		return "N"
	case .Transpose:
		return "T"
	case .Conjugate:
		return "C"
	}
	unreachable()
}

// Note: direction_to_cstring is defined in givens_rotations.odin

// Convert ReflectorStorage to cstring
storage_to_cstring :: proc(storage: ReflectorStorage) -> cstring {
	switch storage {
	case .ColumnWise:
		return "C"
	case .RowWise:
		return "R"
	}
	unreachable()
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Apply Householder reflector to matrix with automatic workspace allocation
apply_householder_reflector_auto :: proc(side: ReflectorSide, V: ^Vector($T), tau: T, C: ^Matrix(T), allocator := context.allocator) -> bool where is_float(T) || is_complex(T) {
	m := C.rows
	n := C.cols
	work_size := side == .Left ? n : m

	work := make([]T, work_size, allocator)
	defer delete(work, allocator)

	when T == f32 {
		solve_apply_householder_reflector_f32(side, m, n, V, tau, C, work)
	} else when T == f64 {
		solve_apply_householder_reflector_f64(side, m, n, V, tau, C, work)
	} else when T == complex64 {
		solve_apply_householder_reflector_c64(side, m, n, V, tau, C, work)
	} else when T == complex128 {
		solve_apply_householder_reflector_c128(side, m, n, V, tau, C, work)
	}

	return true
}
