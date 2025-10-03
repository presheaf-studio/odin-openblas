package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// GIVENS ROTATIONS AND MATRIX SCALING
// ===================================================================================

// Generate Givens rotation proc groups
m_generate_robust_givens :: proc {
	m_generate_robust_givens_f32,
	m_generate_robust_givens_f64,
}

m_generate_givens :: proc {
	m_generate_givens_f32,
	m_generate_givens_f64,
}

// Matrix scaling proc groups
m_scale :: proc {
	m_scale_f32,
	m_scale_f64,
	m_scale_c64,
	m_scale_c128,
}

// Apply Givens sequence proc groups
m_apply_givens_sequence :: proc {
	m_apply_givens_sequence_f32,
	m_apply_givens_sequence_f64,
	m_apply_givens_sequence_c64,
	m_apply_givens_sequence_c128,
}

// ===================================================================================
// GIVENS ROTATION GENERATION
// ===================================================================================

// Generate robust Givens rotation (f32)
// Generates plane rotation with robustness against overflow
m_generate_robust_givens_f32 :: proc(f, g: f32) -> (cs, sn, r: f32) {
	cs_val, sn_val, r_val: f32
	f_val, g_val := f, g
	lapack.slartgp_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
	return cs_val, sn_val, r_val
}

// Generate robust Givens rotation (f64)
// Generates plane rotation with robustness against overflow
m_generate_robust_givens_f64 :: proc(f, g: f64) -> (cs, sn, r: f64) {
	cs_val, sn_val, r_val: f64
	f_val, g_val := f, g
	lapack.dlartgp_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
	return cs_val, sn_val, r_val
}

// Generate standard Givens rotation (f32)
// Generates plane rotation, choosing sign to avoid cancellation
m_generate_givens_f32 :: proc(f, g: f32) -> (cs, sn, r: f32) {
	cs_val, sn_val, r_val: f32
	f_val, g_val := f, g
	lapack.slartgs_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
	return cs_val, sn_val, r_val
}

// Generate standard Givens rotation (f64)
// Generates plane rotation, choosing sign to avoid cancellation
m_generate_givens_f64 :: proc(f, g: f64) -> (cs, sn, r: f64) {
	cs_val, sn_val, r_val: f64
	f_val, g_val := f, g
	lapack.dlartgs_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
	return cs_val, sn_val, r_val
}

// ===================================================================================
// MATRIX SCALING OPERATIONS
// ===================================================================================


// Scale matrix (f32)
m_scale_f32 :: proc(
	A: ^Matrix(f32),
	cfrom, cto: f32,
	scaling_type: MatrixScalingType = .General,
	kl: int = 0, // Lower bandwidth (for banded matrices)
	ku: int = 0, // Upper bandwidth (for banded matrices)
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	assert(len(A.data) > 0, "Matrix data cannot be empty")

	type_c := cast(u8)scaling_type
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	cfrom_val, cto_val := cfrom, cto

	lapack.slascl_(&type_c, &kl_val, &ku_val, &cfrom_val, &cto_val, &m, &n, raw_data(A.data), &lda, &info)

	return info == 0, info
}

// Scale matrix (f64)
m_scale_f64 :: proc(
	A: ^Matrix(f64),
	cfrom, cto: f64,
	scaling_type: MatrixScalingType = .General,
	kl: int = 0, // Lower bandwidth (for banded matrices)
	ku: int = 0, // Upper bandwidth (for banded matrices)
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	assert(len(A.data) > 0, "Matrix data cannot be empty")

	type_c := cast(u8)scaling_type
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	cfrom_val, cto_val := cfrom, cto

	lapack.dlascl_(&type_c, &kl_val, &ku_val, &cfrom_val, &cto_val, &m, &n, raw_data(A.data), &lda, &info)

	return info == 0, info
}

// Scale matrix (c64)
m_scale_c64 :: proc(
	A: ^Matrix(complex64),
	cfrom, cto: f32,
	scaling_type: MatrixScalingType = .General,
	kl: int = 0, // Lower bandwidth (for banded matrices)
	ku: int = 0, // Upper bandwidth (for banded matrices)
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	assert(len(A.data) > 0, "Matrix data cannot be empty")

	type_c := cast(u8)scaling_type
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	cfrom_val, cto_val := cfrom, cto

	lapack.clascl_(&type_c, &kl_val, &ku_val, &cfrom_val, &cto_val, &m, &n, cast(^complex64)raw_data(A.data), &lda, &info)

	return info == 0, info
}

// Scale matrix (c128)
m_scale_c128 :: proc(
	A: ^Matrix(complex128),
	cfrom, cto: f64,
	scaling_type: MatrixScalingType = .General,
	kl: int = 0, // Lower bandwidth (for banded matrices)
	ku: int = 0, // Upper bandwidth (for banded matrices)
) -> (
	success: bool,
	info: Info,
) {
	// Validate matrix
	assert(len(A.data) > 0, "Matrix data cannot be empty")

	type_c := cast(u8)scaling_type
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	cfrom_val, cto_val := cfrom, cto

	lapack.zlascl_(&type_c, &kl_val, &ku_val, &cfrom_val, &cto_val, &m, &n, cast(^complex128)raw_data(A.data), &lda, &info)

	return info == 0, info
}

// ===================================================================================
// CONVENIENCE FUNCTIONS FOR GIVENS ROTATIONS
// ===================================================================================

// Givens rotation parameters
GivensRotation :: struct($T: typeid) where is_float(T) {
	cs, sn: T, // Cosine and sine of rotation angle
	i, j:   int, // Indices of rows/columns to rotate
}

// Apply Givens rotation to vector pair
// Applies rotation [ cs  sn] [x] = [x']
//                  [-sn  cs] [y]   [y']
apply_givens_rotation :: proc(cs, sn: $T, x, y: ^T) {
	temp := cs * x^ + sn * y^
	y^ = -sn * x^ + cs * y^
	x^ = temp
}

// Create Givens rotation to zero out second component
// Returns rotation that transforms [a, b] -> [r, 0]
create_givens_to_zero :: proc(a, b: $T) -> (cs, sn, r: T) where is_float(T) {
	when T == f32 {
		return m_generate_robust_givens_f32(a, b)
	} else when T == f64 {
		return m_generate_robust_givens_f64(a, b)
	}
}

// Pivoting type for plane rotations
GivensPivot :: enum {
	Variable, // "V" - Variable pivot (rotations between i and i+1)
	Bottom, // "B" - Bottom pivot (rotations with last row/column)
	Top, // "T" - Top pivot (rotations with first row/column)
}

pivot_to_char :: proc(pivot: GivensPivot) -> u8 {
	switch pivot {
	case .Variable:
		return 'V'
	case .Bottom:
		return 'B'
	case .Top:
		return 'T'
	}
	unreachable()
}

pivot_to_cstring :: proc(pivot: GivensPivot) -> cstring {
	switch pivot {
	case .Variable:
		return "V"
	case .Bottom:
		return "B"
	case .Top:
		return "T"
	}
	unreachable()
}

// Direction for applying rotations
GivensDirection :: enum {
	Forward, // "F" - Forward (from first to last)
	Backward, // "B" - Backward (from last to first)
}

direction_to_char :: proc(dir: GivensDirection) -> u8 {
	switch dir {
	case .Forward:
		return 'F'
	case .Backward:
		return 'B'
	}
	unreachable()
}

direction_to_cstring :: proc(dir: GivensDirection) -> cstring {
	switch dir {
	case .Forward:
		return "F"
	case .Backward:
		return "B"
	}
	unreachable()
}


// Apply sequence of Givens rotations to matrix using LAPACK SLASR/DLASR (f32)
m_apply_givens_sequence_f32 :: proc(
	A: ^Matrix(f32),
	c, s: []f32, // Cosine and sine arrays
	side: ReflectorSide = .Left,
	pivot: GivensPivot = .Variable,
	direction: GivensDirection = .Forward,
) {
	assert(len(c) == len(s), "Cosine and sine arrays must have same length")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	side_c := cast(u8)side
	pivot_c := pivot_to_char(pivot)
	direct_c := direction_to_char(direction)

	lapack.slasr_(&side_c, &pivot_c, &direct_c, &m, &n, raw_data(c), raw_data(s), raw_data(A.data), &lda)
}

// Apply sequence of Givens rotations to matrix using LAPACK SLASR/DLASR (f64)
m_apply_givens_sequence_f64 :: proc(
	A: ^Matrix(f64),
	c, s: []f64, // Cosine and sine arrays
	side: ReflectorSide = .Left,
	pivot: GivensPivot = .Variable,
	direction: GivensDirection = .Forward,
) {
	assert(len(c) == len(s), "Cosine and sine arrays must have same length")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	side_c := cast(u8)side
	pivot_c := pivot_to_char(pivot)
	direct_c := direction_to_char(direction)

	lapack.dlasr_(&side_c, &pivot_c, &direct_c, &m, &n, raw_data(c), raw_data(s), raw_data(A.data), &lda)
}

// Apply sequence of Givens rotations to matrix using LAPACK CLASR (c64)
m_apply_givens_sequence_c64 :: proc(
	A: ^Matrix(complex64),
	c, s: []f32, // Cosine and sine arrays (real for complex matrices)
	side: ReflectorSide = .Left,
	pivot: GivensPivot = .Variable,
	direction: GivensDirection = .Forward,
) {
	assert(len(c) == len(s), "Cosine and sine arrays must have same length")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	side_c := cast(u8)side
	pivot_c := pivot_to_char(pivot)
	direct_c := direction_to_char(direction)

	lapack.clasr_(&side_c, &pivot_c, &direct_c, &m, &n, raw_data(c), raw_data(s), cast(^complex64)raw_data(A.data), &lda)
}

// Apply sequence of Givens rotations to matrix using LAPACK ZLASR (c128)
m_apply_givens_sequence_c128 :: proc(
	A: ^Matrix(complex128),
	c, s: []f64, // Cosine and sine arrays (real for complex matrices)
	side: ReflectorSide = .Left,
	pivot: GivensPivot = .Variable,
	direction: GivensDirection = .Forward,
) {
	assert(len(c) == len(s), "Cosine and sine arrays must have same length")

	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)

	side_c := cast(u8)side
	pivot_c := pivot_to_char(pivot)
	direct_c := direction_to_char(direction)

	lapack.zlasr_(&side_c, &pivot_c, &direct_c, &m, &n, raw_data(c), raw_data(s), cast(^complex128)raw_data(A.data), &lda)
}

// Apply sequence of Givens rotations to matrix (convenience wrapper)
// Applies rotations to consecutive rows/columns
apply_givens_sequence :: proc(A: ^Matrix($T), rotations: []GivensRotation(T), apply_to_rows: bool = true) where is_float(T) {
	if len(rotations) == 0 do return

	// Extract c and s arrays from rotations
	c := make([]T, len(rotations), context.temp_allocator)
	s := make([]T, len(rotations), context.temp_allocator)
	defer delete(c, context.temp_allocator)
	defer delete(s, context.temp_allocator)

	for i, rot in rotations {
		c[i] = rot.cs
		s[i] = rot.sn
	}

	// Use LAPACK for efficient application
	side := apply_to_rows ? ReflectorSide.Left : ReflectorSide.Right
	when T == f32 {
		m_apply_givens_sequence_f32(A, c, s, side, .Variable, .Forward)
	} else when T == f64 {
		m_apply_givens_sequence_f64(A, c, s, side, .Variable, .Forward)
	}
}

// ===================================================================================
// CONVENIENCE FUNCTIONS FOR MATRIX SCALING
// ===================================================================================

// Scale matrix to avoid over/underflow
// Automatically determines appropriate scaling factors
m_scale_safe :: proc(A: ^Matrix($T), scaling_type := MatrixScalingType.General, kl := 0, ku := 0) -> (success: bool, scale_factor: T) where is_float(T) || is_complex(T) {
	// Use machine parameters to determine safe scaling bounds
	when T == f32 || T == complex64 {
		safmin := f32(machine_parameter(f32, .SafeMin))
		safmax := f32(1.0 / safmin)
		scale_factor = f32(1.0)
	} else when T == f64 || T == complex128 {
		safmin := machine_parameter(f64, .SafeMin)
		safmax := 1.0 / safmin
		scale_factor = f64(1.0)
	}

	// Calculate current maximum value in matrix
	max_val := max_array(A.data[:A.rows * A.cols])

	// Determine if scaling is needed
	when T == f32 || T == complex64 {
		cfrom := f32(max_val)
		cto := f32(1.0)
		if max_val > safmax || max_val < safmin {
			// Need to scale
			if max_val > safmax {
				cto = safmax / f32(max_val)
			} else {
				cto = safmin / f32(max_val)
			}
			scale_factor = cto
			success, _ := m_scale_f32(A, cfrom, cto, scaling_type, kl, ku) if T == f32 else m_scale_c64(A, cfrom, cto, scaling_type, kl, ku)
			return success, T(scale_factor)
		}
	} else when T == f64 || T == complex128 {
		cfrom := max_val
		cto := 1.0
		if max_val > safmax || max_val < safmin {
			// Need to scale
			if max_val > safmax {
				cto = safmax / max_val
			} else {
				cto = safmin / max_val
			}
			scale_factor = cto
			success, _ := m_scale_f64(A, cfrom, cto, scaling_type, kl, ku) if T == f64 else m_scale_c128(A, cfrom, cto, scaling_type, kl, ku)
			return success, T(scale_factor)
		}
	}

	return true, scale_factor
}
