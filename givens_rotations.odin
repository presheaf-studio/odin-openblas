package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// GIVENS ROTATION GENERATION
// ===================================================================================

// Generate robust Givens rotation
// Generates plane rotation with robustness against overflow
generate_robust_givens :: proc(f, g: $T) -> (cs, sn, r: T) where is_float(T) {
    cs_val, sn_val, r_val: T
    f_val, g_val := f, g
    when T == f32 {
        lapack.slartgp_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
    } else when T == f64 {
        lapack.dlartgp_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
    }
    return cs_val, sn_val, r_val
}

// Generate standard Givens rotation
// Generates plane rotation, choosing sign to avoid cancellation
generate_givens :: proc(f, g: $T) -> (cs, sn, r: T) where is_float(T) {
    cs_val, sn_val, r_val: T
    f_val, g_val := f, g
    when T == f32 {
        lapack.slartgs_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
    } else when T == f64 {
        lapack.dlartgs_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
    }
    return cs_val, sn_val, r_val
}

// ===================================================================================
// MATRIX SCALING OPERATIONS
// ===================================================================================

dns_scale :: proc(
    A: ^Matrix($Real_Or_Cmplx),
    cfrom, cto: $Real,
    scaling_type: MatrixScalingType = .General,
    kl: int = 0,
    ku: int = 0,
) -> (
    info: Info,
    ok: bool,
) where (Real_Or_Cmplx == Real) ||
    (Real_Or_Cmplx == complex64 && Real == f32) ||
    (Real_Or_Cmplx == complex128 && Real == f64) {
    // Validate matrix
    assert(len(A.data) > 0, "Matrix data cannot be empty")

    type_c := cast(u8)scaling_type
    kl_val := Blas_Int(kl)
    ku_val := Blas_Int(ku)
    m := Blas_Int(A.rows)
    n := Blas_Int(A.cols)
    lda := Blas_Int(A.ld)
    cfrom_val, cto_val := cfrom, cto

    when T == f32 {
        lapack.slascl_(&type_c, &kl_val, &ku_val, &cfrom_val, &cto_val, &m, &n, raw_data(A.data), &lda, &info)
    } else when T == f64 {
        lapack.dlascl_(&type_c, &kl_val, &ku_val, &cfrom_val, &cto_val, &m, &n, raw_data(A.data), &lda, &info)
    } else when Cmplx == complex64 {
        lapack.clascl_(&type_c, &kl_val, &ku_val, &cfrom_val, &cto_val, &m, &n, raw_data(A.data), &lda, &info)
    } else when Cmplx == complex128 {
        lapack.zlascl_(&type_c, &kl_val, &ku_val, &cfrom_val, &cto_val, &m, &n, raw_data(A.data), &lda, &info)
    }

    return info, info == 0
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
    return generate_robust_givens(a, b)
}

// Pivoting type for plane rotations
GivensPivot :: enum {
    Variable, // "V" - Variable pivot (rotations between i and i+1)
    Bottom, // "B" - Bottom pivot (rotations with last row/column)
    Top, // "T" - Top pivot (rotations with first row/column)
}

// Direction for applying rotations
GivensDirection :: enum {
    Forward, // "F" - Forward (from first to last)
    Backward, // "B" - Backward (from last to first)
}
apply_givens_sequence :: proc {
    apply_givens_sequence_real,
    apply_givens_sequence_complex,
}

apply_givens_sequence_real :: proc(
    A: ^Matrix($T),
    c, s: []T,
    side: ReflectorSide = .Left,
    pivot: GivensPivot = .Variable,
    direction: GivensDirection = .Forward,
) where is_float(T) {
    assert(len(c) == len(s), "Cosine and sine arrays must have same length")

    m := Blas_Int(A.rows)
    n := Blas_Int(A.cols)
    lda := Blas_Int(A.ld)

    side_c := cast(u8)side
    pivot_c := cast(u8)pivot
    direct_c := cast(u8)direction

    when T == f32 {
        lapack.slasr_(&side_c, &pivot_c, &direct_c, &m, &n, raw_data(c), raw_data(s), raw_data(A.data), &lda)
    } else when T == f64 {
        lapack.dlasr_(&side_c, &pivot_c, &direct_c, &m, &n, raw_data(c), raw_data(s), raw_data(A.data), &lda)
    }
}

apply_givens_sequence_complex :: proc(
    A: ^Matrix($Cmplx),
    c, s: []$Real,
    side: ReflectorSide = .Left,
    pivot: GivensPivot = .Variable,
    direction: GivensDirection = .Forward,
) where (Cmplx == complex64 && Real == f32) ||
    (Cmplx == complex128 && Real == f64) {
    assert(len(c) == len(s), "Cosine and sine arrays must have same length")

    m := Blas_Int(A.rows)
    n := Blas_Int(A.cols)
    lda := Blas_Int(A.ld)

    side_c := cast(u8)side
    pivot_c := cast(u8)pivot
    direct_c := cast(u8)direction

    when Cmplx == complex64 {
        lapack.clasr_(
            &side_c,
            &pivot_c,
            &direct_c,
            &m,
            &n,
            raw_data(c),
            raw_data(s),
            cast(^complex64)raw_data(A.data),
            &lda,
        )
    } else when Cmplx == complex128 {
        lapack.zlasr_(
            &side_c,
            &pivot_c,
            &direct_c,
            &m,
            &n,
            raw_data(c),
            raw_data(s),
            cast(^complex128)raw_data(A.data),
            &lda,
        )
    }
}


// ===================================================================================
// CONVENIENCE FUNCTIONS FOR MATRIX SCALING
// ===================================================================================

// Scale matrix to avoid over/underflow
// Automatically determines appropriate scaling factors
dns_scale_safe :: proc(
    A: ^Matrix($T),
    scaling_type := MatrixScalingType.General,
    kl := 0,
    ku := 0,
) -> (
    success: bool,
    scale_factor: T,
) where is_float(T) ||
    is_complex(T) {
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
            success, _ :=
                m_scale_f32(A, cfrom, cto, scaling_type, kl, ku) if T == f32 else m_scale_c64(A, cfrom, cto, scaling_type, kl, ku)
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
            success, _ :=
                m_scale_f64(A, cfrom, cto, scaling_type, kl, ku) if T == f64 else m_scale_c128(A, cfrom, cto, scaling_type, kl, ku)
            return success, T(scale_factor)
        }
    }

    return true, scale_factor
}
