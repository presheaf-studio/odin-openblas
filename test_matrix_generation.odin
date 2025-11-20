package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"
import "core:slice"

// ===================================================================================
// LAPACK TEST MATRIX GENERATION
// ===================================================================================

generate_test_matrix :: proc {
    generate_test_matrix_real,
    generate_test_matrix_complex,
}

generate_test_matrix_eigenvalues :: proc {
    generate_test_matrix_eigenvalues_real,
    generate_test_matrix_eigenvalues_complex,
}

// ===================================================================================
// TEST MATRIX GENERATION PARAMETERS
// ===================================================================================

// Comprehensive parameters for test matrix generation
TestMatrixParams :: struct {
    // Matrix dimensions
    rows, cols:       int,

    // Random distribution
    distribution:     RandomDistribution,

    // Random seed (4 integers)
    seed:             [4]int,

    // Matrix structure
    symmetry:         MatrixSymmetry,

    // Singular values/eigenvalues
    singular_values:  []f64, // D array - controls spectrum

    // Condition number control
    mode:             int, // Type of matrix to generate (1-6)
    condition_number: f64, // Target condition number
    max_element:      f64, // Maximum element magnitude

    // Banding parameters
    lower_bandwidth:  int, // kl - lower bandwidth
    upper_bandwidth:  int, // ku - upper bandwidth

    // Storage format
    packing:          MatrixPacking,
}

// Default test matrix parameters
default_test_matrix_params :: proc() -> TestMatrixParams {
    params := TestMatrixParams {
        rows             = 10,
        cols             = 10,
        distribution     = .Uniform,
        seed             = {1, 2, 3, 4},
        symmetry         = .None,
        singular_values  = nil,
        mode             = 1,
        condition_number = 1.0,
        max_element      = 1.0,
        lower_bandwidth  = 0,
        upper_bandwidth  = 0,
        packing          = .No_Packing,
    }
    return params
}

// ===================================================================================
// TEST MATRIX GENERATION IMPLEMENTATION
// ===================================================================================

// Generate test matrix for f32 and f64
generate_test_matrix_real :: proc(
    A: ^Matrix($T),
    params: TestMatrixParams,
    allocator := context.allocator,
) -> (
    success: bool,
    info: Info,
) where is_float(T) {
    context.allocator = allocator
    // Validate parameters
    assert(
        params.distribution != .ComplexUniform,
        "ComplexUniform distribution is not supported by DLATMS/SLATMS - use Uniform, UniformMinus1To1, or Normal",
    )
    assert(A.rows == params.rows && A.cols == params.cols, "Matrix dimensions must match parameters")

    // Prepare parameters
    m := Blas_Int(params.rows)
    n := Blas_Int(params.cols)
    dist_c := cast(u8)params.distribution

    // Setup seed array
    iseed := make([]Blas_Int, 4, context.allocator)
    for i in 0 ..< 4 {
        iseed[i] = Blas_Int(params.seed[i])
    }
    defer delete(iseed, context.allocator)

    sym_c := cast(u8)params.symmetry

    // Setup singular values array
    D: []T
    if params.singular_values != nil {
        D = make([]T, len(params.singular_values), context.allocator)
        for i in 0 ..< len(params.singular_values) {
            D[i] = T(params.singular_values[i])
        }
    } else {
        D = make([]T, max(params.rows, params.cols), context.allocator)
        // Initialize with default values
        for i in 0 ..< len(D) {
            D[i] = 1.0
        }
    }
    defer delete(D, context.allocator)

    mode := Blas_Int(params.mode)
    cond := T(params.condition_number)
    dmax := T(params.max_element)
    kl := Blas_Int(params.lower_bandwidth)
    ku := Blas_Int(params.upper_bandwidth)
    pack_c := cast(u8)params.packing
    lda := A.ld

    // Allocate workspace
    work := make([]T, 3 * max(params.rows, params.cols), context.allocator)
    defer delete(work, context.allocator)

    info_val: Info
    when T == f32 {
        lapack.slatms_(
            &m,
            &n,
            &dist_c,
            raw_data(iseed),
            &sym_c,
            raw_data(D),
            &mode,
            &cond,
            &dmax,
            &kl,
            &ku,
            &pack_c,
            raw_data(A.data),
            &lda,
            raw_data(work),
            &info_val,
        )
    } else when T == f64 {
        lapack.dlatms_(
            &m,
            &n,
            &dist_c,
            raw_data(iseed),
            &sym_c,
            raw_data(D),
            &mode,
            &cond,
            &dmax,
            &kl,
            &ku,
            &pack_c,
            raw_data(A.data),
            &lda,
            raw_data(work),
            &info_val,
        )
    }

    return info_val == 0, info_val
}

// Generate test matrix for complex64 and complex128
generate_test_matrix_complex :: proc(
    A: ^Matrix($T),
    params: TestMatrixParams,
) -> (
    success: bool,
    info: Info,
) where is_complex(T) {
    Real :: real_type_of(T)

    // Validate parameters
    assert(
        params.distribution != .ComplexUniform,
        "ComplexUniform distribution is not supported by CLATMS/ZLATMS - use Uniform, UniformMinus1To1, or Normal",
    )
    assert(A.rows == params.rows && A.cols == params.cols, "Matrix dimensions must match parameters")

    // Prepare parameters
    m := Blas_Int(params.rows)
    n := Blas_Int(params.cols)
    dist_c := cast(u8)params.distribution

    // Setup seed array
    iseed := make([]Blas_Int, 4, context.allocator)
    for i in 0 ..< 4 {
        iseed[i] = Blas_Int(params.seed[i])
    }
    defer delete(iseed, context.allocator)

    sym_c := cast(u8)params.symmetry

    // Setup singular values array (real for complex matrices)
    D: []Real
    if params.singular_values != nil {
        D = make([]Real, len(params.singular_values), context.allocator)
        for i in 0 ..< len(params.singular_values) {
            D[i] = Real(params.singular_values[i])
        }
    } else {
        D = make([]Real, max(params.rows, params.cols), context.allocator)
        // Initialize with default values
        for i in 0 ..< len(D) {
            D[i] = 1.0
        }
    }
    defer delete(D, context.allocator)

    mode := Blas_Int(params.mode)
    cond := Real(params.condition_number)
    dmax := Real(params.max_element)
    kl := Blas_Int(params.lower_bandwidth)
    ku := Blas_Int(params.upper_bandwidth)
    pack_c := cast(u8)params.packing
    lda := A.ld

    // Allocate workspace
    work := make([]T, 3 * max(params.rows, params.cols), context.allocator)
    defer delete(work, context.allocator)

    info_val: Info
    when T == complex64 {
        lapack.clatms_(
            &m,
            &n,
            &dist_c,
            raw_data(iseed),
            &sym_c,
            raw_data(D),
            &mode,
            &cond,
            &dmax,
            &kl,
            &ku,
            &pack_c,
            raw_data(A.data),
            &lda,
            raw_data(work),
            &info_val,
        )
    } else when T == complex128 {
        lapack.zlatms_(
            &m,
            &n,
            &dist_c,
            raw_data(iseed),
            &sym_c,
            raw_data(D),
            &mode,
            &cond,
            &dmax,
            &kl,
            &ku,
            &pack_c,
            raw_data(A.data),
            &lda,
            raw_data(work),
            &info_val,
        )
    }

    return info_val == 0, info_val
}

// ===================================================================================
// HILBERT MATRIX GENERATION
// ===================================================================================

// Generate Hilbert matrix for f32 and f64
// Hilbert matrix H[i,j] = 1/(i+j+1) - ill-conditioned test matrix
generate_test_hilbert :: proc(A: ^Matrix($T)) where is_float(T) || is_complex(T) {
    assert(A.rows == A.cols, "Hilbert matrix must be square")

    n := A.rows
    for i in 0 ..< n {
        for j in 0 ..< n {
            when is_float(T) {
                A.data[j * A.ld + i] = T(1.0) / T(i + j + 1)
            } else when is_complex(T) {
                A.data[j * A.ld + i] = T(complex(Real(1.0) / Real(i + j + 1), 0))
            }
        }
    }
}

// ===================================================================================
// EIGENVALUE TEST MATRIX GENERATION
// ===================================================================================

// Generate test matrix with specified eigenvalues (f32/f64)
generate_test_matrix_eigenvalues_real :: proc(
    A: ^Matrix($T),
    eigenvalues: []T,
    symmetric: bool = true,
    seed: [4]int = {1, 2, 3, 4},
) -> (
    success: bool,
    info: Info,
) where is_float(T) {
    assert(A.rows == A.cols, "Eigenvalue test matrix must be square")
    assert(len(eigenvalues) == A.rows, "Number of eigenvalues must match matrix size")

    n := A.rows
    params := TestMatrixParams {
        rows             = n,
        cols             = n,
        distribution     = .Uniform,
        seed             = seed,
        symmetry         = symmetric ? .Symmetric : .None,
        singular_values  = make([]f64, len(eigenvalues)),
        mode             = 1,
        condition_number = 1.0,
        max_element      = 1.0,
        lower_bandwidth  = 0,
        upper_bandwidth  = 0,
        packing          = .No_Packing,
    }

    // Copy eigenvalues to singular_values array
    for i in 0 ..< len(eigenvalues) {
        params.singular_values[i] = f64(eigenvalues[i])
    }
    defer delete(params.singular_values)

    return generate_test_matrix_real(A, params)
}

// Generate test matrix with specified eigenvalues (c64/c128)
generate_test_matrix_eigenvalues_complex :: proc(
    A: ^Matrix($T),
    eigenvalues: []$Real,
    hermitian: bool = true,
    seed: [4]int = {1, 2, 3, 4},
) -> (
    success: bool,
    info: Info,
) where is_complex(T),
    Real ==
    real_type_of(T) {
    assert(A.rows == A.cols, "Eigenvalue test matrix must be square")
    assert(len(eigenvalues) == A.rows, "Number of eigenvalues must match matrix size")

    n := A.rows
    params := TestMatrixParams {
        rows             = n,
        cols             = n,
        distribution     = .Uniform,
        seed             = seed,
        symmetry         = hermitian ? .Hermitian : .None,
        singular_values  = make([]f64, len(eigenvalues)),
        mode             = 1,
        condition_number = 1.0,
        max_element      = 1.0,
        lower_bandwidth  = 0,
        upper_bandwidth  = 0,
        packing          = .No_Packing,
    }

    // Copy eigenvalues to singular_values array
    for i in 0 ..< len(eigenvalues) {
        params.singular_values[i] = f64(eigenvalues[i])
    }
    defer delete(params.singular_values)

    return generate_test_matrix_complex(A, params)
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Generate identity matrix
generate_identity :: proc(A: ^Matrix($T)) where is_float(T) || is_complex(T) {
    assert(A.rows == A.cols, "Identity matrix must be square")
    initialize_matrix(A, 0, 1, .Full)
}

// Generate random matrix with uniform distribution
generate_random_uniform :: proc(
    A: ^Matrix($T),
    seed: [4]int = {1, 2, 3, 4},
) -> (
    success: bool,
    info: Info,
) where is_float(T) ||
    is_complex(T) {
    params := TestMatrixParams {
        rows             = A.rows,
        cols             = A.cols,
        distribution     = .Uniform,
        seed             = seed,
        symmetry         = .None,
        singular_values  = nil,
        mode             = 1,
        condition_number = 1.0,
        max_element      = 1.0,
        lower_bandwidth  = 0,
        upper_bandwidth  = 0,
        packing          = .No_Packing,
    }

    when is_float(T) {
        return generate_test_matrix_real(A, params)
    } else when is_complex(T) {
        return generate_test_matrix_complex(A, params)
    }
}

// Generate ill-conditioned matrix with specified condition number
generate_ill_conditioned :: proc(
    A: ^Matrix($T),
    condition_number: f64,
    symmetric: bool = false,
    seed: [4]int = {1, 2, 3, 4},
) -> (
    success: bool,
    info: Info,
) where is_float(T) ||
    is_complex(T) {
    params := TestMatrixParams {
        rows             = A.rows,
        cols             = A.cols,
        distribution     = .Uniform,
        seed             = seed,
        symmetry         = symmetric ? (.Symmetric if is_float(T) else .Hermitian) : .None,
        singular_values  = nil,
        mode             = 1,
        condition_number = condition_number,
        max_element      = 1.0,
        lower_bandwidth  = 0,
        upper_bandwidth  = 0,
        packing          = .No_Packing,
    }

    when is_float(T) {
        return generate_test_matrix_real(A, params)
    } else when is_complex(T) {
        return generate_test_matrix_complex(A, params)
    }
}

// ===================================================================================
// GENERAL MATRIX GENERATION WITH SPECIFIED EIGENVALUES (LAGGE)
// ===================================================================================

// Generate general test matrix with specified eigenvalues proc group
generate_general_eigenvalues :: proc {
    generate_general_eigenvalues_impl,
}

// Query workspace for general matrix generation
query_workspace_lagge :: proc(m, n: int) -> (work_size: int) {
    return max(m, n)
}

// Generate general matrix with specified eigenvalues
// A is m×n, D contains eigenvalues, kl and ku specify bandwidths
// For real types, D is same type as matrix elements
// For complex types, D is real (eigenvalues are real-valued)
generate_general_eigenvalues_impl :: proc(
    A: ^Matrix($T), // Output matrix (m × n)
    D: []$Real, // Eigenvalues (size min(m,n)) - real for complex, same type for real
    kl: int = 0, // Lower bandwidth
    ku: int = 0, // Upper bandwidth
    work: []T, // Workspace (size >= max(m,n))
    seed: [4]int = {1, 2, 3, 4},
) -> (
    info: Info,
    ok: bool,
) where (T == f32 && Real == f32) || (T == f64 && Real == f64) || (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
    m := A.rows
    n := A.cols
    min_mn := min(m, n)

    assert(len(D) >= min_mn, "D array too small")
    assert(len(work) >= max(m, n), "work array too small")

    m_int := Blas_Int(m)
    n_int := Blas_Int(n)
    kl_int := Blas_Int(kl)
    ku_int := Blas_Int(ku)
    lda := A.ld

    // Setup seed array
    iseed := make([]Blas_Int, 4, context.allocator)
    for i in 0 ..< 4 {
        iseed[i] = Blas_Int(seed[i])
    }
    defer delete(iseed, context.allocator)

    when T == f32 {
        lapack.slagge_(
            &m_int,
            &n_int,
            &kl_int,
            &ku_int,
            raw_data(D),
            raw_data(A.data),
            &lda,
            raw_data(iseed),
            raw_data(work),
            &info,
        )
    } else when T == f64 {
        lapack.dlagge_(
            &m_int,
            &n_int,
            &kl_int,
            &ku_int,
            raw_data(D),
            raw_data(A.data),
            &lda,
            raw_data(iseed),
            raw_data(work),
            &info,
        )
    } else when T == complex64 {
        lapack.clagge_(
            &m_int,
            &n_int,
            &kl_int,
            &ku_int,
            raw_data(D),
            raw_data(A.data),
            &lda,
            raw_data(iseed),
            raw_data(work),
            &info,
        )
    } else when T == complex128 {
        lapack.zlagge_(
            &m_int,
            &n_int,
            &kl_int,
            &ku_int,
            raw_data(D),
            raw_data(A.data),
            &lda,
            raw_data(iseed),
            raw_data(work),
            &info,
        )
    }

    return info, info == 0
}

// ===================================================================================
// SYMMETRIC MATRIX GENERATION WITH SPECIFIED EIGENVALUES (LAGSY)
// ===================================================================================

// Query workspace for symmetric matrix generation
query_workspace_lagsy :: proc(n: int) -> (work_size: int) {
    return 2 * n
}

// Generate symmetric matrix with specified eigenvalues
// A is n×n symmetric, D contains eigenvalues, k is number of nonzero off-diagonals
// For real types, D is same type as matrix elements
// For complex types, D is real (eigenvalues are real-valued)
generate_symmetric_eigenvalues :: proc(
    A: ^Matrix($T), // Output symmetric matrix (n × n)
    D: []$Real, // Eigenvalues (size n) - real for complex, same type for real
    k: int = 0, // Number of nonzero off-diagonals (0 = diagonal)
    work: []T, // Workspace (size >= 2*n)
    seed: [4]int = {1, 2, 3, 4},
) -> (
    info: Info,
    ok: bool,
) where (T == f32 && Real == f32) || (T == f64 && Real == f64) || (T == complex64 && Real == f32) || (T == complex128 && Real == f64) {
    n := A.rows
    assert(A.cols == n, "Matrix must be square")
    assert(len(D) >= n, "D array too small")
    assert(len(work) >= 2 * n, "work array too small")

    n_int := Blas_Int(n)
    k_int := Blas_Int(k)
    lda := A.ld

    // Setup seed array
    iseed := make([]Blas_Int, 4, context.allocator)
    for i in 0 ..< 4 {
        iseed[i] = Blas_Int(seed[i])
    }
    defer delete(iseed, context.allocator)

    when T == f32 {
        lapack.slagsy_(&n_int, &k_int, raw_data(D), raw_data(A.data), &lda, raw_data(iseed), raw_data(work), &info)
    } else when T == f64 {
        lapack.dlagsy_(&n_int, &k_int, raw_data(D), raw_data(A.data), &lda, raw_data(iseed), raw_data(work), &info)
    } else when T == complex64 {
        lapack.clagsy_(&n_int, &k_int, raw_data(D), raw_data(A.data), &lda, raw_data(iseed), raw_data(work), &info)
    } else when T == complex128 {
        lapack.zlagsy_(&n_int, &k_int, raw_data(D), raw_data(A.data), &lda, raw_data(iseed), raw_data(work), &info)
    }

    return info, info == 0
}

// ===================================================================================
// HERMITIAN MATRIX GENERATION WITH SPECIFIED EIGENVALUES (LAGHE)
// ===================================================================================

// Query workspace for Hermitian matrix generation
query_workspace_laghe :: proc(n: int) -> (work_size: int) {
    return 2 * n
}

// Generate Hermitian matrix with specified eigenvalues (c64)
// A is n×n Hermitian, D contains eigenvalues (real), k is number of nonzero off-diagonals
generate_hermitian_eigenvalues :: proc(
    A: ^Matrix($Cmplx), // Output Hermitian matrix (n × n)
    D: []f32, // Eigenvalues (size n) - real values
    k: int = 0, // Number of nonzero off-diagonals (0 = diagonal)
    // Workspace (size >= 2*n)
    seed: [4]int = {1, 2, 3, 4},
) -> (
    info: Info,
    ok: bool,
) {
    n := A.rows
    assert(A.cols == n, "Matrix must be square")
    assert(len(D) >= int(n), "D array too small")
    assert(len(work) >= int(2 * n), "work array too small")

    n_int := Blas_Int(n)
    k_int := Blas_Int(k)
    lda := A.ld

    // Setup seed array
    iseed := make([]Blas_Int, 4, context.allocator)
    for i in 0 ..< 4 {
        iseed[i] = Blas_Int(seed[i])
    }
    defer delete(iseed, context.allocator)

    work := make([]Cmplx, 2 * n)
    defer delete(work)

    when Cmplx == complex64 {
        lapack.claghe_(&n_int, &k_int, raw_data(D), raw_data(A.data), &lda, raw_data(iseed), raw_data(work), &info)
    } else when Cmplx == complex128 {
        lapack.zlaghe_(&n_int, &k_int, raw_data(D), raw_data(A.data), &lda, raw_data(iseed), raw_data(work), &info)
    }

    return info, info == 0
}

// ===================================================================================
// RANDOM ORTHOGONAL/UNITARY MATRIX GENERATION (LAROR)
// ===================================================================================
// Query workspace for random orthogonal matrix generation
query_workspace_laror :: proc(m, n: int, side: OrthogonalSide) -> (work_size: int) {
    if side == .Left {
        return n
    } else {
        return m
    }
}

// Generate random orthogonal/unitary matrix for testing
// Multiplies A by a random orthogonal (real) or unitary (complex) matrix
generate_test_random_orthogonal :: proc(
    A: ^Matrix($T), // Matrix to multiply (m × n)
    X: []T, // Workspace for random vector (size n if Left, m if Right)
    side: OrthogonalSide = .Left,
    init: OrthogonalInit = .Identity,
    seed: [4]int = {1, 2, 3, 4},
) -> (
    info: Info,
    ok: bool,
) where is_float(T) || is_complex(T) {
    m := A.rows
    n := A.cols

    work_size := n if side == .Left else m
    assert(len(X) >= work_size, "X array too small")

    m_int := Blas_Int(m)
    n_int := Blas_Int(n)
    lda := A.ld

    side_c := cast(u8)side
    init_c := cast(u8)init

    // Setup seed array
    iseed := make([]Blas_Int, 4, context.allocator)
    for i in 0 ..< 4 {
        iseed[i] = Blas_Int(seed[i])
    }
    defer delete(iseed, context.allocator)

    when T == f32 {
        lapack.slaror_(
            &side_c,
            &init_c,
            &m_int,
            &n_int,
            raw_data(A.data),
            &lda,
            raw_data(iseed),
            raw_data(X),
            &info,
            1,
            1,
        )
    } else when T == f64 {
        lapack.dlaror_(
            &side_c,
            &init_c,
            &m_int,
            &n_int,
            raw_data(A.data),
            &lda,
            raw_data(iseed),
            raw_data(X),
            &info,
            1,
            1,
        )
    } else when T == complex64 {
        lapack.claror_(
            &side_c,
            &init_c,
            &m_int,
            &n_int,
            raw_data(A.data),
            &lda,
            raw_data(iseed),
            raw_data(X),
            &info,
            1,
            1,
        )
    } else when T == complex128 {
        lapack.zlaror_(
            &side_c,
            &init_c,
            &m_int,
            &n_int,
            raw_data(A.data),
            &lda,
            raw_data(iseed),
            raw_data(X),
            &info,
            1,
            1,
        )
    }

    return info, info == 0
}

// ===================================================================================
// APPLY ROTATION TO MATRIX ROWS/COLUMNS (LAROT)
// ===================================================================================
// Apply rotation to matrix rows or columns
// Applies a plane rotation to rows or columns of A
// For real types: c and s are both real
// For complex types: c is real, s is complex
apply_test_rotation :: proc(
    A: ^Matrix($T), // Matrix (m × n or nl×nl depending on lrows)
    c: $CType, // Cosine of rotation angle (real for all types)
    s: $SType, // Sine of rotation angle (real for real types, complex for complex types)
    lrows: bool = true, // true = rotate rows, false = rotate columns
    lleft: bool = false, // Include left end point in rotation
    lright: bool = false, // Include right end point in rotation
    nl: int = 0, // Number of rows/columns to rotate (0 = all)
    xleft: ^T = nil, // Left extra element for rotation
    xright: ^T = nil, // Right extra element for rotation
) where (T == f32 && CType == f32 && SType == f32) || (T == f64 && CType == f64 && SType == f64) || (T == complex64 && CType == f32 && SType == complex64) || (T == complex128 && CType == f64 && SType == complex128) {
    lrows_int: Blas_Int = 1 if lrows else 0
    lleft_int: Blas_Int = 1 if lleft else 0
    lright_int: Blas_Int = 1 if lright else 0

    nl_val := nl
    if nl == 0 {
        nl_val = A.cols if lrows else A.rows
    }
    nl_int := Blas_Int(nl_val)

    lda := A.ld

    c_val := c
    s_val := s

    xleft_ptr := xleft
    xright_ptr := xright

    when T == f32 {
        lapack.slarot_(
            &lrows_int,
            &lleft_int,
            &lright_int,
            &nl_int,
            &c_val,
            &s_val,
            raw_data(A.data),
            &lda,
            xleft_ptr,
            xright_ptr,
        )
    } else when T == f64 {
        lapack.dlarot_(
            &lrows_int,
            &lleft_int,
            &lright_int,
            &nl_int,
            &c_val,
            &s_val,
            raw_data(A.data),
            &lda,
            xleft_ptr,
            xright_ptr,
        )
    } else when T == complex64 {
        lapack.clarot_(
            &lrows_int,
            &lleft_int,
            &lright_int,
            &nl_int,
            &c_val,
            &s_val,
            raw_data(A.data),
            &lda,
            xleft_ptr,
            xright_ptr,
        )
    } else when T == complex128 {
        lapack.zlarot_(
            &lrows_int,
            &lleft_int,
            &lright_int,
            &nl_int,
            &c_val,
            &s_val,
            raw_data(A.data),
            &lda,
            xleft_ptr,
            xright_ptr,
        )
    }
}
