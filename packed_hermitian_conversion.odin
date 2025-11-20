package openblas

import lapack "./f77"
import "core:math"

// ===================================================================================
// PACKED HERMITIAN MATRIX CONVERSIONS
// ===================================================================================
//
// This file provides conversion operations between different storage formats for
// Hermitian matrices. Hermitian matrices satisfy A* = A (conjugate transpose equals
// original matrix).
//
// Conversions supported:
// - Full matrix ↔ Packed storage
// - Hermitian ↔ Real symmetric (real part extraction)
// - Hermitian → Imaginary antisymmetric (imaginary part extraction)

// ===================================================================================
// MEMORY LAYOUT CONVERSIONS
// ===================================================================================

// Convert full dense Hermitian matrix to packed storage
// Non-allocating version: requires pre-allocated packed array
pack_herm_from_dns :: proc(
    A: []$T, // Full matrix (n×n) in column-major order
    AP: []T, // Pre-allocated packed array (n*(n+1)/2 elements)
    n: int, // Matrix dimension
    lda: int, // Leading dimension of A
    uplo: MatrixRegion = .Upper,
) where is_complex(T) {
    assert(validate_packed_storage(n, len(AP)), "Packed array too small")
    assert(len(A) >= n * lda, "Full matrix array too small")

    idx := 0
    switch uplo {
    case .Upper:
        // Pack upper triangle column by column
        for j in 0 ..< n {
            for i in 0 ..= j {
                AP[idx] = A[i + j * lda]
                idx += 1
            }
        }
    case .Lower:
        // Pack lower triangle column by column
        for j in 0 ..< n {
            for i in j ..< n {
                AP[idx] = A[i + j * lda]
                idx += 1
            }
        }
    case .Full:
        panic("Full storage not supported for packed format")
    }
}

// Convert packed storage to full dense Hermitian matrix
// Non-allocating version: requires pre-allocated full matrix
dns_from_pack_herm :: proc(
    AP: []$T, // Packed array
    A: []T, // Pre-allocated full matrix (n×n)
    n: int, // Matrix dimension
    lda: int, // Leading dimension of A
    uplo: MatrixRegion = .Upper,
) where is_complex(T) {
    assert(validate_packed_storage(n, len(AP)), "Packed array too small")
    assert(len(A) >= n * lda, "Full matrix array too small")

    // Initialize full matrix to zero
    for i in 0 ..< n * lda {
        A[i] = T(0)
    }

    idx := 0
    switch uplo {
    case .Upper:
        // Unpack upper triangle and conjugate transpose to lower
        for j in 0 ..< n {
            for i in 0 ..= j {
                val := AP[idx]
                A[i + j * lda] = val
                if i != j {
                    // Hermitian property: A[j,i] = conj(A[i,j])
                    A[j + i * lda] = conj(val)
                }
                idx += 1
            }
        }
    case .Lower:
        // Unpack lower triangle and conjugate transpose to upper
        for j in 0 ..< n {
            for i in j ..< n {
                val := AP[idx]
                A[i + j * lda] = val
                if i != j {
                    // Hermitian property: A[j,i] = conj(A[i,j])
                    A[j + i * lda] = conj(val)
                }
                idx += 1
            }
        }
    case .Full:
        panic("Full storage not supported for packed format")
    }
}

// ===================================================================================
// ALLOCATING CONVERSION FUNCTIONS
// ===================================================================================

// Create PackedHermitian from full matrix (allocating version for convenience)
pack_herm_make :: proc(
    A: []$T, // Full matrix
    n: int, // Matrix dimension
    lda: int, // Leading dimension of A
    uplo: MatrixRegion = .Upper,
    allocator := context.allocator,
) -> PackedHermitian(T) where is_complex(T) {
    packed_size := packed_storage_size(n)
    data := make([]T, packed_size, allocator)

    pack_herm_from_dns(A, data, n, lda, uplo)

    return PackedHermitian(T){data = data, n = n, uplo = uplo}
}

// Extract full dense matrix from PackedHermitian (allocating version for convenience)
dns_from_pack_herm_alloc :: proc(
    packed: ^PackedHermitian($T),
    lda: int, // Leading dimension for output
    allocator := context.allocator,
) -> []T where is_complex(T) {
    if lda < packed.n {
        panic("Leading dimension too small")
    }

    A := make([]T, packed.n * lda, allocator)
    dns_from_pack_herm(packed.data, A, packed.n, lda, packed.uplo)

    return A
}

// ===================================================================================
// HERMITIAN TO REAL/IMAGINARY CONVERSIONS
// ===================================================================================

// Convert packed Hermitian matrix to real symmetric (for real part analysis)
// Generic implementation: extracts real part of complex64 → f32 or complex128 → f64
pack_herm_extract_real :: proc(
    packed: ^PackedHermitian($Cmplx),
    $Real: typeid,
    allocator := context.allocator,
) -> PackedSymmetric(Real) where (Cmplx == complex64 && Real == f32) ||
    (Cmplx == complex128 && Real == f64) {
    real_data := make([]Real, len(packed.data), allocator)
    for val, i in packed.data {
        real_data[i] = real(val)
    }

    return PackedSymmetric(Real){data = real_data, n = packed.n, uplo = packed.uplo}
}

// Convert packed Hermitian matrix to imaginary antisymmetric (for imaginary part analysis)
// Generic implementation: extracts imaginary part of complex64 → f32 or complex128 → f64
pack_herm_extract_imag :: proc(
    packed: ^PackedHermitian($Cmplx),
    $Real: typeid,
    allocator := context.allocator,
) -> []Real where (Cmplx == complex64 && Real == f32) ||
    (Cmplx == complex128 && Real == f64) {
    imag_data := make([]Real, len(packed.data), allocator)
    for val, i in packed.data {
        imag_data[i] = imag(val)
    }

    return imag_data
}
