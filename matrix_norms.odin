package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import "core:c"
import "core:math"
import "core:mem"

// ===================================================================================
// MATRIX NORM COMPUTATION
// ===================================================================================

// Compute banded matrix norm proc group
norm_banded :: proc {
	norm_banded_f32_c64,
	norm_banded_f64_c128,
}

// Compute general matrix norm proc group
norm_general :: proc {
	norm_general_f32_c64,
	norm_general_f64_c128,
}

// Compute tridiagonal matrix norm proc group
norm_tridiagonal :: proc {
	norm_tridiagonal_f32_c64,
	norm_tridiagonal_f64_c128,
}

// Compute Hermitian banded matrix norm proc group
norm_banded_hermitian :: proc {
	norm_banded_hermitian_c64,
	norm_banded_hermitian_c128,
}

// Compute Hermitian matrix norm proc group
norm_hermitian :: proc {
	norm_hermitian_c64,
	norm_hermitian_c128,
}

// Compute Hermitian packed matrix norm proc group
norm_hermitian_packed :: proc {
	norm_hermitian_packed_c64,
	norm_hermitian_packed_c128,
}

// Compute Hessenberg matrix norm proc group
norm_hessenberg :: proc {
	norm_hessenberg_f32_c64,
	norm_hessenberg_f64_c128,
}

// Compute symmetric matrix norm proc group
norm_symmetric :: proc {
	norm_symmetric_f32_c64,
	norm_symmetric_f64_c128,
}

// Compute triangular banded matrix norm proc group
norm_triangular_banded :: proc {
	norm_triangular_banded_f64_c128,
	norm_triangular_banded_f32_c64,
}

// Compute triangular packed matrix norm proc group
norm_triangular_packed :: proc {
	norm_triangular_packed_f64_c128,
	norm_triangular_packed_f32_c64,
}

// Compute triangular matrix norm proc group (general storage)
norm_triangular_general :: proc {
	norm_triangular_general_f64_c128,
	norm_triangular_general_f32_c64,
}

// Swap rows during factorization
swap_rows :: proc {
	swap_rows_f32,
	swap_rows_f64,
	swap_rows_c64,
	swap_rows_c128,
}

// Euclidean norm proc group
euclidean_norm :: proc {
	euclidean_norm_2d,
	euclidean_norm_3d,
}

// ===================================================================================
// BANDED MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of banded matrix for f32/c64
norm_banded_f32_c64 :: proc(
	A: ^Matrix($T), // Banded matrix
	kl: int, // Number of subdiagonals
	ku: int, // Number of superdiagonals
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n := A.cols
	n_int := Blas_Int(n)
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slangb_(&norm_c, &n_int, &kl_int, &ku_int, raw_data(A.data), &lda, nil, 1)
	} else {
		work: [1]f32
		return lapack.clangb_(&norm_c, &n_int, &kl_int, &ku_int, cast(^complex64)raw_data(A.data), &lda, raw_data(work), 1)
	}
}

// Compute norm of banded matrix for f64/c128
norm_banded_f64_c128 :: proc(
	A: ^Matrix($T), // Banded matrix
	kl: int, // Number of subdiagonals
	ku: int, // Number of superdiagonals
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n := A.cols
	n_int := Blas_Int(n)
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm

	when T == f64 {
		return lapack.dlangb_(&norm_c, &n_int, &kl_int, &ku_int, raw_data(A.data), &lda, nil, 1)
	} else {
		work: [1]f64
		return lapack.zlangb_(&norm_c, &n_int, &kl_int, &ku_int, cast(^complex128)raw_data(A.data), &lda, raw_data(work), 1)
	}
}

// ===================================================================================
// GENERAL MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of general matrix for f32/c64
norm_general_f32_c64 :: proc(
	A: ^Matrix($T), // General matrix
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	m := A.rows
	n := A.cols
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slange_(&norm_c, &m_int, &n_int, raw_data(A.data), &lda, nil, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clange_(&norm_c, &m_int, &n_int, cast(^complex64)raw_data(A.data), &lda, raw_data(work), 1)
		} else {
			return lapack.clange_(&norm_c, &m_int, &n_int, cast(^complex64)raw_data(A.data), &lda, nil, 1)
		}
	}
}

// Compute norm of general matrix for f64/c128
norm_general_f64_c128 :: proc(
	A: ^Matrix($T), // General matrix
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	m := A.rows
	n := A.cols
	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm

	when T == f64 {
		return lapack.dlange_(&norm_c, &m_int, &n_int, raw_data(A.data), &lda, nil, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlange_(&norm_c, &m_int, &n_int, cast(^complex128)raw_data(A.data), &lda, raw_data(work), 1)
		} else {
			return lapack.zlange_(&norm_c, &m_int, &n_int, cast(^complex128)raw_data(A.data), &lda, nil, 1)
		}
	}
}

// ===================================================================================
// TRIDIAGONAL MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of tridiagonal matrix for f32/c64
norm_tridiagonal_f32_c64 :: proc(
	d: []$T, // Main diagonal
	e: []T, // Off-diagonal
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n := len(d)
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slangt_(&norm_c, &n_int, raw_data(e), raw_data(d), raw_data(e[1:]), 1)
	} else {
		return lapack.clangt_(&norm_c, &n_int, cast(^complex64)raw_data(e), cast(^complex64)raw_data(d), cast(^complex64)raw_data(e[1:]), 1)
	}
}

// Compute norm of tridiagonal matrix for f64/c128
norm_tridiagonal_f64_c128 :: proc(
	d: []$T, // Main diagonal
	e: []T, // Off-diagonal
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n := len(d)
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm

	when T == f64 {
		return lapack.dlangt_(&norm_c, &n_int, raw_data(e), raw_data(d), raw_data(e[1:]), 1)
	} else {
		return lapack.zlangt_(&norm_c, &n_int, cast(^complex128)raw_data(e), cast(^complex128)raw_data(d), cast(^complex128)raw_data(e[1:]), 1)
	}
}

// ===================================================================================
// HERMITIAN MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of Hermitian banded matrix for c64
norm_banded_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian banded matrix
	k: int, // Number of super/subdiagonals
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> f32 {
	n := A.cols
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when norm == .FrobeniusNorm {
		work: [1]f32
		return lapack.clanhb_(&norm_c, &uplo_c, &n_int, &k_int, cast(^complex64)raw_data(A.data), &lda, raw_data(work), 1, 1)
	} else {
		return lapack.clanhb_(&norm_c, &uplo_c, &n_int, &k_int, cast(^complex64)raw_data(A.data), &lda, nil, 1, 1)
	}
}

// Compute norm of Hermitian banded matrix for c128
norm_banded_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian banded matrix
	k: int, // Number of super/subdiagonals
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> f64 {
	n := A.cols
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when norm == .FrobeniusNorm {
		work: [1]f64
		return lapack.zlanhb_(&norm_c, &uplo_c, &n_int, &k_int, cast(^complex128)raw_data(A.data), &lda, raw_data(work), 1, 1)
	} else {
		return lapack.zlanhb_(&norm_c, &uplo_c, &n_int, &k_int, cast(^complex128)raw_data(A.data), &lda, nil, 1, 1)
	}
}

// Compute norm of Hermitian matrix for c64
norm_hermitian_c64 :: proc(
	A: ^Matrix(complex64), // Hermitian matrix
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> f32 {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when norm == .FrobeniusNorm {
		work: [1]f32
		return lapack.clanhe_(&norm_c, &uplo_c, &n_int, cast(^complex64)raw_data(A.data), &lda, raw_data(work), 1, 1)
	} else {
		return lapack.clanhe_(&norm_c, &uplo_c, &n_int, cast(^complex64)raw_data(A.data), &lda, nil, 1, 1)
	}
}

// Compute norm of Hermitian matrix for c128
norm_hermitian_c128 :: proc(
	A: ^Matrix(complex128), // Hermitian matrix
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> f64 {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when norm == .FrobeniusNorm {
		work: [1]f64
		return lapack.zlanhe_(&norm_c, &uplo_c, &n_int, cast(^complex128)raw_data(A.data), &lda, raw_data(work), 1, 1)
	} else {
		return lapack.zlanhe_(&norm_c, &uplo_c, &n_int, cast(^complex128)raw_data(A.data), &lda, nil, 1, 1)
	}
}

// Compute norm of Hermitian packed matrix for c64
norm_hermitian_packed_c64 :: proc(
	AP: []complex64, // Hermitian packed matrix
	n: int,
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> f32 {
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when norm == .FrobeniusNorm {
		work: [1]f32
		return lapack.clanhp_(&norm_c, &uplo_c, &n_int, cast(^complex64)raw_data(AP), raw_data(work), 1, 1)
	} else {
		return lapack.clanhp_(&norm_c, &uplo_c, &n_int, cast(^complex64)raw_data(AP), nil, 1, 1)
	}
}

// Compute norm of Hermitian packed matrix for c128
norm_hermitian_packed_c128 :: proc(
	AP: []complex128, // Hermitian packed matrix
	n: int,
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> f64 {
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when norm == .FrobeniusNorm {
		work: [1]f64
		return lapack.zlanhp_(&norm_c, &uplo_c, &n_int, cast(^complex128)raw_data(AP), raw_data(work), 1, 1)
	} else {
		return lapack.zlanhp_(&norm_c, &uplo_c, &n_int, cast(^complex128)raw_data(AP), nil, 1, 1)
	}
}

// ===================================================================================
// HESSENBERG MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of Hessenberg matrix for f32/c64
norm_hessenberg_f32_c64 :: proc(
	A: ^Matrix($T), // Hessenberg matrix
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slanhs_(&norm_c, &n_int, raw_data(A.data), &lda, nil, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clanhs_(&norm_c, &n_int, cast(^complex64)raw_data(A.data), &lda, raw_data(work), 1)
		} else {
			return lapack.clanhs_(&norm_c, &n_int, cast(^complex64)raw_data(A.data), &lda, nil, 1)
		}
	}
}

// Compute norm of Hessenberg matrix for f64/c128
norm_hessenberg_f64_c128 :: proc(
	A: ^Matrix($T), // Hessenberg matrix
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm

	when T == f64 {
		return lapack.dlanhs_(&norm_c, &n_int, raw_data(A.data), &lda, nil, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlanhs_(&norm_c, &n_int, cast(^complex128)raw_data(A.data), &lda, raw_data(work), 1)
		} else {
			return lapack.zlanhs_(&norm_c, &n_int, cast(^complex128)raw_data(A.data), &lda, nil, 1)
		}
	}
}

// ===================================================================================
// SYMMETRIC MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of symmetric matrix for f32/c64
norm_symmetric_f32_c64 :: proc(
	A: ^Matrix($T), // Symmetric matrix
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when T == f32 {
		return lapack.slansy_(&norm_c, &uplo_c, &n_int, raw_data(A.data), &lda, nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clansy_(&norm_c, &uplo_c, &n_int, cast(^complex64)raw_data(A.data), &lda, raw_data(work), 1, 1)
		} else {
			return lapack.clansy_(&norm_c, &uplo_c, &n_int, cast(^complex64)raw_data(A.data), &lda, nil, 1, 1)
		}
	}
}

// Compute norm of symmetric matrix for f64/c128
norm_symmetric_f64_c128 :: proc(
	A: ^Matrix($T), // Symmetric matrix
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when T == f64 {
		return lapack.dlansy_(&norm_c, &uplo_c, &n_int, raw_data(A.data), &lda, nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlansy_(&norm_c, &uplo_c, &n_int, cast(^complex128)raw_data(A.data), &lda, raw_data(work), 1, 1)
		} else {
			return lapack.zlansy_(&norm_c, &uplo_c, &n_int, cast(^complex128)raw_data(A.data), &lda, nil, 1, 1)
		}
	}
}

// ===================================================================================
// TRIANGULAR MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of triangular banded matrix for f32/c64
norm_triangular_banded_f32_c64 :: proc(
	A: ^Matrix($T), // Triangular banded matrix
	k: int, // Number of super/subdiagonals
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n := A.cols
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo
	diag_c := cast(u8)diag

	when T == f32 {
		return lapack.slantb_(&norm_c, &uplo_c, &diag_c, &n_int, &k_int, raw_data(A.data), &lda, nil, 1, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clantb_(&norm_c, &uplo_c, &diag_c, &n_int, &k_int, cast(^complex64)raw_data(A.data), &lda, raw_data(work), 1, 1, 1)
		} else {
			return lapack.clantb_(&norm_c, &uplo_c, &diag_c, &n_int, &k_int, cast(^complex64)raw_data(A.data), &lda, nil, 1, 1, 1)
		}
	}
}

// Compute norm of triangular banded matrix for f64/c128
norm_triangular_banded_f64_c128 :: proc(
	A: ^Matrix($T), // Triangular banded matrix
	k: int, // Number of super/subdiagonals
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n := A.cols
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo
	diag_c := cast(u8)diag

	when T == f64 {
		return lapack.dlantb_(&norm_c, &uplo_c, &diag_c, &n_int, &k_int, raw_data(A.data), &lda, nil, 1, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlantb_(&norm_c, &uplo_c, &diag_c, &n_int, &k_int, cast(^complex128)raw_data(A.data), &lda, raw_data(work), 1, 1, 1)
		} else {
			return lapack.zlantb_(&norm_c, &uplo_c, &diag_c, &n_int, &k_int, cast(^complex128)raw_data(A.data), &lda, nil, 1, 1, 1)
		}
	}
}

// Compute norm of triangular packed matrix for f32/c64
norm_triangular_packed_f32_c64 :: proc(
	AP: []$T, // Triangular packed matrix
	n: int,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo
	diag_c := cast(u8)diag

	when T == f32 {
		return lapack.slantp_(&norm_c, &uplo_c, &diag_c, &n_int, raw_data(AP), nil, 1, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clantp_(&norm_c, &uplo_c, &diag_c, &n_int, cast(^complex64)raw_data(AP), raw_data(work), 1, 1, 1)
		} else {
			return lapack.clantp_(&norm_c, &uplo_c, &diag_c, &n_int, cast(^complex64)raw_data(AP), nil, 1, 1, 1)
		}
	}
}

// Compute norm of triangular packed matrix for f64/c128
norm_triangular_packed_f64_c128 :: proc(
	AP: []$T, // Triangular packed matrix
	n: int,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo
	diag_c := cast(u8)diag

	when T == f64 {
		return lapack.dlantp_(&norm_c, &uplo_c, &diag_c, &n_int, raw_data(AP), nil, 1, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlantp_(&norm_c, &uplo_c, &diag_c, &n_int, cast(^complex128)raw_data(AP), raw_data(work), 1, 1, 1)
		} else {
			return lapack.zlantp_(&norm_c, &uplo_c, &diag_c, &n_int, cast(^complex128)raw_data(AP), nil, 1, 1, 1)
		}
	}
}

// Compute norm of triangular matrix for f32/c64
norm_triangular_general_f32_c64 :: proc(
	A: ^Matrix($T), // Triangular matrix
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo
	diag_c := cast(u8)diag

	when T == f32 {
		return lapack.slantr_(&norm_c, &uplo_c, &diag_c, &n_int, &n_int, raw_data(A.data), &lda, nil, 1, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clantr_(&norm_c, &uplo_c, &diag_c, &n_int, &n_int, cast(^complex64)raw_data(A.data), &lda, raw_data(work), 1, 1, 1)
		} else {
			return lapack.clantr_(&norm_c, &uplo_c, &diag_c, &n_int, &n_int, cast(^complex64)raw_data(A.data), &lda, nil, 1, 1, 1)
		}
	}
}

// Compute norm of triangular matrix for f64/c128
norm_triangular_general_f64_c128 :: proc(
	A: ^Matrix($T), // Triangular matrix
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo
	diag_c := cast(u8)diag

	when T == f64 {
		return lapack.dlantr_(&norm_c, &uplo_c, &diag_c, &n_int, &n_int, raw_data(A.data), &lda, nil, 1, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlantr_(&norm_c, &uplo_c, &diag_c, &n_int, &n_int, cast(^complex128)raw_data(A.data), &lda, raw_data(work), 1, 1, 1)
		} else {
			return lapack.zlantr_(&norm_c, &uplo_c, &diag_c, &n_int, &n_int, cast(^complex128)raw_data(A.data), &lda, nil, 1, 1, 1)
		}
	}
}

// ===================================================================================
// MATRIX ROW SWAPPING (LASWP) - Different from permutation in matrix_utilities.odin
// ===================================================================================
// Note: These are row-swapping functions (laswp) used during factorization
// For general row/column permutation, use the functions in matrix_utilities.odin

// Swap rows of matrix during factorization (f32)
swap_rows_f32 :: proc(A: ^Matrix(f32), perm: []Blas_Int, k1: int = 0, k2: int = -1, incx: int = 1) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	k1_int := Blas_Int(k1)
	k2_int := Blas_Int(k2 == -1 ? len(perm) - 1 : k2)
	incx_int := Blas_Int(incx)

	lapack.slaswp_(&n_int, raw_data(A.data), &lda, &k1_int, &k2_int, raw_data(perm), &incx_int)
}

// Swap rows of matrix during factorization (f64)
swap_rows_f64 :: proc(A: ^Matrix(f64), perm: []Blas_Int, k1: int = 0, k2: int = -1, incx: int = 1) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	k1_int := Blas_Int(k1)
	k2_int := Blas_Int(k2 == -1 ? len(perm) - 1 : k2)
	incx_int := Blas_Int(incx)

	lapack.dlaswp_(&n_int, raw_data(A.data), &lda, &k1_int, &k2_int, raw_data(perm), &incx_int)
}

// Swap rows of matrix during factorization (c64)
swap_rows_c64 :: proc(A: ^Matrix(complex64), perm: []Blas_Int, k1: int = 0, k2: int = -1, incx: int = 1) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	k1_int := Blas_Int(k1)
	k2_int := Blas_Int(k2 == -1 ? len(perm) - 1 : k2)
	incx_int := Blas_Int(incx)

	lapack.claswp_(&n_int, cast(^complex64)raw_data(A.data), &lda, &k1_int, &k2_int, raw_data(perm), &incx_int)
}

// Swap rows of matrix during factorization (c128)
swap_rows_c128 :: proc(A: ^Matrix(complex128), perm: []Blas_Int, k1: int = 0, k2: int = -1, incx: int = 1) {
	n := A.cols
	n_int := Blas_Int(n)
	lda := Blas_Int(A.ld)
	k1_int := Blas_Int(k1)
	k2_int := Blas_Int(k2 == -1 ? len(perm) - 1 : k2)
	incx_int := Blas_Int(incx)

	lapack.zlaswp_(&n_int, cast(^complex128)raw_data(A.data), &lda, &k1_int, &k2_int, raw_data(perm), &incx_int)
}

// ===================================================================================
// EUCLIDEAN NORM COMPUTATION
// ===================================================================================

// Compute 2D Euclidean norm (x^2 + y^2)^(1/2)
euclidean_norm_2d :: proc(x, y: $T) -> T where is_float(T) {
	when T == f32 {
		return lapack.slapy2_(&x, &y)
	} else when T == f64 {
		return lapack.dlapy2_(&x, &y)
	}
	return 0
}

// Compute 3D Euclidean norm (x^2 + y^2 + z^2)^(1/2)
euclidean_norm_3d :: proc(x, y, z: $T) -> T where is_float(T) {
	when T == f32 {
		return lapack.slapy3_(&x, &y, &z)
	} else when T == f64 {
		return lapack.dlapy3_(&x, &y, &z)
	}
	return 0
}

// ===================================================================================
// HERMITIAN TRIDIAGONAL MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of Hermitian tridiagonal matrix proc group
norm_hermitian_tridiagonal :: proc {
	norm_hermitian_tridiagonal_c64,
	norm_hermitian_tridiagonal_c128,
}

// Compute norm of Hermitian tridiagonal matrix for c64
// D is real diagonal, E is complex off-diagonal
norm_hermitian_tridiagonal_c64 :: proc(
	D: []f32, // Real diagonal
	E: []complex64, // Complex off-diagonal
	norm: MatrixNorm = .OneNorm,
) -> f32 {
	n := len(D)
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm

	return lapack.clanht_(&norm_c, &n_int, raw_data(D), cast(^complex64)raw_data(E), 1)
}

// Compute norm of Hermitian tridiagonal matrix for c128
// D is real diagonal, E is complex off-diagonal
norm_hermitian_tridiagonal_c128 :: proc(
	D: []f64, // Real diagonal
	E: []complex128, // Complex off-diagonal
	norm: MatrixNorm = .OneNorm,
) -> f64 {
	n := len(D)
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm

	return lapack.zlanht_(&norm_c, &n_int, raw_data(D), cast(^complex128)raw_data(E), 1)
}

// ===================================================================================
// SYMMETRIC TRIDIAGONAL MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of symmetric tridiagonal matrix proc group
norm_symmetric_tridiagonal :: proc {
	norm_symmetric_tridiagonal_f32,
	norm_symmetric_tridiagonal_f64,
}

// Compute norm of symmetric tridiagonal matrix for f32
norm_symmetric_tridiagonal_f32 :: proc(
	D: []f32, // Main diagonal
	E: []f32, // Off-diagonal
	norm: MatrixNorm = .OneNorm,
) -> f32 {
	n := len(D)
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm

	return lapack.slanst_(&norm_c, &n_int, raw_data(D), raw_data(E), 1)
}

// Compute norm of symmetric tridiagonal matrix for f64
norm_symmetric_tridiagonal_f64 :: proc(
	D: []f64, // Main diagonal
	E: []f64, // Off-diagonal
	norm: MatrixNorm = .OneNorm,
) -> f64 {
	n := len(D)
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm

	return lapack.dlanst_(&norm_c, &n_int, raw_data(D), raw_data(E), 1)
}

// ===================================================================================
// SYMMETRIC BANDED MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of symmetric banded matrix proc group
norm_symmetric_banded :: proc {
	norm_symmetric_banded_f32_c64,
	norm_symmetric_banded_f64_c128,
}

// Compute norm of symmetric banded matrix for f32/c64
norm_symmetric_banded_f32_c64 :: proc(
	AB: ^Matrix($T), // Banded matrix storage
	k: int, // Number of super/subdiagonals
	uplo: MatrixRegion = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n := AB.cols
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldab := Blas_Int(AB.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when T == f32 {
		return lapack.slansb_(&norm_c, &uplo_c, &n_int, &k_int, raw_data(AB.data), &ldab, nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clansb_(&norm_c, &uplo_c, &n_int, &k_int, cast(^complex64)raw_data(AB.data), &ldab, raw_data(work), 1, 1)
		} else {
			return lapack.clansb_(&norm_c, &uplo_c, &n_int, &k_int, cast(^complex64)raw_data(AB.data), &ldab, nil, 1, 1)
		}
	}
}

// Compute norm of symmetric banded matrix for f64/c128
norm_symmetric_banded_f64_c128 :: proc(
	AB: ^Matrix($T), // Banded matrix storage
	k: int, // Number of super/subdiagonals
	uplo: MatrixRegion = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n := AB.cols
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldab := Blas_Int(AB.ld)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when T == f64 {
		return lapack.dlansb_(&norm_c, &uplo_c, &n_int, &k_int, raw_data(AB.data), &ldab, nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlansb_(&norm_c, &uplo_c, &n_int, &k_int, cast(^complex128)raw_data(AB.data), &ldab, raw_data(work), 1, 1)
		} else {
			return lapack.zlansb_(&norm_c, &uplo_c, &n_int, &k_int, cast(^complex128)raw_data(AB.data), &ldab, nil, 1, 1)
		}
	}
}

// ===================================================================================
// SYMMETRIC PACKED MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of symmetric packed matrix proc group
norm_symmetric_packed :: proc {
	norm_symmetric_packed_f32_c64,
	norm_symmetric_packed_f64_c128,
}

// Compute norm of symmetric packed matrix for f32/c64
norm_symmetric_packed_f32_c64 :: proc(
	AP: []$T, // Packed matrix
	n: int,
	uplo: MatrixRegion = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f32 && R == f32) || (T == complex64 && R == f32) {
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when T == f32 {
		return lapack.slansp_(&norm_c, &uplo_c, &n_int, raw_data(AP), nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clansp_(&norm_c, &uplo_c, &n_int, cast(^complex64)raw_data(AP), raw_data(work), 1, 1)
		} else {
			return lapack.clansp_(&norm_c, &uplo_c, &n_int, cast(^complex64)raw_data(AP), nil, 1, 1)
		}
	}
}

// Compute norm of symmetric packed matrix for f64/c128
norm_symmetric_packed_f64_c128 :: proc(
	AP: []$T, // Packed matrix
	n: int,
	uplo: MatrixRegion = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> $R where (T == f64 && R == f64) || (T == complex128 && R == f64) {
	n_int := Blas_Int(n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when T == f64 {
		return lapack.dlansp_(&norm_c, &uplo_c, &n_int, raw_data(AP), nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlansp_(&norm_c, &uplo_c, &n_int, cast(^complex128)raw_data(AP), raw_data(work), 1, 1)
		} else {
			return lapack.zlansp_(&norm_c, &uplo_c, &n_int, cast(^complex128)raw_data(AP), nil, 1, 1)
		}
	}
}
