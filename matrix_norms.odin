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
band_norm :: proc {
	band_norm_f32_c64,
	band_norm_f64_c128,
}

// Compute general matrix norm proc group
dns_norm :: proc {
	dns_norm_f32_c64,
	dns_norm_f64_c128,
}

// Compute tridiagonal matrix norm proc group
trid_norm :: proc {
	trid_norm_f32_c64,
	trid_norm_f64_c128,
}

// Compute Hessenberg matrix norm proc group
dns_hessenberg_norm :: proc {
	dns_hessenberg_norm_f32_c64,
	dns_hessenberg_norm_f64_c128,
}

// Compute symmetric matrix norm proc group
dns_sym_norm :: proc {
	dns_sym_norm_f32_c64,
	dns_sym_norm_f64_c128,
}

// Compute triangular banded matrix norm proc group
band_tri_norm :: proc {
	band_tri_norm_f64_c128,
	band_tri_norm_f32_c64,
}

// Compute triangular packed matrix norm proc group
pack_tri_norm :: proc {
	pack_tri_norm_f64_c128,
	pack_tri_norm_f32_c64,
}

// Compute triangular matrix norm proc group (general storage)
tri_norm :: proc {
	tri_norm_f64_c128,
	tri_norm_f32_c64,
}

// Swap rows during factorization
swap_rows :: proc {
	swap_rows_f32_c64,
	swap_rows_f64_c128,
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
band_norm_f32_c64 :: proc(
	A: ^Matrix($T), // Banded matrix
	kl: int, // Number of subdiagonals
	ku: int, // Number of superdiagonals
	norm: MatrixNorm = .OneNorm,
) -> f32 where T == f32 || T == complex64 {
	n := A.cols
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	lda := A.ld
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slangb_(&norm_c, &n, &kl_int, &ku_int, raw_data(A.data), &lda, nil, 1)
	} else {
		work: [1]f32
		return lapack.clangb_(&norm_c, &n, &kl_int, &ku_int, raw_data(A.data), &lda, raw_data(work), 1)
	}
}

// Compute norm of banded matrix for f64/c128
band_norm_f64_c128 :: proc(
	A: ^Matrix($T), // Banded matrix
	kl: int, // Number of subdiagonals
	ku: int, // Number of superdiagonals
	norm: MatrixNorm = .OneNorm,
) -> f64 where T == f64 || T == complex128 {
	n := A.cols
	kl_int := Blas_Int(kl)
	ku_int := Blas_Int(ku)
	lda := A.ld
	norm_c := cast(u8)norm

	when T == f64 {
		return lapack.dlangb_(&norm_c, &n, &kl_int, &ku_int, raw_data(A.data), &lda, nil, 1)
	} else {
		work: [1]f64
		return lapack.zlangb_(&norm_c, &n, &kl_int, &ku_int, raw_data(A.data), &lda, raw_data(work), 1)
	}
}

// ===================================================================================
// GENERAL MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of general matrix for f32/c64
dns_norm_f32_c64 :: proc(
	A: ^Matrix($T), // General matrix
	norm: MatrixNorm = .OneNorm,
) -> f32 where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slange_(&norm_c, &m, &n, raw_data(A.data), &lda, nil, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clange_(&norm_c, &m, &n, raw_data(A.data), &lda, raw_data(work), 1)
		} else {
			return lapack.clange_(&norm_c, &m, &n, raw_data(A.data), &lda, nil, 1)
		}
	}
}

// Compute norm of general matrix for f64/c128
dns_norm_f64_c128 :: proc(
	A: ^Matrix($T), // General matrix
	norm: MatrixNorm = .OneNorm,
) -> f64 where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := cast(u8)norm

	when T == f64 {
		return lapack.dlange_(&norm_c, &m, &n, raw_data(A.data), &lda, nil, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlange_(&norm_c, &m, &n, raw_data(A.data), &lda, raw_data(work), 1)
		} else {
			return lapack.zlange_(&norm_c, &m, &n, raw_data(A.data), &lda, nil, 1)
		}
	}
}

// ===================================================================================
// TRIDIAGONAL MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of tridiagonal matrix for f32/c64
trid_norm_f32_c64 :: proc(tm: ^Tridiagonal($T), norm: MatrixNorm = .OneNorm) -> f32 where T == f32 || T == complex64 {
	n := tm.n
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slangt_(&norm_c, &n, raw_data(tm.dl), raw_data(tm.d), raw_data(tm.du), 1)
	} else {
		return lapack.clangt_(&norm_c, &n, raw_data(tm.dl), raw_data(tm.d), raw_data(tm.du), 1)
	}
}

// Compute norm of tridiagonal matrix for f64/c128
trid_norm_f64_c128 :: proc(tm: ^Tridiagonal($T), norm: MatrixNorm = .OneNorm) -> f64 where T == f64 || T == complex128 {
	n := tm.n
	norm_c := cast(u8)norm

	when T == f64 {
		return lapack.dlangt_(&norm_c, &n, raw_data(tm.dl), raw_data(tm.d), raw_data(tm.du), 1)
	} else {
		return lapack.zlangt_(&norm_c, &n, raw_data(tm.dl), raw_data(tm.d), raw_data(tm.du), 1)
	}
}

// ===================================================================================
// HERMITIAN MATRIX NORM COMPUTATION
// ===================================================================================

// Query workspace for Hermitian banded matrix norm (only needed for infinity norm)
query_workspace_band_herm_norm :: proc(hb: ^HermBand($Cmplx), norm: MatrixNorm) -> int where is_complex(Cmplx) {
	// Only infinity norm requires workspace
	if norm == .InfinityNorm {
		return int(hb.n)
	}
	return 0
}

// Compute norm of Hermitian banded matrix (complex types)
band_herm_norm :: proc(
	hb: ^HermBand($Cmplx),
	work: []$Real, // Workspace (only needed for infinity norm, size n)
	norm: MatrixNorm = .OneNorm,
) -> Real where is_complex(Cmplx) {
	n := hb.n
	kd := hb.kd
	ldab := hb.ldab
	norm_c := cast(u8)norm
	uplo_c := cast(u8)hb.uplo

	// Workspace only needed for infinity norm
	if norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small for infinity norm")
	}

	when Cmplx == complex64 {
		if norm == .InfinityNorm {
			return lapack.clanhb_(&norm_c, &uplo_c, &n, &kd, raw_data(hb.data), &ldab, raw_data(work), 1, 1)
		} else {
			return lapack.clanhb_(&norm_c, &uplo_c, &n, &kd, raw_data(hb.data), &ldab, nil, 1, 1)
		}
	} else when Cmplx == complex128 {
		if norm == .InfinityNorm {
			return lapack.zlanhb_(&norm_c, &uplo_c, &n, &kd, raw_data(hb.data), &ldab, raw_data(work), 1, 1)
		} else {
			return lapack.zlanhb_(&norm_c, &uplo_c, &n, &kd, raw_data(hb.data), &ldab, nil, 1, 1)
		}
	}
	return 0
}

// Query workspace for Hermitian matrix norm (only needed for infinity norm with complex matrices)
query_workspace_dns_herm_norm :: proc(A: ^Matrix($Cmplx), norm: MatrixNorm) -> int where is_complex(Cmplx) {
	// Only infinity norm requires workspace for complex Hermitian matrices
	if norm == .InfinityNorm {
		return int(A.cols)
	}
	return 0
}

// Compute norm of Hermitian matrix (complex types)
dns_herm_norm :: proc(
	A: ^Matrix($Cmplx), // Hermitian matrix
	work: []$Real, // Workspace (only needed for infinity norm, size n)
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> Real where is_complex(Cmplx) {
	n := A.cols
	lda := A.ld
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	// Workspace only needed for infinity norm
	if norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small for infinity norm")
	}

	when Cmplx == complex64 {
		if norm == .InfinityNorm {
			return lapack.clanhe_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(work), 1, 1)
		} else {
			return lapack.clanhe_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, nil, 1, 1)
		}
	} else when Cmplx == complex128 {
		if norm == .InfinityNorm {
			return lapack.zlanhe_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(work), 1, 1)
		} else {
			return lapack.zlanhe_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, nil, 1, 1)
		}
	}
	return 0
}

// Query workspace for Hermitian packed matrix norm (only needed for infinity norm)
query_workspace_pack_herm_norm :: proc(n: int, norm: MatrixNorm) -> int {
	// Only infinity norm requires workspace
	if norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of Hermitian packed matrix (complex types)
pack_herm_norm :: proc(
	AP: []$Cmplx, // Hermitian packed matrix
	work: []$Real, // Workspace (only needed for infinity norm, size n)
	n: int,
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> Real where is_complex(Cmplx) {
	n := Blas_Int(n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	// Workspace only needed for infinity norm
	if norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small for infinity norm")
	}

	when Cmplx == complex64 {
		if norm == .InfinityNorm {
			return lapack.clanhp_(&norm_c, &uplo_c, &n, raw_data(AP), raw_data(work), 1, 1)
		} else {
			return lapack.clanhp_(&norm_c, &uplo_c, &n, raw_data(AP), nil, 1, 1)
		}
	} else when Cmplx == complex128 {
		if norm == .InfinityNorm {
			return lapack.zlanhp_(&norm_c, &uplo_c, &n, raw_data(AP), raw_data(work), 1, 1)
		} else {
			return lapack.zlanhp_(&norm_c, &uplo_c, &n, raw_data(AP), nil, 1, 1)
		}
	}
	return 0
}

// ===================================================================================
// HESSENBERG MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of Hessenberg matrix for f32/c64
dns_hessenberg_norm_f32_c64 :: proc(
	A: ^Matrix($T), // Hessenberg matrix
	norm: MatrixNorm = .OneNorm,
) -> f32 where T == f32 || T == complex64 {
	n := A.cols
	lda := A.ld
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slanhs_(&norm_c, &n, raw_data(A.data), &lda, nil, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clanhs_(&norm_c, &n, raw_data(A.data), &lda, raw_data(work), 1)
		} else {
			return lapack.clanhs_(&norm_c, &n, raw_data(A.data), &lda, nil, 1)
		}
	}
}

// Compute norm of Hessenberg matrix for f64/c128
dns_hessenberg_norm_f64_c128 :: proc(
	A: ^Matrix($T), // Hessenberg matrix
	norm: MatrixNorm = .OneNorm,
) -> f64 where T == f64 || T == complex128 {
	n := A.cols
	lda := A.ld
	norm_c := cast(u8)norm

	when T == f64 {
		return lapack.dlanhs_(&norm_c, &n, raw_data(A.data), &lda, nil, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlanhs_(&norm_c, &n, raw_data(A.data), &lda, raw_data(work), 1)
		} else {
			return lapack.zlanhs_(&norm_c, &n, raw_data(A.data), &lda, nil, 1)
		}
	}
}

// ===================================================================================
// SYMMETRIC MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of symmetric matrix for f32/c64
dns_sym_norm_f32_c64 :: proc(
	A: ^Matrix($T), // Symmetric matrix
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> f32 where T == f32 || T == complex64 {
	n := A.cols
	lda := A.ld
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when T == f32 {
		return lapack.slansy_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clansy_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(work), 1, 1)
		} else {
			return lapack.clansy_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, nil, 1, 1)
		}
	}
}

// Compute norm of symmetric matrix for f64/c128
dns_sym_norm_f64_c128 :: proc(
	A: ^Matrix($T), // Symmetric matrix
	uplo: MatrixTriangle = .Upper,
	norm: MatrixNorm = .OneNorm,
) -> f64 where T == f64 || T == complex128 {
	n := A.cols
	lda := A.ld
	norm_c := cast(u8)norm
	uplo_c := cast(u8)uplo

	when T == f64 {
		return lapack.dlansy_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlansy_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, raw_data(work), 1, 1)
		} else {
			return lapack.zlansy_(&norm_c, &uplo_c, &n, raw_data(A.data), &lda, nil, 1, 1)
		}
	}
}

// ===================================================================================
// TRIANGULAR MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of triangular banded matrix for f32/c64
band_tri_norm_f32_c64 :: proc(
	tb: ^TriBand($T), // Triangular banded matrix
	norm: MatrixNorm = .OneNorm,
) -> f32 where T == f32 || T == complex64 {
	n := tb.n
	k := tb.k
	ldab := tb.ldab
	norm_c := cast(u8)norm
	uplo_c := cast(u8)tb.uplo
	diag_c := cast(u8)tb.diag

	when T == f32 {
		return lapack.slantb_(&norm_c, &uplo_c, &diag_c, &n, &k, raw_data(tb.data), &ldab, nil)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clantb_(&norm_c, &uplo_c, &diag_c, &n, &k, raw_data(tb.data), &ldab, raw_data(work))
		} else {
			return lapack.clantb_(&norm_c, &uplo_c, &diag_c, &n, &k, raw_data(tb.data), &ldab, nil)
		}
	}
}

// Compute norm of triangular banded matrix for f64/c128
band_tri_norm_f64_c128 :: proc(
	tb: ^TriBand($T), // Triangular banded matrix
	norm: MatrixNorm = .OneNorm,
) -> f64 where T == f64 || T == complex128 {
	n := tb.n
	k := tb.k
	ldab := tb.ldab
	norm_c := cast(u8)norm
	uplo_c := cast(u8)tb.uplo
	diag_c := cast(u8)tb.diag

	when T == f64 {
		return lapack.dlantb_(&norm_c, &uplo_c, &diag_c, &n, &k, raw_data(tb.data), &ldab, nil)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlantb_(&norm_c, &uplo_c, &diag_c, &n, &k, raw_data(tb.data), &ldab, raw_data(work))
		} else {
			return lapack.zlantb_(&norm_c, &uplo_c, &diag_c, &n, &k, raw_data(tb.data), &ldab, nil)
		}
	}
}

// Compute norm of triangular packed matrix for f32/c64
pack_tri_norm_f32_c64 :: proc(
	pt: ^PackedTriangular($T), // Triangular packed matrix
	norm: MatrixNorm = .OneNorm,
) -> f32 where T == f32 || T == complex64 {
	n := Blas_Int(pt.n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)pt.uplo
	diag_c := cast(u8)pt.diag

	when T == f32 {
		return lapack.slantp_(&norm_c, &uplo_c, &diag_c, &n, raw_data(pt.data), nil)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clantp_(&norm_c, &uplo_c, &diag_c, &n, raw_data(pt.data), raw_data(work))
		} else {
			return lapack.clantp_(&norm_c, &uplo_c, &diag_c, &n, raw_data(pt.data), nil)
		}
	}
}

// Compute norm of triangular packed matrix for f64/c128
pack_tri_norm_f64_c128 :: proc(
	pt: ^PackedTriangular($T), // Triangular packed matrix
	norm: MatrixNorm = .OneNorm,
) -> f64 where T == f64 || T == complex128 {
	n := Blas_Int(pt.n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)pt.uplo
	diag_c := cast(u8)pt.diag

	when T == f64 {
		return lapack.dlantp_(&norm_c, &uplo_c, &diag_c, &n, raw_data(pt.data), nil)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlantp_(&norm_c, &uplo_c, &diag_c, &n, raw_data(pt.data), raw_data(work))
		} else {
			return lapack.dlantp_(&norm_c, &uplo_c, &diag_c, &n, raw_data(pt.data), nil)
		}
	}
}

// Compute norm of triangular matrix for f32/c64
tri_norm_f32_c64 :: proc(
	tri: ^Triangular($T), // Triangular matrix
	norm: MatrixNorm = .OneNorm,
) -> f32 where T == f32 || T == complex64 {
	n := Blas_Int(tri.n)
	lda := Blas_Int(tri.lda)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)tri.uplo
	diag_c := cast(u8)tri.diag

	when T == f32 {
		return lapack.slantr_(&norm_c, &uplo_c, &diag_c, &n, &n, raw_data(tri.data), &lda, nil)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clantr_(&norm_c, &uplo_c, &diag_c, &n, &n, raw_data(tri.data), &lda, raw_data(work))
		} else {
			return lapack.clantr_(&norm_c, &uplo_c, &diag_c, &n, &n, raw_data(tri.data), &lda, nil)
		}
	}
}

// Compute norm of triangular matrix for f64/c128
tri_norm_f64_c128 :: proc(
	tri: ^Triangular($T), // Triangular matrix
	norm: MatrixNorm = .OneNorm,
) -> f64 where T == f64 || T == complex128 {
	n := Blas_Int(tri.n)
	lda := Blas_Int(tri.lda)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)tri.uplo
	diag_c := cast(u8)tri.diag

	when T == f64 {
		return lapack.dlantr_(&norm_c, &uplo_c, &diag_c, &n, &n, raw_data(tri.data), &lda, nil)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlantr_(&norm_c, &uplo_c, &diag_c, &n, &n, raw_data(tri.data), &lda, raw_data(work))
		} else {
			return lapack.zlantr_(&norm_c, &uplo_c, &diag_c, &n, &n, raw_data(tri.data), &lda, nil)
		}
	}
}

// ===================================================================================
// MATRIX ROW SWAPPING (LASWP) - Different from permutation in matrix_utilities.odin
// ===================================================================================
// Note: These are row-swapping functions (laswp) used during factorization
// For general row/column permutation, use the functions in matrix_utilities.odin

// Swap rows of matrix during factorization for f32/c64
swap_rows_f32_c64 :: proc(A: ^Matrix($T), perm: []Blas_Int, k1: int = 0, k2: int = -1, incx: int = 1) where T == f32 || T == complex64 {
	n := A.cols
	lda := A.ld
	k1_int := Blas_Int(k1)
	k2_int := Blas_Int(k2 == -1 ? len(perm) - 1 : k2)
	incx_int := Blas_Int(incx)

	when T == f32 {
		lapack.slaswp_(&n, raw_data(A.data), &lda, &k1_int, &k2_int, raw_data(perm), &incx_int)
	} else {
		lapack.claswp_(&n, raw_data(A.data), &lda, &k1_int, &k2_int, raw_data(perm), &incx_int)
	}
}

// Swap rows of matrix during factorization for f64/c128
swap_rows_f64_c128 :: proc(A: ^Matrix($T), perm: []Blas_Int, k1: int = 0, k2: int = -1, incx: int = 1) where T == f64 || T == complex128 {
	n := A.cols
	lda := A.ld
	k1_int := Blas_Int(k1)
	k2_int := Blas_Int(k2 == -1 ? len(perm) - 1 : k2)
	incx_int := Blas_Int(incx)

	when T == f64 {
		lapack.dlaswp_(&n, raw_data(A.data), &lda, &k1_int, &k2_int, raw_data(perm), &incx_int)
	} else {
		lapack.zlaswp_(&n, raw_data(A.data), &lda, &k1_int, &k2_int, raw_data(perm), &incx_int)
	}
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
trid_herm_norm :: proc {
	trid_herm_norm_c64,
	trid_herm_norm_c128,
}

// Compute norm of Hermitian tridiagonal matrix for c64
// D is real diagonal, E is complex off-diagonal
trid_herm_norm_c64 :: proc(
	D: []f32, // Real diagonal
	E: []complex64, // Complex off-diagonal
	norm: MatrixNorm = .OneNorm,
) -> f32 {
	n := Blas_Int(len(D))
	norm_c := cast(u8)norm

	return lapack.clanht_(&norm_c, &n, raw_data(D), raw_data(E), 1)
}

// Compute norm of Hermitian tridiagonal matrix for c128
// D is real diagonal, E is complex off-diagonal
trid_herm_norm_c128 :: proc(
	D: []f64, // Real diagonal
	E: []complex128, // Complex off-diagonal
	norm: MatrixNorm = .OneNorm,
) -> f64 {
	n := Blas_Int(len(D))
	norm_c := cast(u8)norm

	return lapack.zlanht_(&norm_c, &n, raw_data(D), raw_data(E), 1)
}

// ===================================================================================
// SYMMETRIC TRIDIAGONAL MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of symmetric tridiagonal matrix (real types only)
// Note: For symmetric tridiagonal, dl == du, so we use d and du
trid_sym_norm :: proc(tm: ^Tridiagonal($T), norm: MatrixNorm = .OneNorm) -> T where T == f32 || T == f64 {
	assert(tm.symmetric, "trid_sym_norm requires symmetric tridiagonal matrix")

	n := tm.n
	norm_c := cast(u8)norm

	when T == f32 {
		return lapack.slanst_(&norm_c, &n, raw_data(tm.d), raw_data(tm.du), 1)
	} else when T == f64 {
		return lapack.dlanst_(&norm_c, &n, raw_data(tm.d), raw_data(tm.du), 1)
	}
	return 0
}

// ===================================================================================
// SYMMETRIC BANDED MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of symmetric banded matrix proc group
band_sym_norm :: proc {
	band_sym_norm_f32_c64,
	band_sym_norm_f64_c128,
}

// Compute norm of symmetric banded matrix for f32/c64
band_sym_norm_f32_c64 :: proc(sb: ^SymBand($T), norm: MatrixNorm = .OneNorm) -> f32 where T == f32 || T == complex64 {
	n := sb.n
	kd := sb.kd
	ldab := sb.ldab
	norm_c := cast(u8)norm
	uplo_c := cast(u8)sb.uplo

	when T == f32 {
		return lapack.slansb_(&norm_c, &uplo_c, &n, &kd, raw_data(sb.data), &ldab, nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clansb_(&norm_c, &uplo_c, &n, &kd, raw_data(sb.data), &ldab, raw_data(work), 1, 1)
		} else {
			return lapack.clansb_(&norm_c, &uplo_c, &n, &kd, raw_data(sb.data), &ldab, nil, 1, 1)
		}
	}
}

// Compute norm of symmetric banded matrix for f64/c128
band_sym_norm_f64_c128 :: proc(sb: ^SymBand($T), norm: MatrixNorm = .OneNorm) -> f64 where T == f64 || T == complex128 {
	n := sb.n
	kd := sb.kd
	ldab := sb.ldab
	norm_c := cast(u8)norm
	uplo_c := cast(u8)sb.uplo

	when T == f64 {
		return lapack.dlansb_(&norm_c, &uplo_c, &n, &kd, raw_data(sb.data), &ldab, nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlansb_(&norm_c, &uplo_c, &n, &kd, raw_data(sb.data), &ldab, raw_data(work), 1, 1)
		} else {
			return lapack.zlansb_(&norm_c, &uplo_c, &n, &kd, raw_data(sb.data), &ldab, nil, 1, 1)
		}
	}
}

// ===================================================================================
// SYMMETRIC PACKED MATRIX NORM COMPUTATION
// ===================================================================================

// Compute norm of symmetric packed matrix proc group
pack_sym_norm :: proc {
	pack_sym_norm_f32_c64,
	pack_sym_norm_f64_c128,
}

// Compute norm of symmetric packed matrix for f32/c64
pack_sym_norm_f32_c64 :: proc(ps: ^PackedSymmetric($T), norm: MatrixNorm = .OneNorm) -> f32 where T == f32 || T == complex64 {
	n := Blas_Int(ps.n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)ps.uplo

	when T == f32 {
		return lapack.slansp_(&norm_c, &uplo_c, &n, raw_data(ps.data), nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f32
			return lapack.clansp_(&norm_c, &uplo_c, &n, raw_data(ps.data), raw_data(work), 1, 1)
		} else {
			return lapack.clansp_(&norm_c, &uplo_c, &n, raw_data(ps.data), nil, 1, 1)
		}
	}
}

// Compute norm of symmetric packed matrix for f64/c128
pack_sym_norm_f64_c128 :: proc(ps: ^PackedSymmetric($T), norm: MatrixNorm = .OneNorm) -> f64 where T == f64 || T == complex128 {
	n := Blas_Int(ps.n)
	norm_c := cast(u8)norm
	uplo_c := cast(u8)ps.uplo

	when T == f64 {
		return lapack.dlansp_(&norm_c, &uplo_c, &n, raw_data(ps.data), nil, 1, 1)
	} else {
		when norm == .FrobeniusNorm {
			work: [1]f64
			return lapack.zlansp_(&norm_c, &uplo_c, &n, raw_data(ps.data), raw_data(work), 1, 1)
		} else {
			return lapack.zlansp_(&norm_c, &uplo_c, &n, raw_data(ps.data), nil, 1, 1)
		}
	}
}
