export read_bvec_kstencil

"""
    $(SIGNATURES)

Build a [`KspaceStencil`](@ref) from a wannier90 `prefix.bvec` file plus
`kpb_k`/`kpb_G` from a `mmn` file.

This is used when wannier90 was run with higher-order finite differences
(`higher_order_n > 1` or `higher_order_nearest_shells`): the b-vector set
contains parallel vectors at multiple norms, so the equal-norm-shell weight
algorithm in [`compute_bweights`](@ref) does not apply. In that case the
weights solved by wannier90 itself (written when `write_bvec = .true.`) are
authoritative.

# Arguments
- `prefix`: filename prefix; reads `\$prefix.bvec`
- `recip_lattice`: each column a reciprocal lattice vector in Å⁻¹
- `kgrid_size`: kpoint grid (e.g., `win.mp_grid`)
- `kpoints`: kpoint fractional coordinates (e.g., `win.kpoints`)
- `kpb_k`, `kpb_G`: from the `mmn` file; supply the per-kpoint mapping

# Keyword Arguments
- `atol`: tolerance for cross-checking the b-vectors read from `.bvec` against
    the analytic reconstruction `recip_lattice * (kpoints[kpb_k] + kpb_G - k)`.
    Defaults to wannier90's `kmesh_tol`.

The b-vectors stored in `KspaceStencil.bvectors` follow the first-kpoint
ordering convention (see [`KspaceStencil`](@ref)); the per-kpoint permutation
is encoded in `kpb_k`/`kpb_G`.
"""
function read_bvec_kstencil(
    prefix::AbstractString,
    recip_lattice::AbstractMatrix,
    kgrid_size::AbstractVector,
    kpoints::AbstractVector,
    kpb_k::AbstractVector,
    kpb_G::AbstractVector;
    atol::Real=default_w90_kmesh_tol(),
)
    bvec = WannierIO.read_bvec("$prefix.bvec")

    n_kpts = length(kpoints)
    n_bvecs = length(kpb_k[1])
    length(bvec.bvecs_cart) == n_kpts ||
        error("bvec n_kpts=$(length(bvec.bvecs_cart)) differs from win/mmn n_kpts=$n_kpts")
    length(bvec.bvecs_cart[1]) == n_bvecs || error(
        "bvec n_bvecs=$(length(bvec.bvecs_cart[1])) differs from mmn n_bvecs=$n_bvecs",
    )
    length(bvec.wb) == n_bvecs ||
        error("bvec wb length=$(length(bvec.wb)) differs from mmn n_bvecs=$n_bvecs")

    # Cross-check that the b-vectors in .bvec at the first kpoint match the
    # analytic reconstruction from .mmn's kpb_k/kpb_G. If this fails, the
    # .bvec and .mmn files are inconsistent (e.g., one was regenerated with
    # a different higher_order_n).
    recip = Mat3(recip_lattice)
    for ib in 1:n_bvecs
        b_reconstructed = recip * (kpoints[kpb_k[1][ib]] + kpb_G[1][ib] - kpoints[1])
        b_from_file = bvec.bvecs_cart[1][ib]
        isapprox(b_from_file, b_reconstructed; atol) || error(
            "bvec file inconsistent with mmn at ib=$ib: " *
            "from file = $b_from_file, reconstructed = $b_reconstructed " *
            "(atol = $atol). The .bvec and .mmn files were likely produced " *
            "by different wannier90 runs.",
        )
    end

    bvectors = bvec.bvecs_cart[1]
    return KspaceStencil(
        recip, Vec3(kgrid_size), kpoints, bvectors, bvec.wb, kpb_k, kpb_G
    )
end
