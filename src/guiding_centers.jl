export compute_guiding_centers

"""
    compute_guiding_centers(kstencil, M, U) -> Vector{Vec3{T}}

Compute Wannier90-style guiding centers `rguide[n]` (Cartesian Å) for each
Wannier function. Used by `omega(model; guiding_centers=true)` to pick the
correct branch of `Im[ln M_nn^{k,b}]` in the spread / center formula on coarse
k-grids, where the principal branch can wrap and displace centers by lattice
vectors.

Port of Wannier90's `wann_phases` (`wannierise.F90:1776`). For each WF `n`
independently:

```
csum[d]   = Σ_k (U† M U)_{nn}^{k, ib(k,d)}                          # phase accumulator
xx[d]     = -Im[ln csum[d]]                                         # d ≤ 3, arbitrary branch
          = xx0 − Im[ln(csum[d] · cis(xx0))],  xx0 = b[d] · rguide  # d > 3, branch closest to xx0
rguide[n] = least-squares solution of  b[d] · rguide ≈ xx[d]        # 3×3 system, solved incrementally
```

Here `d` indexes canonical b-directions (defined from the k=1 stencil), and
`ib(k,d)` is the per-k slot at which direction `d` appears — Wannier.jl orders
neighbors per k independently, so `M[ik][ib]` for fixed `ib` is generally a
*different* direction at each `k`. We remap via b-vector matching, the same
job Wannier90's `kmesh_info%neigh(nkp, na)` serves.

The least-squares update fires once `det(smat) > 1e-6` (matches Wannier90's
`eps6` guard). The `±b` pairs in the stencil produce mutually conjugate `csum`
and antisymmetric `xx`, so including all directions just adds duplicate
equations — the LS solution is the same as Wannier90's half-stencil `nnh`
form.
"""
function compute_guiding_centers(
    kstencil::KspaceStencil{T}, M::AbstractVector, U
) where {T<:Real}
    n_kpts = length(M)
    n_bvecs = length(M[1])
    n_wann = U isa AbstractVector ? size(U[1], 2) : size(U, 2)

    cache = Cache(kstencil, M, U)
    compute_MU_UtMU!(cache, kstencil, M, U)
    UtMU = cache.UtMU

    kpb_k = kstencil.kpb_k
    kpb_G = kstencil.kpb_G
    recip_lattice = reciprocal_lattice(kstencil)
    kpoints = kstencil.kpoints

    # Canonical b-direction list, taken from the k=1 stencil.
    bs = Vector{Vec3{T}}(undef, n_bvecs)
    for ib in 1:n_bvecs
        bs[ib] = recip_lattice * (kpoints[kpb_k[1][ib]] + kpb_G[1][ib] - kpoints[1])
    end

    # dir_at[ik][ib] = canonical direction index for the (ik, ib) slot.
    # Wannier.jl reorders neighbors per k; we recover Wannier90's `neigh(nkp, na)`.
    tol = T(1e-6)
    dir_at = [zeros(Int, n_bvecs) for _ in 1:n_kpts]
    for ik in 1:n_kpts
        for ib in 1:n_bvecs
            b = recip_lattice * (kpoints[kpb_k[ik][ib]] + kpb_G[ik][ib] - kpoints[ik])
            id = findfirst(bd -> norm(b - bd) < tol, bs)
            id === nothing && error(
                "b-vector at (ik=$ik, ib=$ib) does not match any canonical direction; " *
                "stencil is non-uniform across k",
            )
            dir_at[ik][ib] = id
        end
    end

    rguide = zeros(Vec3{T}, n_wann)
    csum = zeros(Complex{T}, n_bvecs)

    for n in 1:n_wann
        fill!(csum, zero(Complex{T}))
        for ik in 1:n_kpts
            UtMUk = UtMU[ik]
            dir_k = dir_at[ik]
            for ib in 1:n_bvecs
                csum[dir_k[ib]] += UtMUk[ib][n, n]
            end
        end

        smat = zero(SMatrix{3,3,T,9})
        svec = zero(SVector{3,T})
        rg = zero(SVector{3,T})

        for d in 1:n_bvecs
            b = bs[d]
            xx = if d ≤ 3
                -imaglog(csum[d])
            else
                xx0 = b ⋅ rg
                xx0 - imaglog(csum[d] * cis(xx0))
            end
            smat = smat + b * b'
            svec = svec + b * xx
            if d ≥ 3 && abs(det(smat)) > T(1e-6)
                rg = smat \ svec
            end
        end

        rguide[n] = rg
    end

    return rguide
end
