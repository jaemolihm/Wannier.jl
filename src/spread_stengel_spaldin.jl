export StengelSpaldinSpread

@doc raw"""
    struct StengelSpaldinSpread

The Stengel-Spaldin (SS) spread functional, i.e. wannier90's `use_ss_functional`.

The SS functional takes the average over ``\bm{k}`` *before* the logarithm,
where the Marzari-Vanderbilt (MV) functional of [`Spread`](@ref) takes it after:

```math
S_{n\bm{b}} = \frac{1}{N_k} \sum_{\bm{k}} M_{nn}^{\bm{k},\bm{b}}, \qquad
\langle \bm{r} \rangle_n = -\sum_{\bm{b}} w_{\bm{b}} \bm{b} \, \Im \log S_{n\bm{b}}
```

Only the diagonal part of the spread changes; ``\Omega_{\mathrm{I}}`` and
``\Omega_{\mathrm{OD}}`` are identical to MV. ``\Omega_{\mathrm{D}}`` becomes the
variance over ``\bm{k}`` of the diagonal overlaps, and is therefore manifestly
non-negative:

```math
\Omega_{\mathrm{D}} = \sum_{n\bm{b}} w_{\bm{b}} \left(
    \frac{1}{N_k} \sum_{\bm{k}} | M_{nn}^{\bm{k},\bm{b}} |^2
    - | S_{n\bm{b}} |^2 \right)
```

The total collapses to ``\Omega = \sum_{n\bm{b}} w_{\bm{b}}
(1 - | S_{n\bm{b}} |^2)``, which [`omega!`](@ref) reports per WF in `ω`.

Both functionals converge to the same result on an infinitely fine
``\bm{k}``-grid, and both are translationally invariant, but only SS is size
consistent. See Stengel and Spaldin, Phys. Rev. B **73**, 075121 (2006).

Use it in place of [`SpreadPenalty`](@ref):

```julia
p = StengelSpaldinSpread(model.kstencil)
model.gauges .= max_localize(p, model)
```

!!! note

    Because ``S_{n\bm{b}}`` sums over ``\bm{k}`` at fixed ``\bm{b}``, slot `ib`
    must denote the same ``\bm{b}``-vector at every kpoint. Rather than
    reordering the overlaps, this struct stores the permutation `nnord` and
    indexes through it, which is what wannier90's `kmesh_info%nnord` does.

# Fields
$(FIELDS)
"""
struct StengelSpaldinSpread{T <: Real} <: AbstractPenalty
    """``\\bm{b}``-vectors in the reference (Γ-point) ordering, Cartesian
    coordinates in Å⁻¹ unit"""
    bvectors::Vector{Vec3{T}}

    """weight of each of `bvectors`, Å² unit"""
    bweights::Vector{T}

    """`nnord[ik][ib]` is the slot at kpoint `ik` holding `bvectors[ib]`, i.e.
    wannier90's `kmesh_info%nnord`"""
    nnord::Vector{Vector{Int}}

    """`ibrev[ib]` is the reference index of `-bvectors[ib]`, i.e. wannier90's
    `kmesh_info%nnrev` composed with `nnord`"""
    ibrev::Vector{Int}
end

"""
    $(SIGNATURES)

Build the b-vector orderings the SS functional needs from a `KspaceStencil`.

Errors if the stencil is not closed under ``\\bm{b} \\to -\\bm{b}``, which the
gradient requires.
"""
function StengelSpaldinSpread(kstencil::KspaceStencil{T}) where {T}
    bvectors = kstencil.bvectors
    nbvecs = length(bvectors)
    inv_recip_lattice = inv(reciprocal_lattice(kstencil))
    bvectors_frac = map(b -> inv_recip_lattice * b, bvectors)

    nnord = map(1:n_kpoints(kstencil)) do ik
        map(enumerate(bvectors_frac)) do (ib, b)
            slot = index_bvector(kstencil, ik, b)
            isnothing(slot) && error("bvector $ib not found at kpoint $ik")
            return slot
        end
    end

    ibrev = map(enumerate(bvectors_frac)) do (ib, b)
        slot = findfirst(isapprox(-b), bvectors_frac)
        isnothing(slot) && error(
            "no -b partner for bvector $ib; the Stengel-Spaldin gradient needs " *
            "a stencil closed under b -> -b"
        )
        return slot
    end

    return StengelSpaldinSpread{T}(bvectors, kstencil.bweights, nnord, ibrev)
end

StengelSpaldinSpread(model::Model) = StengelSpaldinSpread(model.kstencil)

n_bvectors(p::StengelSpaldinSpread) = length(p.bvectors)

"""
    $(SIGNATURES)

Compute ``S_{n\\bm{b}} = \\langle (U^\\dagger M U)_{nn} \\rangle_{\\bm{k}}``,
the k-average taken before the logarithm.

Returns a `n_wann * n_bvectors` matrix.
"""
function compute_ss_overlaps(p::StengelSpaldinSpread, UtMU::AbstractVector, nwann::Integer)
    nkpts = length(UtMU)
    nbvecs = n_bvectors(p)
    S = zeros(eltype(UtMU[1][1]), nwann, nbvecs)

    for ik in 1:nkpts
        Nk = UtMU[ik]
        nnord = p.nnord[ik]
        for ib in 1:nbvecs
            Nkb = Nk[nnord[ib]]
            for n in 1:nwann
                S[n, ib] += Nkb[n, n]
            end
        end
    end

    S ./= nkpts
    return S
end

function omega!(
        p::StengelSpaldinSpread, cache::Cache{FT}, bvectors::KspaceStencil{FT}, M
    ) where {FT <: Real}
    UtMU = cache.UtMU
    nwann = n_wann(cache)
    nkpts = n_kpts(cache)
    nbvecs = n_bvectors(p)

    S = compute_ss_overlaps(p, UtMU, nwann)
    # ⟨|M_nn|²⟩_k, the other half of the k-variance
    absM² = zeros(FT, nwann, nbvecs)

    ΩI::FT = 0.0
    ΩOD::FT = 0.0

    for ik in 1:nkpts
        Nk = UtMU[ik]
        nnord = p.nnord[ik]
        for ib in 1:nbvecs
            Nkb = Nk[nnord[ib]]
            wᵇ = p.bweights[ib]

            # ΩI and ΩOD are unchanged from MV
            ts = zero(FT)
            ts2 = zero(FT)
            for i in axes(Nkb, 2)
                for j in axes(Nkb, 1)
                    a2 = abs2(Nkb[j, i])
                    ts += a2
                    if i == j
                        absM²[i, ib] += a2
                    else
                        ts2 += a2
                    end
                end
            end

            ΩI += wᵇ * (nwann - ts)
            ΩOD += wᵇ * ts2
        end
    end

    absM² ./= nkpts
    ΩI /= nkpts
    ΩOD /= nkpts

    # ΩD is the k-variance of the diagonal overlaps, hence ≥ 0
    ΩD::FT = 0.0
    for ib in 1:nbvecs
        wᵇ = p.bweights[ib]
        for n in 1:nwann
            ΩD += wᵇ * (absM²[n, ib] - abs2(S[n, ib]))
        end
    end

    r = cache.r
    fill!(r, zero(eltype(r)))
    ω = zeros(FT, nwann)
    for ib in 1:nbvecs
        wb_b = p.bweights[ib] * p.bvectors[ib]
        wᵇ = p.bweights[ib]
        for n in 1:nwann
            r[n] -= imaglog(S[n, ib]) * wb_b
            ω[n] += wᵇ * (1 - abs2(S[n, ib]))
        end
    end

    Ω̃ = ΩOD + ΩD
    Ω = ΩI + Ω̃
    return Spread(Ω, ΩI, ΩOD, ΩD, Ω̃, ω, copy(r))
end

function omega(p::StengelSpaldinSpread, bvectors::KspaceStencil, M, U)
    cache = Cache(bvectors, M, U)
    compute_MU_UtMU!(cache, bvectors, M, U)
    return omega!(p, cache, bvectors, M)
end

function omega(p::StengelSpaldinSpread, bvectors::KspaceStencil, M, X, Y)
    return omega(p, bvectors, M, X_Y_to_U(X, Y))
end

@doc raw"""
    $(SIGNATURES)

Gradient of the Stengel-Spaldin spread w.r.t. the gauge.

With ``G = 2 \partial \Omega / \partial \bar{U}``, which is the convention of
[`omega_grad!`](@ref) for the MV functional, and
``\Omega = \sum_{n\bm{b}} w_{\bm{b}} (1 - | S_{n\bm{b}} |^2)``:

```math
\frac{\partial \Omega}{\partial U_{\bm{k}}}[m,n] = -\frac{4}{N_k}
    \sum_{\bm{b}} w_{\bm{b}} \, S^{*}_{n\bm{b}}
    \left( M^{\bm{k},\bm{b}} U_{\bm{k}+\bm{b}} \right)[m,n]
```

Differentiating ``|S_{n\bm{b}}|^2`` gives one term from each of ``U_{\bm{k}}``
and ``\bar{U}_{\bm{k}}``, at ``+\bm{b}`` and ``-\bm{b}`` respectively. They
combine because ``M^{\bm{k},\bm{b}\dagger} = M^{\bm{k}+\bm{b},-\bm{b}}`` turns
the ``-\bm{b}`` term into another `MU` at ``\bm{k}``, and because
``S_{n,-\bm{b}} = S^{*}_{n\bm{b}}`` — which is why the stencil must be closed
under ``\bm{b} \to -\bm{b}``.

!!! note

    This is *not* wannier90's `wann_domega` expression. That routine computes
    ``\partial \Omega / \partial W`` for a unitary generator
    (``U \to U e^{W}``), which carries an extra antihermitian structure; here
    the unconstrained ``\partial \Omega / \partial U`` is wanted, matching what
    [`max_localize`](@ref) hands to `Optim`.
"""
function omega_grad!(
        p::StengelSpaldinSpread, cache::Cache{FT}, bvectors::KspaceStencil{FT}, M
    ) where {FT <: Real}
    G = cache.G
    fill!(G, 0)
    UtMU = cache.UtMU
    MU = cache.MU

    nbands, nwann, nkpts = size(G)
    nbvecs = n_bvectors(p)

    S = compute_ss_overlaps(p, UtMU, nwann)

    @inbounds for ik in 1:nkpts
        MUk = MU[ik]
        nnord = p.nnord[ik]
        Gk = view(G, :, :, ik)
        for ib in 1:nbvecs
            wᵇ = p.bweights[ib]
            MUkb = MUk[nnord[ib]]
            for n in 1:nwann
                c = -4 * wᵇ * conj(S[n, ib])
                for m in 1:nbands
                    Gk[m, n] += c * MUkb[m, n]
                end
            end
        end
    end

    G ./= nkpts
    return G
end

function omega_grad(p::StengelSpaldinSpread, bvectors::KspaceStencil, M, U)
    cache = Cache(bvectors, M, U)
    compute_MU_UtMU!(cache, bvectors, M, U)
    return omega_grad!(p, cache, bvectors, M)
end

function omega_grad(p::StengelSpaldinSpread, bvectors::KspaceStencil, M, X, Y, frozen)
    U = X_Y_to_U(X, Y)
    G = omega_grad(p, bvectors, M, U)
    return GU_to_GX_GY(G, X, Y, frozen)
end
