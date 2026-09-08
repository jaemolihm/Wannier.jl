using NearestNeighbors: knn, KDTree

"""
    sorted_multi_indices(d, n)

Generate all sorted multi-indices of length `n` in `{1,...,d}`.

These index the unique elements of a symmetric `n`-th order tensor in `d` dimensions.
The number of such indices is `binomial(d + n - 1, n)`.

# Examples
```julia
sorted_multi_indices(3, 2)  # [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]
```
"""
function sorted_multi_indices(d::Int, n::Int)
    indices = NTuple{n,Int}[]
    # recursive generation of sorted tuples
    function _generate(current::Vector{Int}, start::Int)
        if length(current) == n
            push!(indices, NTuple{n,Int}(current))
            return
        end
        for i in start:d
            push!(current, i)
            _generate(current, i)
            pop!(current)
        end
    end
    _generate(Int[], 1)
    return indices
end

"""
    symmetric_tensor_sum(bvectors_shell, n)

For a shell of b-vectors, compute the unique elements of `∑_b b^⊗n`.

Returns a vector of length `binomial(3 + n - 1, n)` containing the unique elements
of the symmetric tensor `∑_b ∏_{j ∈ multi_index} b[j]`.
"""
function symmetric_tensor_sum(bvectors_shell::Vector{Vec3{T}}, n::Int) where {T}
    indices = sorted_multi_indices(3, n)
    result = zeros(T, length(indices))
    for b in bvectors_shell
        for (idx, mi) in enumerate(indices)
            result[idx] += prod(b[j] for j in mi)
        end
    end
    return result
end

"""
    $(TYPEDEF)

Shells of b-vectors.

The ``\\mathbf{b}``-vectors are are vectors connecting a kpoint to its neighboring
kpoints. To find a stencil for approximating finite differences, the neighboring
kpoints are sorted by their distance to the original kpoint, such that equal-distance
``\\mathbf{b}``-vectors are grouped into one shell.

# Fields
$(FIELDS)
"""
struct KspaceStencilShells{T<:Real}
    """reciprocal lattice vectors, 3 * 3, each column is a reciprocal lattice
    vector in Å⁻¹ unit"""
    recip_lattice::Mat3{T}

    """number of kpoints along three reciprocal lattice vectors"""
    kgrid_size::Vec3{Int}

    """kpoint fractional coordinates, length-`n_kpoints` vector of `Vec3`.
    Should be a uniformlly-spaced kpoint grid."""
    kpoints::Vector{Vec3{T}}

    """dengeneracy (i.e. the number of bvectors) of each shell,
    length-`n_shells` vector of integers"""
    n_degens::Vector{Int}

    """bvectors of each shell, Cartesian coordinates in Å⁻¹ unit,
    length-`n_shells` vector, each element is a length-`n_degens[i_shell]`
    vector of `Vec3`.

    Here we use Cartesian coordinates because it is easier to compute the
    completeness condition (MV1997 Eq. (B1)).
    """
    bvectors::Vector{Vector{Vec3{T}}}

    """bvector weight of each shell, length-`n_shells` vector, Å² unit."""
    bweights::Vector{T}
end

n_kpoints(shells::KspaceStencilShells) = length(shells.kpoints)
"""number of b-vector shells"""
n_shells(shells::KspaceStencilShells) = length(shells.n_degens)
n_bvectors(shells::KspaceStencilShells) = sum(shells.n_degens)
reciprocal_lattice(shells::KspaceStencilShells) = shell.recip_lattice

"""
    $(SIGNATURES)

Convenience constructor of `KspaceStencilShells`, auto set `n_degens`.

# Arguments
See the fields of [`KspaceStencilShells`](@ref) struct.
"""
function KspaceStencilShells(recip_lattice, kgrid_size, kpoints, bvectors, bweights)
    n_degens = [length(bvecs) for bvecs in bvectors]
    return KspaceStencilShells(
        recip_lattice, Vec3(kgrid_size), kpoints, n_degens, bvectors, bweights
    )
end

function KspaceStencilShells(
    recip_lattice, kgrid_size, kpoints;
    atol=default_w90_kmesh_tol(),
    order::Int=1,
)
    # find shells, search more shells for higher-order FD
    shells = search_shells(
        recip_lattice, kgrid_size, kpoints;
        atol, max_shells=default_w90_bvectors_search_shells() * order,
    )
    keep_shells = check_parallel(shells)
    shells = delete_shells(shells, keep_shells)

    keep_shells, bweights = compute_bweights(shells; atol, order)
    shells = delete_shells(shells, keep_shells)
    shells.bweights .= bweights

    # Γ-point calculation only keep half of the bvectors
    if all(kgrid_size .== 1)
        shells = delete_shells_Γ(shells)
    end

    check_completeness(shells; atol, order)
    return shells
end

function Base.show(io::IO, ::MIME"text/plain", shells::KspaceStencilShells)
    nshells = n_shells(shells)
    @printf(io, "                 [bx, by, bz] (Å⁻¹)\n")
    for (ish, (bvecs, w)) in enumerate(zip(shells.bvectors, shells.bweights))
        n = isempty(bvecs) ? NaN : norm(bvecs[1])
        @printf(
            io, "b-vector shell %3d:    norm = %8.5f (Å⁻¹)   weight = %8.5f (Å²)", ish, n, w
        )
        for (ib, bvec) in enumerate(shells.bvectors[ish])
            @printf(io, "\n%3d    %11.6f %11.6f %11.6f", ib, bvec...)
        end
        ish != nshells && println(io)
    end
end

"""
    $(SIGNATURES)

Search bvector shells satisfing completeness condition.

# Arguments
- `recip_lattice`: each column is a reciprocal lattice vector
- `kpoints`: fractional coordinates

# Keyword Arguments
- `atol`: tolerance to select a shell (points having equal distances)
- `max_shells`: max number of nearest-neighbor shells

# Return
- a `KspaceStencilShells` struct, note the `bweights` are not computed yet, all zeros!

!!! note
    To reproduce wannier90's behavior,
    - `atol` should be set to wannier90's input parameter `kmesh_tol`
    - `max_shells` should be set to wannier90's input parameter `search_shells`
"""
function search_shells(
    recip_lattice::Mat3,
    kgrid_size::AbstractVector,
    kpoints::AbstractVector;
    atol=default_w90_kmesh_tol(),
    max_shells=default_w90_bvectors_search_shells(),
)
    # Usually these "magic" numbers work well for normal recip_lattice.
    # Number of nearest-neighbors to be returned
    max_neighbors = 500
    # Max number of stencils in one shell
    max_degens = 40
    # max_shells = round(Int, max_neighbors / max_degens)

    # 1. Generate a supercell to search bvectors
    supercell, _ = make_supercell(kpoints)
    # To cartesian coordinates
    supercell_cart = map(supercell) do cell
        recip_lattice * cell
    end
    # use the 1st kpt to search bvectors, usually Γ point
    kpt_orig = recip_lattice * kpoints[1]

    # 2. KDTree to search nearest neighbors
    kdtree = KDTree(supercell_cart)
    idxs, dists = knn(kdtree, kpt_orig, max_neighbors, true)
    # activate debug info with: JULIA_DEBUG=Main julia
    # @debug "KDTree nearest neighbors" dists
    # @debug "KDTree nearest neighbors" idxs

    # 3. Arrange equal-distance kpoint indices in layer of shells
    shells = Vector{Vector{Int}}()

    # The 1st result is the search point itself, dist = 0
    inb = 2  # index of neighbors
    ish = 1  # index of shells
    while (inb <= max_neighbors) && (ish <= max_shells)
        # use the 1st kpoint to find bvector shells & bweights
        eqdist_idxs = findall(x -> isapprox(x, dists[inb]; atol), dists)
        degen = length(eqdist_idxs)
        if degen >= max_degens
            # skip large-degeneracy shells
            inb += degen
            break
        end
        push!(shells, idxs[eqdist_idxs])
        inb += degen
        ish += 1
    end

    # 4. Get Cartesian-coordinate vectors
    T = eltype(recip_lattice)
    bvectors = map(shells) do idxs
        kpb_cart = supercell_cart[idxs]
        return map(v -> Vec3{T}(v .- kpt_orig), kpb_cart)
    end
    @debug "Found bvector shells" bvectors

    bweights = zeros(T, length(shells))
    return KspaceStencilShells(recip_lattice, kgrid_size, kpoints, bvectors, bweights)
end

"""
    $(SIGNATURES)

Check if the columns of matrix `A` and columns of matrix `B` are parallel.

# Arguments
- `A`: vector, each element is a vector, often `Vec3`
- `B`: similar to `A`

# Keyword Arguments
- `atol`: tolerance to check parallelism.

# Return
- `checkerboard`: boolean matrix, `checkerboard[i, j]` is `true`
    if `A[i]` and `B[j]` are parallel

!!! note
    Wannier90 uses a constant `1e-6` as `atol`, thus here its default is set to
    `1e-6` as well to reproduce the same result.
"""
function are_parallel(
    A::AbstractVector, B::AbstractVector; atol=default_w90_bvectors_check_parallel_atol()
)
    n_A = length(A)
    n_B = length(B)

    checkerboard = fill(false, n_A, n_B)

    for (i, a) in enumerate(A)
        for (j, b) in enumerate(B)
            p = cross(a, b)
            if all(isapprox.(0, p; atol))
                checkerboard[i, j] = true
            end
        end
    end

    return checkerboard
end

"""
    $(SIGNATURES)

Check if shells having parallel bvectors.

# Arguments
- `bvectors`: vector of bvectors in each shell

# Keyword Arguments
- `atol`: tolerance to check parallelism, see [`are_parallel`](@ref)

# Return
- `keep_shells`: indices of shells that do not have parallel bvectors

!!! note

    To reproduce wannier90's behavior,
    - `atol` should be set to wannier90's internal constant `1e-6`.
"""
function check_parallel(
    bvectors::Vector{Vector{Vec3{T}}}; atol=default_w90_bvectors_check_parallel_atol()
) where {T}
    nshells = length(bvectors)
    keep_shells = collect(1:nshells)

    for ish in 2:nshells
        for jsh in 1:(ish - 1)
            if !(jsh in keep_shells)
                continue
            end

            p = are_parallel(bvectors[jsh], bvectors[ish]; atol)
            if any(p)
                @debug "has parallel bvectors between shells $jsh $ish"
                filter!(s -> s != ish, keep_shells)
                break
            end
        end
    end
    return keep_shells
end

"""
    $(SIGNATURES)

Check if shells having parallel bvectors.

# Arguments
- `shells`: `KspaceStencilShells` containing bvectors in each shell
"""
function check_parallel(shells::KspaceStencilShells)
    return check_parallel(shells.bvectors)
end

function delete_shells(bvectors::Vector{Vector{Vec3{T}}}, keep_shells) where {T}
    return bvectors[keep_shells]
end

"""
    $(SIGNATURES)

Remove shells.

# Arguments
- `keep_shells`: indices of shells to keep
"""
function delete_shells(shells::KspaceStencilShells, keep_shells)
    bvectors = delete_shells(shells.bvectors, keep_shells)
    bweights = shells.bweights[keep_shells]
    return KspaceStencilShells(
        shells.recip_lattice, shells.kgrid_size, shells.kpoints, bvectors, bweights
    )
end

"""
    $(SIGNATURES)

Delete negetive bvectors for Γ-point calculation.

Since bvectors are symmetric, this removes half of the bvectors.
"""
function delete_shells_Γ(shells::KspaceStencilShells)
    bvectors = map(shells.bvectors) do bvecs  # for each shell
        bvecs_new = filter(v -> all(v .>= 0), bvecs)
        if length(bvecs_new) != length(bvecs)//2
            error("Non-symmetric bvectors for Γ-point calculation: ", bvecs)
        end
        bvecs_new
    end
    bweights = [2w for w in shells.bweights]
    return KspaceStencilShells(
        shells.recip_lattice, shells.kgrid_size, shells.kpoints, bvectors, bweights
    )
end

"""
    $(SIGNATURES)

Try to guess bvector bweights from MV1997 Eq. (B1).

The input bvectors are overcomplete vectors found during shell search, i.e., from
[`search_shells`](@ref). This function tries to find the minimum number of bvector
shells that satisfy the B1 condition, and return the new `KspaceStencilShells` and bweights.

# Arguments
- `bvectors`: vector of bvectors in each shell

# Keyword Arguments
- `atol`: tolerance to satisfy B1 condition

!!! note

    To reproduce wannier90's behavior,
    - `atol` should be set to wannier90's input parameter `kmesh_tol`
"""
function compute_bweights(
    bvectors::Vector{Vector{Vec3{T}}}; atol=default_w90_kmesh_tol(), order::Int=1
) where {T}
    nshells = length(bvectors)
    @assert nshells > 0 "empty bvectors"
    @assert order >= 1 "order must be >= 1"

    # Number of rows = sum of unique elements of symmetric tensors of even orders 2,4,...,2*order
    # For d=3: binomial(3+2k-1, 2k) = 6, 15, 28 for k=1,2,3
    nrows = sum(binomial(3 + 2k - 1, 2k) for k in 1:order)

    B = zeros(T, nrows, nshells)

    # Target vector: 2nd-order moment = identity (unique elements of δ_ij),
    # higher-order moments = 0
    target = zeros(T, nrows)
    # Fill the 2nd-order target: unique elements of I₃ for sorted pairs
    # (1,1) -> 1, (1,2) -> 0, (1,3) -> 0, (2,2) -> 1, (2,3) -> 0, (3,3) -> 1
    indices_2 = sorted_multi_indices(3, 2)
    for (idx, mi) in enumerate(indices_2)
        target[idx] = (mi[1] == mi[2]) ? one(T) : zero(T)
    end

    # weight of each shell
    W = zeros(nshells)

    # singular value tolerance, to reproduce W90 behavior
    σ_atol = default_w90_bvectors_singular_value_atol()
    σ_atol = 0.0

    keep_shells = zeros(Int, 0)
    ish = 1
    while ish <= nshells
        push!(keep_shells, ish)
        # Build column for this shell: concatenate symmetric_tensor_sum for each even order
        col = T[]
        for k in 1:order
            append!(col, symmetric_tensor_sum(bvectors[ish], 2k))
        end
        B[:, ish] = col
        # Solve equation B * W = target
        # B = U * S * V' -> W = V * S^-1 * U' * target
        U, S, V = svd(B[:, keep_shells])
        @debug "S" ish S = S' keep_shells = keep_shells'
        if all(S .> σ_atol)
            W .= 0
            W[keep_shells] .= B[:, keep_shells] \ target
            BW = B[:, keep_shells] * W[keep_shells]
            @debug "BW" ish BW = BW'
            if isapprox(BW, target; atol)
                break
            end
        else
            pop!(keep_shells)
        end
        ish += 1
    end
    if ish == nshells + 1
        error("not enough shells to satisfy completeness condition (order=$order)")
    end

    bweights = W[keep_shells]
    return keep_shells, bweights
end

"""
    $(SIGNATURES)

Try to guess bvector bweights from MV1997 Eq. (B1).

# Arguments
- `shells`: `KspaceStencilShells` containing bvectors in each shell

# Keyword Arguments
- `atol`: tolerance to satisfy B1 condition

!!! note

    To reproduce wannier90's behavior,
    - `atol` should be set to wannier90's input parameter `kmesh_tol`
"""
function compute_bweights(shells::KspaceStencilShells; atol=default_w90_kmesh_tol(), order::Int=1)
    return compute_bweights(shells.bvectors; atol, order)
end

"""
    $(SIGNATURES)

Check completeness (B1 condition) of `KspaceStencilShells`.

# Arguments
- `shells`: `KspaceStencilShells` containing bvectors in each shell

# Keyword Arguments
- `atol`: floating point tolerance

!!! note

    To reproduce wannier90's behavior,
    - `atol` should be set to wannier90's input parameter `kmesh_tol`
"""
function check_completeness(
    shells::KspaceStencilShells{T}; atol=default_w90_kmesh_tol(), order::Int=1
) where {T}
    for k in 1:order
        n = 2k
        # Compute moment: M_{2k} = ∑_shells w * symmetric_tensor_sum(bvecs, 2k)
        indices = sorted_multi_indices(3, n)
        M = zeros(T, length(indices))
        for (bvecs, w) in zip(shells.bvectors, shells.bweights)
            M .+= w .* symmetric_tensor_sum(bvecs, n)
        end

        if k == 1
            # 2nd moment should equal δ_ij (unique elements of identity)
            target = T[mi[1] == mi[2] ? one(T) : zero(T) for mi in indices]
        else
            # Higher even moments should be zero
            target = zeros(T, length(indices))
        end

        @debug "Bvector moment order $n" M target
        Δ = M - target
        if !all(isapprox.(Δ, 0; atol))
            error("""b-vector completeness condition not satisfied for moment order $n
                     atol = $atol
                     Δ = $(maximum(abs.(Δ)))
                     try increasing atol?""")
        end
    end

    @info "b-vector completeness condition satisfied (order=$order)"
    return nothing
end

"""
    $(SIGNATURES)

Unwrap nested shell vectors into a flattened vector.

# Return
- `bvectors`: length-`n_bvectors` vector, each element is a `Vec3`
- `bweights`: length-`n_bvectors` vector of bweights for each bvector
"""
function flatten_shells(shells::KspaceStencilShells{T}) where {T}
    nbvecs = n_bvectors(shells)

    bvectors = zeros(Vec3{T}, nbvecs)
    bweights = zeros(T, nbvecs)

    counter = 1
    for (bvecs, w, degen) in zip(shells.bvectors, shells.bweights, shells.n_degens)
        bvectors[counter:(counter + degen - 1)] = bvecs
        bweights[counter:(counter + degen - 1)] .= w
        counter += degen
    end

    return bvectors, bweights
end
