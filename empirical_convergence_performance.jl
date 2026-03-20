using Pkg
Pkg.activate(".")

using Distributed
if nprocs() == 1
    addprocs()
end

@everywhere begin
    using PsfUnmixing
    using Random, LinearAlgebra, Statistics
    using DelimitedFiles
    using ProgressMeter

    # ------------------- kernel + ranges -------------------
    u = 0.5 # naturalized gaussian kernel

    s_min = 1e-3
    s_max = 1e-2

    θ_min = s_laplace(s_min, u, s_min)
    θ_max = s_laplace(s_max, u, s_min)

    pd_list = [(2, 1), (5, 5)]

    k_0(θ, t) = laplace_0_n(θ, t, u, umin = s_min)
    k_1(θ, t) = laplace_1_n(θ, t, u, umin = s_min)
    k_2(θ, t) = laplace_2_n(θ, t, u, umin = s_min)
    k_3(θ, t) = laplace_3_n(θ, t, u, umin = s_min)
end

# ------------------------------------------------------------
# Joint residual/Jacobian/Hessian (θ,η)  [full LS]
# ------------------------------------------------------------
@everywhere function residual(θ, η, x_obs, dictionary, xgrid)
    @assert length(dictionary) == length(θ)
    @assert sum(length.(dictionary)) == length(η)
    @assert length(x_obs) == length(xgrid)
    A = multi_block(k_0, θ, dictionary, xgrid)
    return x_obs - A * η
end

@everywhere function block_diag_matrices(matrices::Vector{<:AbstractMatrix})
    total_rows = sum(size.(matrices, 1))
    total_cols = sum(size.(matrices, 2))
    result = zeros(eltype(first(matrices)), total_rows, total_cols)

    row_start = 1
    col_start = 1
    for mat in matrices
        rows, cols = size(mat)
        result[row_start:(row_start + rows - 1), col_start:(col_start + cols - 1)] .= mat
        row_start += rows
        col_start += cols
    end
    return result
end

@everywhere function jacobian(θ, η, dictionary, xgrid)
    @assert length(dictionary) == length(θ)
    @assert sum(length.(dictionary)) == length(η)

    d = length(dictionary[1])

    G0 = multi_block(k_0, θ, dictionary, xgrid)
    G1 = multi_block(k_1, θ, dictionary, xgrid)
    diag_eta = block_diag(collect(Iterators.partition(η, d)))

    # NOTE: this is d(A(θ)η)/d(θ,η); residual is r = x - Aη,
    # so dr/dz = -[G1*diag_eta, G0]
    return hcat(G1 * diag_eta, G0)
end

@everywhere function hessian(θ, η, x_obs, dictionary, xgrid)
    @assert length(dictionary) == length(θ)
    @assert sum(length.(dictionary)) == length(η)
    @assert length(x_obs) == length(xgrid)

    d = length(dictionary[1])
    p = length(θ)

    G0 = multi_block(k_0, θ, dictionary, xgrid)
    G1 = multi_block(k_1, θ, dictionary, xgrid)

    r = residual(θ, η, x_obs, dictionary, xgrid)
    diag_eta = block_diag(collect(Iterators.partition(η, d)))

    ∇_θr = G1 * diag_eta
    ∇_ηr = G0

    C = vcat(
        hcat(transpose(∇_θr) * ∇_θr, transpose(∇_θr) * ∇_ηr),
        hcat(transpose(∇_ηr) * ∇_θr, transpose(∇_ηr) * ∇_ηr),
    )

    G1_blocks = Vector{Matrix{Float64}}()
    G2_blocks = Vector{Matrix{Float64}}()
    for (group, θ_i) in zip(dictionary, θ)
        push!(G1_blocks, single_block(k_1, θ_i, group, xgrid))
        push!(G2_blocks, single_block(k_2, θ_i, group, xgrid))
    end

    r∇_θr = transpose(kron(I(p), r)) * block_diag_matrices(G2_blocks) * diag_eta
    r∇_ηθr = transpose(block_diag_matrices(G1_blocks)) * kron(I(p), r)

    R = vcat(
        hcat(r∇_θr, transpose(r∇_ηθr)),
        hcat(r∇_ηθr, zeros(d * p, d * p)),
    )

    return C - R
end

# ------------------------------------------------------------
# Variable projection residual/Jacobian/Hessian in θ only  [VP]
# ------------------------------------------------------------
@everywhere function residual_vp(θ, x_obs, dictionary, xgrid)
    @assert length(dictionary) == length(θ)
    @assert length(x_obs) == length(xgrid)
    G0 = multi_block(k_0, θ, dictionary, xgrid)
    η̂ = G0 \ x_obs
    return x_obs - G0 * η̂
end

@everywhere function jacobian_vp(θ, x_obs, dictionary, xgrid)
    @assert length(dictionary) == length(θ)
    @assert length(x_obs) == length(xgrid)

    d = length(dictionary[1])

    G0 = multi_block(k_0, θ, dictionary, xgrid)
    G1 = multi_block(k_1, θ, dictionary, xgrid)

    η̂ = G0 \ x_obs
    diag_eta = block_diag(collect(Iterators.partition(η̂, d)))

    grad_theta_r = G1 * diag_eta

    F = svd(G0)
    U = F.U

    # dr/dθ = -(I-UU') * grad_theta_r
    return -grad_theta_r + U * transpose(U) * grad_theta_r
end

@everywhere function hessian_vp(θ, x_obs, dictionary, xgrid)
    @assert length(dictionary) == length(θ)
    @assert length(x_obs) == length(xgrid)

    d = length(dictionary[1])
    p = length(θ)

    G0 = multi_block(k_0, θ, dictionary, xgrid)
    G1 = multi_block(k_1, θ, dictionary, xgrid)

    r = residual_vp(θ, x_obs, dictionary, xgrid)
    η̂ = G0 \ x_obs
    diag_eta = block_diag(collect(Iterators.partition(η̂, d)))

    ∇_θr = G1 * diag_eta
    ∇_ηr = G0

    G1_tensor = Vector{Matrix{Float64}}()
    G2_blocks = Vector{Matrix{Float64}}()
    for (group, θ_i) in zip(dictionary, θ)
        push!(G1_tensor, single_block(k_1, θ_i, group, xgrid))
        push!(G2_blocks, single_block(k_2, θ_i, group, xgrid))
    end

    r∇_θr = -transpose(kron(I(p), r)) * block_diag_matrices(G2_blocks) * diag_eta
    r∇_ηθr = -transpose(block_diag_matrices(G1_tensor)) * kron(I(p), r)

    ∇_θL = transpose(∇_θr) * ∇_θr + r∇_θr
    ∇_ηθL = transpose(∇_ηr) * ∇_θr + r∇_ηθr

    M = transpose(G0) * G0 + 1e-8 * I
    return ∇_θL - transpose(∇_ηθL) * (M \ ∇_ηθL)
end

# ------------------------------------------------------------
# Perturbations
# ------------------------------------------------------------
@everywhere function projected_perturbation(θ_star, ϵ; rng = Random.default_rng())
    dθ = randn(rng, length(θ_star))
    return θ_star + ϵ * dθ ./ norm(dθ, 2)
end

# ------------------------------------------------------------
# Loss + gradient norms for least-squares objectives
#   L(z) = 0.5 ||r(z)||^2
#   ∇L(z) = J(z)' r(z)        where J = dr/dz
# ------------------------------------------------------------
@everywhere loss_from_residual(r::AbstractVector) = 0.5 * dot(r, r)

@everywhere function grad_from_Jr(J::AbstractMatrix, r::AbstractVector)
    # ∇L = J' r
    return transpose(J) * r
end

# ------------------------------------------------------------
# Iteration traces for:
#   - Levenberg–Marquardt (LM): (J'J + λI)δ = -J'r with adaptive λ
#   - Gauss–Newton (GN):        (J'J)δ      = -J'r
#   - Gradient descent (GD):    z <- z - α ∇L with backtracking
#
# IMPORTANT: we record per-iteration:
#   - loss value  L_k
#   - gradient norm ||∇L_k||
#
# We always record iter 0 (initial point) plus maxiters steps.
# ------------------------------------------------------------
@everywhere function trace_gn_lm(
    rfun!, jfun!, z0::Vector{Float64}, m::Int;
    maxiters::Int = 100,
    method::Symbol = :LM,          # :LM or :GN
    λ0::Float64 = 1e-3,
    ν::Float64 = 10.0,             # LM damping multiplier
    λ_min::Float64 = 1e-12,
    λ_max::Float64 = 1e12,
)
    z = copy(z0)

    r = zeros(Float64, m)
    J = zeros(Float64, m, length(z))

    losses = fill(NaN, maxiters + 1)
    gradnorms = fill(NaN, maxiters + 1)

    # eval at iter 0
    rfun!(r, z)
    jfun!(J, z)
    g = grad_from_Jr(J, r)
    L = loss_from_residual(r)

    losses[1] = L
    gradnorms[1] = norm(g)

    λ = (method == :LM) ? λ0 : 0.0

    for k in 1:maxiters
        # build normal equations
        JtJ = transpose(J) * J
        Jtr = transpose(J) * r
        A = Symmetric(JtJ + λ * I)
        δ = -(A \ Jtr)

        z_trial = z .+ δ

        # evaluate trial
        r_trial = similar(r)
        rfun!(r_trial, z_trial)
        L_trial = loss_from_residual(r_trial)

        if method == :GN
            # accept always
            z .= z_trial
            r .= r_trial
            jfun!(J, z)
        else
            # LM: accept if loss decreases, else increase λ and retry next iter
            if isfinite(L_trial) && (L_trial <= L)
                z .= z_trial
                r .= r_trial
                jfun!(J, z)

                # decrease damping
                λ = max(λ / ν, λ_min)
            else
                # reject step, increase damping; keep z, r, J
                λ = min(λ * ν, λ_max)
            end
        end

        # record
        g = grad_from_Jr(J, r)
        L = loss_from_residual(r)

        losses[k + 1] = L
        gradnorms[k + 1] = norm(g)
    end

    return losses, gradnorms
end

@everywhere function trace_gd(
    rfun!, jfun!, z0::Vector{Float64}, m::Int;
    maxiters::Int = 100,
    α0::Float64 = 1.0,
    β::Float64 = 0.5,         # backtracking shrink
    c1::Float64 = 1e-4,       # Armijo parameter
    α_min::Float64 = 1e-12,
)
    z = copy(z0)

    r = zeros(Float64, m)
    J = zeros(Float64, m, length(z))

    losses = fill(NaN, maxiters + 1)
    gradnorms = fill(NaN, maxiters + 1)

    # iter 0
    rfun!(r, z)
    jfun!(J, z)
    g = grad_from_Jr(J, r)
    L = loss_from_residual(r)

    losses[1] = L
    gradnorms[1] = norm(g)

    for k in 1:maxiters
        g = grad_from_Jr(J, r)
        gnorm2 = dot(g, g)
        if !isfinite(gnorm2) || gnorm2 == 0.0
            # stagnation
            losses[k + 1] = L
            gradnorms[k + 1] = norm(g)
            continue
        end

        α = α0
        z_trial = similar(z)
        r_trial = similar(r)

        # backtracking on loss
        while true
            z_trial .= z .- α .* g
            rfun!(r_trial, z_trial)
            L_trial = loss_from_residual(r_trial)

            if isfinite(L_trial) && (L_trial <= L - c1 * α * gnorm2)
                # accept
                z .= z_trial
                r .= r_trial
                jfun!(J, z)
                L = L_trial
                break
            end

            α *= β
            if α < α_min
                # give up, accept no-op
                break
            end
        end

        g = grad_from_Jr(J, r)

        losses[k + 1] = L
        gradnorms[k + 1] = norm(g)
    end

    return losses, gradnorms
end

# ------------------------------------------------------------
# Lipschitz constants (unchanged)
# ------------------------------------------------------------
@everywhere function lipschitz_constant(k, dictionary, xgrid)
    θ_range = range(θ_min, θ_max, length = 100)
    opnorms = zeros(Float64, length(θ_range), length(dictionary))
    for (i, θ) in enumerate(θ_range)
        for (j, group) in enumerate(dictionary)
            opnorms[i, j] = opnorm(single_block(k, θ, group, xgrid))
        end
    end
    return maximum(opnorms)
end

# ------------------------------------------------------------
# One noise realization:
#   For BOTH problems:
#     - VP (θ only)
#     - full LS (θ,η)
#   Run THREE methods:
#     - LM
#     - GN
#     - GD
#   Record per-iteration (loss, gradnorm)
# ------------------------------------------------------------
@everywhere function one_noisy_realization_traces(
    p::Int, d::Int, r::Int,
    θ_star::Vector{Float64}, η_star::Vector{Float64},
    dictionary, xgrid::Vector{Float64},
    x_obs_clean::Vector{Float64},
    snr_db::Float64, noise_seed::Int,
    ϵ_init::Float64,
    maxiters::Int,
)
    rng_noise = MersenneTwister(noise_seed + r + 10_000 * p + d)

    # noise
    w_dir = randn(rng_noise, length(x_obs_clean))
    w_dir ./= norm(w_dir)
    w_norm = (isfinite(snr_db) ? norm(x_obs_clean) * 10.0^(-snr_db / 20.0) : 0.0)
    w = w_norm * w_dir
    x_obs_noisy = x_obs_clean + w

    # shared init θ
    θ_init = projected_perturbation(θ_star, ϵ_init; rng = rng_noise)

    # induced η_init from θ_init
    A_init = multi_block(k_0, θ_init, dictionary, xgrid)
    η_init = A_init \ x_obs_noisy

    # ---------------- VP (θ only) setup ----------------
    m = length(x_obs_noisy)

    function rfun_vp!(out, θ)
        out .= residual_vp(θ, x_obs_noisy, dictionary, xgrid)
        return out
    end

    function jfun_vp!(J, θ)
        J .= jacobian_vp(θ, x_obs_noisy, dictionary, xgrid)
        return J
    end

    # ---------------- full LS ((θ,η)) setup ----------------
    z0 = vcat(θ_init, η_init)

    function rfun_ls!(out, z)
        θ = @view z[1:p]
        η = @view z[(p + 1):end]
        out .= residual(θ, η, x_obs_noisy, dictionary, xgrid)
        return out
    end

    function jfun_ls!(J, z)
        θ = @view z[1:p]
        η = @view z[(p + 1):end]
        # dr/dz = - jacobian(...) (see comment in jacobian)
        J .= -jacobian(θ, η, dictionary, xgrid)
        return J
    end

    # ---------- VP traces ----------
    vp_LM_loss, vp_LM_grad = trace_gn_lm(rfun_vp!, jfun_vp!, copy(θ_init), m; maxiters=maxiters, method=:LM)
    vp_GN_loss, vp_GN_grad = trace_gn_lm(rfun_vp!, jfun_vp!, copy(θ_init), m; maxiters=maxiters, method=:GN)
    vp_GD_loss, vp_GD_grad = trace_gd(rfun_vp!, jfun_vp!, copy(θ_init), m; maxiters=maxiters)

    # ---------- LS traces ----------
    ls_LM_loss, ls_LM_grad = trace_gn_lm(rfun_ls!, jfun_ls!, copy(z0), m; maxiters=maxiters, method=:LM)
    ls_GN_loss, ls_GN_grad = trace_gn_lm(rfun_ls!, jfun_ls!, copy(z0), m; maxiters=maxiters, method=:GN)
    ls_GD_loss, ls_GD_grad = trace_gd(rfun_ls!, jfun_ls!, copy(z0), m; maxiters=maxiters)

    return (
        vp_LM_loss = vp_LM_loss, vp_LM_grad = vp_LM_grad,
        vp_GN_loss = vp_GN_loss, vp_GN_grad = vp_GN_grad,
        vp_GD_loss = vp_GD_loss, vp_GD_grad = vp_GD_grad,
        ls_LM_loss = ls_LM_loss, ls_LM_grad = ls_LM_grad,
        ls_GN_loss = ls_GN_loss, ls_GN_grad = ls_GN_grad,
        ls_GD_loss = ls_GD_loss, ls_GD_grad = ls_GD_grad,
    )
end

# ------------------------------------------------------------
# Main experiment: pmap over noise realizations, aggregate traces
# ------------------------------------------------------------
function basin_trace_data_parallel_pd(
    θ_min, θ_max, pd_list, k_0, k_1, k_2, k_3;
    N::Int = 10_000, T::Real = 1.0, Δ::Real = 1e-2,
    snr_db::Real = Inf,
    realizations::Int = 10,
    dict_seed::Int = 1234,
    noise_seed::Int = 1,
    ϵ_init::Real = 1e-1,
    maxiters::Int = 100,
)
    xgrid = collect(range(-T, T, length = N))

    out = Dict{Tuple{Int, Int}, Any}()
    prog = Progress(length(pd_list); desc = "traces per (p,d)")

    for (p, d) in pd_list
        rng_dict = MersenneTwister(dict_seed + 10_000 * p + d)

        θ_star = 0.5 * (θ_min + θ_max) * ones(p)
        η_star = ones(p * d)

        dictionary = generate_spike_groups(rng_dict, -T, T, Δ, p, d)
        x_obs_clean = multi_block(k_0, θ_star, dictionary, xgrid) * η_star

        rs = collect(1:realizations)
        res = pmap(rs) do r
            one_noisy_realization_traces(
                p, d, r,
                Float64.(θ_star), Float64.(η_star),
                dictionary, xgrid,
                Float64.(x_obs_clean),
                float(snr_db), noise_seed,
                float(ϵ_init),
                maxiters,
            )
        end

        # stack helper: build matrix (R x (maxiters+1)) for a given field
        function stack(field::Symbol)
            M = reduce(vcat, (reshape(getfield(z, field), 1, :) for z in res))
            return M
        end

        # compute mean/std vectors
        function mean_std(field::Symbol)
            M = stack(field)
            return (vec(mean(M, dims=1)), vec(std(M, dims=1)))
        end

        vp_LM_loss_m, vp_LM_loss_s = mean_std(:vp_LM_loss)
        vp_LM_grad_m, vp_LM_grad_s = mean_std(:vp_LM_grad)

        vp_GN_loss_m, vp_GN_loss_s = mean_std(:vp_GN_loss)
        vp_GN_grad_m, vp_GN_grad_s = mean_std(:vp_GN_grad)

        vp_GD_loss_m, vp_GD_loss_s = mean_std(:vp_GD_loss)
        vp_GD_grad_m, vp_GD_grad_s = mean_std(:vp_GD_grad)

        ls_LM_loss_m, ls_LM_loss_s = mean_std(:ls_LM_loss)
        ls_LM_grad_m, ls_LM_grad_s = mean_std(:ls_LM_grad)

        ls_GN_loss_m, ls_GN_loss_s = mean_std(:ls_GN_loss)
        ls_GN_grad_m, ls_GN_grad_s = mean_std(:ls_GN_grad)

        ls_GD_loss_m, ls_GD_loss_s = mean_std(:ls_GD_loss)
        ls_GD_grad_m, ls_GD_grad_s = mean_std(:ls_GD_grad)

        out[(p, d)] = (
            iters = collect(0:maxiters),

            vp_LM_loss_mean = vp_LM_loss_m, vp_LM_loss_std = vp_LM_loss_s,
            vp_LM_grad_mean = vp_LM_grad_m, vp_LM_grad_std = vp_LM_grad_s,

            vp_GN_loss_mean = vp_GN_loss_m, vp_GN_loss_std = vp_GN_loss_s,
            vp_GN_grad_mean = vp_GN_grad_m, vp_GN_grad_std = vp_GN_grad_s,

            vp_GD_loss_mean = vp_GD_loss_m, vp_GD_loss_std = vp_GD_loss_s,
            vp_GD_grad_mean = vp_GD_grad_m, vp_GD_grad_std = vp_GD_grad_s,

            ls_LM_loss_mean = ls_LM_loss_m, ls_LM_loss_std = ls_LM_loss_s,
            ls_LM_grad_mean = ls_LM_grad_m, ls_LM_grad_std = ls_LM_grad_s,

            ls_GN_loss_mean = ls_GN_loss_m, ls_GN_loss_std = ls_GN_loss_s,
            ls_GN_grad_mean = ls_GN_grad_m, ls_GN_grad_std = ls_GN_grad_s,

            ls_GD_loss_mean = ls_GD_loss_m, ls_GD_loss_std = ls_GD_loss_s,
            ls_GD_grad_mean = ls_GD_grad_m, ls_GD_grad_std = ls_GD_grad_s,
        )

        next!(prog)
    end

    return out
end

# ------------------------------------------------------------
# Save to CSV (wide format, per (p,d,iter))
# ------------------------------------------------------------
function save_trace_csv(path::AbstractString, out::Dict{Tuple{Int, Int}, Any})
    rows = Any[]

    push!(rows, [
        "p","d","iter",
        "vp_LM_loss_mean","vp_LM_loss_std","vp_LM_grad_mean","vp_LM_grad_std",
        "vp_GN_loss_mean","vp_GN_loss_std","vp_GN_grad_mean","vp_GN_grad_std",
        "vp_GD_loss_mean","vp_GD_loss_std","vp_GD_grad_mean","vp_GD_grad_std",
        "ls_LM_loss_mean","ls_LM_loss_std","ls_LM_grad_mean","ls_LM_grad_std",
        "ls_GN_loss_mean","ls_GN_loss_std","ls_GN_grad_mean","ls_GN_grad_std",
        "ls_GD_loss_mean","ls_GD_loss_std","ls_GD_grad_mean","ls_GD_grad_std",
    ])

    for ((p, d), B) in sort(collect(out); by = x -> x[1])
        for i in eachindex(B.iters)
            push!(rows, [
                p, d, B.iters[i],

                B.vp_LM_loss_mean[i], B.vp_LM_loss_std[i], B.vp_LM_grad_mean[i], B.vp_LM_grad_std[i],
                B.vp_GN_loss_mean[i], B.vp_GN_loss_std[i], B.vp_GN_grad_mean[i], B.vp_GN_grad_std[i],
                B.vp_GD_loss_mean[i], B.vp_GD_loss_std[i], B.vp_GD_grad_mean[i], B.vp_GD_grad_std[i],

                B.ls_LM_loss_mean[i], B.ls_LM_loss_std[i], B.ls_LM_grad_mean[i], B.ls_LM_grad_std[i],
                B.ls_GN_loss_mean[i], B.ls_GN_loss_std[i], B.ls_GN_grad_mean[i], B.ls_GN_grad_std[i],
                B.ls_GD_loss_mean[i], B.ls_GD_loss_std[i], B.ls_GD_grad_mean[i], B.ls_GD_grad_std[i],
            ])
        end
    end

    mkpath(dirname(path))
    writedlm(path, rows, ',')
end

# ------------------- EXAMPLE USAGE -------------------
N = 10_000
T = 1.0
Δ = 5e-3

snr_db = 10.0
realizations = 10


dict_seed = 20250208
noise_seed = 1

# single initialization size for ALL runs (since we now track traces vs iteration)
ϵ_init = 1e-1

# number of iterations to record (you'll get iter=0..maxiters)
maxiters = 1000

pd_list = [(2, 1), (5, 5)]

out = basin_trace_data_parallel_pd(
    θ_min, θ_max, pd_list,
    k_0, k_1, k_2, k_3;
    N = N, T = T, Δ = Δ,
    snr_db = snr_db,
    realizations = realizations,
    dict_seed = dict_seed,
    noise_seed = noise_seed,
    ϵ_init = ϵ_init,
    maxiters = maxiters,
)

save_trace_csv(
    "results/trace_loss_grad_vp_vs_ls_dB=$(snr_db)_u=$(u)_eps=$(ϵ_init)_K=$(maxiters).csv",
    out,
)