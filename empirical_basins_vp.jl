# ============================================================
# VP noisy basins (parallel over noise realizations, per (p,d))
# Saves long-format CSV for later plotting
#
# Run with:  julia -p <NWORKERS> vp_noisy_basins_parallel.jl
# ============================================================

using Pkg
Pkg.activate(".")

using Distributed

if nprocs() == 1
	addprocs()
end

# ---------------------- preamble (your setup) ----------------------
@everywhere begin
	using PsfUnmixing
	using Random, LinearAlgebra, Statistics
	using DelimitedFiles
	using ProgressMeter

	u = 2 # naturalized gaussian kernel

	s_min = 1e-3
	s_max = 1e-2

	θ_min = s_laplace(s_min, u, s_min)
	θ_max = s_laplace(s_max, u, s_min)

	k_0(θ, t) = laplace_0_n(θ, t, u, umin = s_min)
	k_1(θ, t) = laplace_1_n(θ, t, u, umin = s_min)
	k_2(θ, t) = laplace_2_n(θ, t, u, umin = s_min)
	k_3(θ, t) = laplace_3_n(θ, t, u, umin = s_min)
end

# ---------------------- helper functions ----------------------
@everywhere function projected_perturbation(
	θ_star,
	ϵ;
	rng = Random.default_rng(),
)
	dθ = rand(rng, length(θ_star))
	return θ_star + ϵ * dθ ./ norm(dθ, 2)
end


@everywhere function projected_envelope(
	ϵ,
	norm_J_star,
	w_norm,
	θ_star,
	η_star,
	k_0,
	x_obs,
	σ_1,
	σ_2,
	dictionary,
	xgrid,
)
	G0_star = multi_block(k_0, θ_star, dictionary, xgrid)
	sigma_min_G_star, sigma_max_G_star = extrema(svdvals(G0_star))

	η_hat = G0_star \ x_obs

	# ||η̂(θ) - η̂(θ*)|| ≤ (σ1 ||η̂(θ*)|| ||G(θ_star)|| / σmin(G*)^2 + ||w|| ) * ||θ-θ*||
	eta_term =
		(σ_1 * norm(η_hat, 2) * sigma_max_G_star) / sigma_min_G_star^2 + w_norm

	# --- envelope terms ---
	t_1 = (σ_2 * norm(η_star, 2) + σ_1) * ϵ + σ_1 * eta_term * ϵ
	# t_2 =  σ_1 * norm(η_star, 2) * ϵ + σ_0 * eta_term * ϵ
	# t_3 = (σ_3 * norm(η_star, 2) + 2σ_2) * ϵ + 2σ_2 * eta_term * ϵ

	# env = 2 * norm_J_star * t_1 + t_1^2 + norm_DJ_star * t_2 + (noise_power + t_2) * t_3
	return 2 * norm_J_star * t_1 + t_1^2
end



# @everywhere function projected_envelope(
# 	ϵ, norm_J_star, norm_DJ_star, noise_power,
# 	θ_star, η_star, k_0, x_obs, σ_0, σ_1, σ_2, σ_3,
# 	dictionary, xgrid; trials = 100,
# )
# 	max_pert = -Inf
# 	for _ in 1:trials
# 		θ_pert = projected_perturbation(θ_star, ϵ)
# 		η_pert = multi_block(k_0, θ_pert, dictionary, xgrid) \ x_obs

# 		t_1 =
# 			(σ_2 * norm(η_star, 2) + σ_1) * norm(θ_star - θ_pert, 2) +
# 			σ_1 * norm(η_star - η_pert, 2)

# 		t_2 =
# 			σ_1 * norm(η_star, 2) * norm(θ_star - θ_pert, 2) +
# 			σ_0 * norm(η_star - η_pert, 2)

# 		t_3 =
# 			(σ_3 * norm(η_star, 2) + 2σ_2) * norm(θ_star - θ_pert, 2) +
# 			2σ_2 * norm(η_star - η_pert, 2)

# 		env = 2 * norm_J_star * t_1 + t_1^2 + norm_DJ_star * t_2  + (noise_power + t_2) * t_3
# 		max_pert = max(max_pert, env)
# 	end
# 	return max_pert
# end

@everywhere function monte_carlo_extr_spectrum_vp(M, θ_star, ϵ; trials = 100)
	σ_min = Inf
	σ_max = -Inf
	for _ in 1:trials
		θ = projected_perturbation(θ_star, ϵ)
		G = M(θ)
		if size(G, 1) == size(G, 2)
			eigvals_ = eigvals(Hermitian(G))
			σ_min = min(σ_min, minimum(eigvals_))
			σ_max = max(σ_max, maximum(eigvals_))
		else
			svdvals_ = svdvals(G)
			σ_min = min(σ_min, minimum(svdvals_))
			σ_max = max(σ_max, maximum(svdvals_))
		end
	end
	return σ_max, σ_min
end

@everywhere function monte_carlo_envelope_vp(M, θ_star, ϵ; trials = 100)
	σ_max = -Inf
	M_star = M(θ_star)
	for _ in 1:trials
		θ = projected_perturbation(θ_star, ϵ)
		E = M_star - M(θ)
		σ_max = max(σ_max, opnorm(E))
	end
	return σ_max
end

# ---- VP residual / Hessian (your code, verbatim except minor spacing) ----
@everywhere function residual_vp(θ, x_obs, dictionary, xgrid)
	@assert length(dictionary) == length(θ)
	@assert length(x_obs) == length(xgrid)

	A = multi_block(k_0, θ, dictionary, xgrid)
	U, = svd(A)

	return x_obs - U * (transpose(U) * x_obs)
end

@everywhere function jacobian(θ, η, dictionary, xgrid)

	@assert length(dictionary) == length(θ) # should be p
	@assert sum(length.(dictionary)) == length(η) #should be p*d

	# once assertions are done we can get sizes
	d = length(dictionary[1])

	G0 = multi_block(k_0, θ, dictionary, xgrid)
	G1 = multi_block(k_1, θ, dictionary, xgrid)
	diag_eta = block_diag(collect(Iterators.partition(η, d)))

	return hcat(G1 * diag_eta, G0)
end

@everywhere function block_diag_matrices(matrices::Vector{<:AbstractMatrix})
	total_rows = sum(size.(matrices, 1))
	total_cols = sum(size.(matrices, 2))
	result = zeros(eltype(first(matrices)), total_rows, total_cols)

	row_start = 1
	col_start = 1
	for mat in matrices
		rows, cols = size(mat)
		result[row_start:(row_start+rows-1), col_start:(col_start+cols-1)] .=
			mat
		row_start += rows
		col_start += cols
	end

	return result
end

@everywhere function residual(θ, η, x_obs, dictionary, xgrid)
	@assert length(dictionary) == length(θ) # should be p
	@assert sum(length.(dictionary)) == length(η) #should be p*d
	@assert length(x_obs) == length(xgrid) # just to be same
	A = multi_block(k_0, θ, dictionary, xgrid) # we don change the xgrid or the kernel
	return x_obs - A * η
end

@everywhere function hessian(θ, η, x_obs, dictionary, xgrid)

	@assert length(dictionary) == length(θ) # should be p
	@assert sum(length.(dictionary)) == length(η) #should be p*d
	@assert length(x_obs) == length(xgrid) # just to be same

	# once assertions are done we can get sizes
	d = length(dictionary[1])
	p = length(θ)

	G0 = multi_block(k_0, θ, dictionary, xgrid)
	G1 = multi_block(k_1, θ, dictionary, xgrid)

	r = residual(θ, η, x_obs, dictionary, xgrid)

	diag_eta = block_diag(collect(Iterators.partition(η, d)))

	# Curvature part
	∇_θr = G1 * diag_eta
	∇_ηr = G0

	C = vcat(hcat(transpose(∇_θr) * ∇_θr, transpose(∇_θr) * ∇_ηr),
		hcat(transpose(∇_ηr) * ∇_θr, transpose(∇_ηr) * ∇_ηr))

	# diagonal tensor flattened representation
	G1 = Vector{Matrix{Float64}}()
	G2 = Vector{Matrix{Float64}}()
	for (group, θ_i) in zip(dictionary, θ) # theta is univariate
		push!(G1, single_block(k_1, θ_i, group, xgrid))
		push!(G2, single_block(k_2, θ_i, group, xgrid))
	end

	# Residual part
	r∇_θr = transpose(kron(I(p), r)) * block_diag_matrices(G2) * diag_eta
	r∇_ηθr = transpose(block_diag_matrices(G1)) * kron(I(p), r)

	R = vcat(hcat(r∇_θr, transpose(r∇_ηθr)), hcat(r∇_ηθr, zeros(d*p, d*p)))

	return C - R
end

@everywhere function hessian_vp(θ, x_obs, dictionary, xgrid)
	@assert length(dictionary) == length(θ)
	@assert length(x_obs) == length(xgrid)

	d = length(dictionary[1])
	p = length(θ)

	G0 = multi_block(k_0, θ, dictionary, xgrid)
	G1 = multi_block(k_1, θ, dictionary, xgrid)

	r = residual_vp(θ, x_obs, dictionary, xgrid)

	η = G0 \ x_obs
	diag_eta = block_diag(collect(Iterators.partition(η, d)))

	∇_θr = G1 * diag_eta
	∇_ηr = G0

	G1_tensor = Vector{Matrix{Float64}}()
	G2 = Vector{Matrix{Float64}}()
	for (group, θ_i) in zip(dictionary, θ)
		push!(G1_tensor, single_block(k_1, θ_i, group, xgrid))
		push!(G2, single_block(k_2, θ_i, group, xgrid))
	end

	r∇_θr = -transpose(kron(I(p), r)) * block_diag_matrices(G2) * diag_eta
	r∇_ηθr = -transpose(block_diag_matrices(G1_tensor)) * kron(I(p), r)

	∇_θL = transpose(∇_θr) * ∇_θr + r∇_θr
	∇_ηθL = transpose(∇_ηr) * ∇_θr + r∇_ηθr

	return ∇_θL - transpose(∇_ηθL) * inv(transpose(G0)*G0 + 1e-8*I) * ∇_ηθL
end

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

# ---------------------- one realization job ----------------------
@everywhere function one_vp_noisy_realization(
	p::Int, d::Int, r::Int,
	θ_star::Vector{Float64}, η_star::Vector{Float64},
	dictionary, xgrid::Vector{Float64}, ϵ::Vector{Float64},
	x::Vector{Float64},
	σ_0::Float64, σ_1::Float64, σ_2::Float64, σ_3::Float64,
	norm_DJ_star::Float64,
	snr_db::Float64, noise_seed::Int, mc_trials::Int,
)
	N_mc = length(ϵ)

	rng_noise = MersenneTwister(noise_seed + r + 10_000*p + d)

	w_dir = randn(rng_noise, length(x))
	w_dir ./= norm(w_dir)

	w_norm = norm(x) * 10.0^(-snr_db/20.0)
	w = w_norm * w_dir
	x_obs= x + w

	snr_realized = 20 * log10(norm(x) / norm(w))

	H_vp_noisy(θ) = hessian_vp(θ, x_obs, dictionary, xgrid)

	# keep your original definition (uses x_obs_clean inside the LS for η)
	H_restricted_noisy(θ) = begin
		η = multi_block(k_0, θ, dictionary, xgrid) \ x_obs
		return hessian(θ, η, x_obs, dictionary, xgrid)
	end

	norm_J_star = opnorm(jacobian(θ_star, η_star, dictionary, xgrid))

	μ_min_vp = minimum(eigvals(Hermitian(H_vp_noisy(θ_star))))

	min_eigs = zeros(N_mc)
	env_vp   = zeros(N_mc)
	env_r    = zeros(N_mc)
	env_proj = zeros(N_mc)

	for (i, δ) in enumerate(ϵ)
		_, min_eigs[i] = monte_carlo_extr_spectrum_vp(H_vp_noisy, θ_star, δ; trials = mc_trials)
		env_vp[i]      = monte_carlo_envelope_vp(H_vp_noisy, θ_star, δ; trials = mc_trials)
		env_r[i]       = monte_carlo_envelope_vp(H_restricted_noisy, θ_star, δ; trials = mc_trials)

		env_proj[i] = projected_envelope(
			δ,
			norm_J_star,
			w_norm,
			θ_star,
			η_star,
			k_0,
			x_obs,
			σ_1,
			σ_2,
			dictionary,
			xgrid,
		)
	end

	return (
		snr = snr_realized,
		wnorm = norm(w),
		μmin = μ_min_vp,
		mineigs = min_eigs,
		env_vp = env_vp,
		env_r = env_r,
		env_proj = env_proj,
	)
end

# ---------------------- main experiment (progress per (p,d)) ----------------------
function basin_data_ranges_vp_noisy_db_parallel(
	θ_min, θ_max, ϵ_min, ϵ_max,
	pd_list;
	N = 1000, T = 1.0, Δ = 0.1, N_mc = 100, mc_trials = 1000,
	snr_db::Real = 30.0,
	realizations::Int = 10,
	dict_seed::Int = 1234,   # must match noiseless dict seed if you want identical dictionaries
	noise_seed::Int = 1,
)
	xgrid = collect(range(-T, T, length = N))
	ϵ = collect(logrange(ϵ_min, ϵ_max, length = N_mc))

	out = Dict{Tuple{Int, Int}, Any}()

	prog = Progress(length(pd_list); desc = "VP noisy basins")

	for (p, d) in pd_list
		θ_star = 0.5 * (θ_min + θ_max) * ones(p)
		η_star = ones(p*d)

		rng_dict = MersenneTwister(dict_seed + 10_000*p + d)
		dictionary = generate_spike_groups(rng_dict, -T, T, Δ, p, d)

		x = multi_block(k_0, θ_star, dictionary, xgrid) * η_star

		σ_0 = sqrt(p) * lipschitz_constant(k_0, dictionary, xgrid)
		σ_1 = lipschitz_constant(k_1, dictionary, xgrid)
		σ_2 = lipschitz_constant(k_2, dictionary, xgrid)
		σ_3 = lipschitz_constant(k_3, dictionary, xgrid)

		norm_DJ_star =
			opnorm(single_block(k_2, θ_star[1], dictionary[1], xgrid)) *
			norm(η_star, 2) +
			opnorm(single_block(k_1, θ_star[1], dictionary[1], xgrid))

		rs = collect(1:realizations)

		res = pmap(rs) do r
			one_vp_noisy_realization(
				p, d, r,
				Float64.(θ_star), Float64.(η_star),
				dictionary, xgrid, ϵ,
				Float64.(x),
				σ_0, σ_1, σ_2, σ_3,
				norm_DJ_star,
				float(snr_db), noise_seed, mc_trials,
			)
		end

		snrs   = [z.snr for z in res]
		wnorms = [z.wnorm for z in res]
		μmins  = [z.μmin for z in res]

		mineigs_mat = reduce(vcat, (reshape(z.mineigs, 1, :) for z in res))
		envvp_mat   = reduce(vcat, (reshape(z.env_vp, 1, :) for z in res))
		envr_mat    = reduce(vcat, (reshape(z.env_r, 1, :) for z in res))
		envp_mat    = reduce(vcat, (reshape(z.env_proj, 1, :) for z in res))

		x = ϵ ./ norm(θ_star, 2)

		lower_vp   = μmins .- envvp_mat
		lower_r    = μmins .- envr_mat
		lower_proj = μmins .- envp_mat

		out[(p, d)] = (
			x                         = x,
			snr_db_mean               = mean(snrs),
			snr_db_std                = std(snrs),
			w_norm_mean               = mean(wnorms),
			w_norm_std                = std(wnorms),
			μ_min_vp_noisy_mean       = mean(μmins),
			μ_min_vp_noisy_std        = std(μmins),
			min_eigval_noisy_mean     = vec(mean(mineigs_mat, dims = 1)),
			min_eigval_noisy_std      = vec(std(mineigs_mat, dims = 1)),
			lower_env_vp_noisy_mean   = vec(mean(lower_vp, dims = 1)),
			lower_env_vp_noisy_std    = vec(std(lower_vp, dims = 1)),
			lower_env_r_noisy_mean    = vec(mean(lower_r, dims = 1)),
			lower_env_r_noisy_std     = vec(std(lower_r, dims = 1)),
			lower_env_proj_noisy_mean = vec(mean(lower_proj, dims = 1)),
			lower_env_proj_noisy_std  = vec(std(lower_proj, dims = 1)),
		)

		next!(prog)
	end

	return out
end

# ---------------------- CSV writer (long format) ----------------------
function save_vp_noisy_basins_csv(
	path::AbstractString,
	out::Dict{Tuple{Int, Int}, Any},
)
	rows = Any[]
	push!(
		rows,
		[
			"p", "d", "idx", "x",
			"snr_db_mean", "snr_db_std",
			"w_norm_mean", "w_norm_std",
			"mu_min_mean", "mu_min_std",
			"mineig_mean", "mineig_std",
			"lower_vp_mean", "lower_vp_std",
			"lower_r_mean", "lower_r_std",
			"lower_proj_mean", "lower_proj_std",
		],
	)

	for ((p, d), B) in sort(collect(out); by = x->x[1])
		for i in eachindex(B.x)
			push!(
				rows,
				[
					p, d, i, B.x[i],
					B.snr_db_mean, B.snr_db_std,
					B.w_norm_mean, B.w_norm_std,
					B.μ_min_vp_noisy_mean, B.μ_min_vp_noisy_std,
					B.min_eigval_noisy_mean[i], B.min_eigval_noisy_std[i],
					B.lower_env_vp_noisy_mean[i], B.lower_env_vp_noisy_std[i],
					B.lower_env_r_noisy_mean[i], B.lower_env_r_noisy_std[i],
					B.lower_env_proj_noisy_mean[i], B.lower_env_proj_noisy_std[i],
				],
			)
		end
	end

	mkpath(dirname(path))
	writedlm(path, rows, ',')
end

# ---------------------- run + save ----------------------
# Set these how you like
N = 1000
T = 1.0
Δ = 5e-3
N_mc = 100
mc_trials = 100
snr_db = Inf
realizations = 1

ϵ_min = 1e-4
ϵ_max = 1e3

pd_list = [(2, 1), (2, 5), (5,1), (5, 5)]

# IMPORTANT: choose dict_seed and reuse it for noiseless runs if you want identical dictionaries
dict_seed  = 20250208
noise_seed = 1

out = basin_data_ranges_vp_noisy_db_parallel(
	θ_min, θ_max, ϵ_min, ϵ_max,
	pd_list;
	N = N, T = T, Δ = Δ, N_mc = N_mc, mc_trials = mc_trials,
	snr_db = snr_db,
	realizations = realizations,
	dict_seed = dict_seed,
	noise_seed = noise_seed,
)

save_vp_noisy_basins_csv("results/noisy_basins_vp_db=$(snr_db).csv", out)

println("Wrote results/noisy_basins_vp.csv")
