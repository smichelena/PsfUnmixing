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
	using LeastSquaresOptim

	u = 2 # naturalized gaussian kernel

	s_min = 1e-3
	s_max = 1e-2

	θ_min = s_laplace(s_min, u, s_min)
	θ_max = s_laplace(s_max, u, s_min)

	ϵ_min = 1e-4
	ϵ_max = 1e3

	pd_list = [(2, 1), (5, 5)]

	k_0(θ, t) = laplace_0_n(θ, t, u, umin = s_min)
	k_1(θ, t) = laplace_1_n(θ, t, u, umin = s_min)
	k_2(θ, t) = laplace_2_n(θ, t, u, umin = s_min)
	k_3(θ, t) = laplace_3_n(θ, t, u, umin = s_min)
end

@everywhere function residual(θ, η, x_obs, dictionary, xgrid)
	@assert length(dictionary) == length(θ) # should be p
	@assert sum(length.(dictionary)) == length(η) #should be p*d
	@assert length(x_obs) == length(xgrid) # just to be same
	A = multi_block(k_0, θ, dictionary, xgrid) # we don change the xgrid or the kernel
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
		result[row_start:(row_start+rows-1), col_start:(col_start+cols-1)] .=
			mat
		row_start += rows
		col_start += cols
	end

	return result
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

	C = vcat(
		hcat(transpose(∇_θr) * ∇_θr, transpose(∇_θr) * ∇_ηr),
		hcat(transpose(∇_ηr) * ∇_θr, transpose(∇_ηr) * ∇_ηr),
	)

	# diagonal tensor flattened representation
	G1_blocks = Vector{Matrix{Float64}}()
	G2_blocks = Vector{Matrix{Float64}}()
	for (group, θ_i) in zip(dictionary, θ) # theta is univariate
		push!(G1_blocks, single_block(k_1, θ_i, group, xgrid))
		push!(G2_blocks, single_block(k_2, θ_i, group, xgrid))
	end

	# Residual part
	r∇_θr = transpose(kron(I(p), r)) * block_diag_matrices(G2_blocks) * diag_eta
	r∇_ηθr = transpose(block_diag_matrices(G1_blocks)) * kron(I(p), r)

	R = vcat(
		hcat(r∇_θr, transpose(r∇_ηθr)),
		hcat(r∇_ηθr, zeros(d * p, d * p)),
	)

	return C - R
end

@everywhere function random_perturbation(
	θ_star,
	η_star,
	σ_1,
	σ_2,
	ϵ;
	rng = Random.MersenneTwister(),
)
	dθ = randn(rng, length(θ_star))
	dη = randn(rng, length(η_star))
	finsler_norm =
		(σ_2 * norm(η_star, 2) + σ_1) * norm(dθ, 2) + σ_1 * norm(dη, 2)
	dθ ./= finsler_norm
	dη ./= finsler_norm
	return θ_star + ϵ * dθ, η_star + ϵ * dη
end

@everywhere function monte_carlo_extr_spectrum(
	M,
	θ_star,
	η_star,
	σ_1,
	σ_2,
	ϵ;
	trials = 100,
)
	σ_min = Inf
	σ_max = -Inf
	for _ in 1:trials
		θ, η = random_perturbation(θ_star, η_star, σ_1, σ_2, ϵ)
		G = M(θ, η)
		if size(G, 1) == size(G, 2)
			eigvals_ = eigvals(Hermitian(G))
			σ_min = min(σ_min, minimum(eigvals_))
			σ_max = max(σ_max, maximum(eigvals_))
		else
			svals = svdvals(G)
			σ_min = min(σ_min, minimum(svals))
			σ_max = max(σ_max, maximum(svals)) # FIX
		end
	end
	return σ_max, σ_min
end

# ------------------------------------------------------------
# Monte Carlo convergence success using LeastSquaresOptim
# ------------------------------------------------------------
@everywhere function finsler_rho(theta_hat, eta_hat,
	theta_star, eta_star,
	sigma1, sigma2)
	c = (sigma2 * norm(eta_star, 2) + sigma1)
	return c * norm(theta_hat - theta_star, 2) +
		   sigma1 * norm(eta_hat - eta_star, 2)
end

@everywhere function monte_carlo_convergence_success(
	θ_star::Vector{Float64},
	η_star::Vector{Float64},
	ϵ::Float64,
	x_obs::Vector{Float64},
	dictionary,
	xgrid::Vector{Float64},
	σ_1::Float64,
	σ_2::Float64;
	unmixing_tol=1e-3,
	trials::Int = 100,
	maxiters=1000,
)
	success = 0.0
	p = length(θ_star)
	m = length(x_obs)

	for _ in 1:trials
		θ_init, η_init = random_perturbation(θ_star, η_star, σ_1, σ_2, ϵ)
		z0 = vcat(θ_init, η_init)

		function f!(out, z)
			theta = @view z[1:p]
			eta   = @view z[(p+1):end]
			out   .= residual(theta, eta, x_obs, dictionary, xgrid)
			return out
		end

		function j!(J, z)
			theta = @view z[1:p]
			eta   = @view z[(p+1):end]
			J     .= -jacobian(theta, eta, dictionary, xgrid)
			return J
		end

		prob = LeastSquaresProblem(
			x = z0,
			f! = f!,
			g! = j!,
			output_length = m,
			autodiff = :none,
		)
		optimize!(prob, LevenbergMarquardt(), iterations = maxiters, show_trace=false)

		z_hat = prob.x

		θ_hat = copy(@view z_hat[1:p])
		η_hat = copy(@view z_hat[(p+1):end])

		finsler_norm =
			(σ_2 * norm(η_star, 2) + σ_1) * norm(θ_star, 2) +
			σ_1 * norm(η_star, 2)

		unmixing_error = (finsler_rho(θ_hat, η_hat, θ_star, η_star, σ_1, σ_2)/finsler_norm)^2

		pert_size      = (finsler_rho(θ_init, η_init, θ_star, η_star, σ_1, σ_2)/finsler_norm)^2

		println("unmixing error p=$(length(θ_star)), d=$(length(η_star)/length(θ_star)) = $(unmixing_error), success = $(unmixing_error <= unmixing_tol), perturbation size = $(pert_size)")

		success += unmixing_error <= unmixing_tol ? 1.0 : 0.0
	end

	return success / trials
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

# ------------------------------------------------------------
# One noise realization for a FIXED (p,d) dictionary
# ------------------------------------------------------------
@everywhere function one_noisy_realization(
	p::Int, d::Int, r::Int,
	θ_star::Vector{Float64}, η_star::Vector{Float64},
	dictionary, xgrid::Vector{Float64}, ϵ::Vector{Float64},
	x_obs_clean::Vector{Float64},
	σ_1::Float64, σ_2::Float64,
	c_r_0::Float64, c_r_1::Float64, c_r_2::Float64, J_star::Float64, # FIX: name
	snr_db::Float64, noise_seed::Int, mc_trials::Int,
)
	N_mc = length(ϵ)

	rng_noise = MersenneTwister(noise_seed + r + 10_000 * p + d)

	w_dir = randn(rng_noise, length(x_obs_clean))
	w_dir ./= norm(w_dir)

	w_norm = norm(x_obs_clean) * 10.0^(-snr_db / 20.0)
	w = w_norm * w_dir
	x_obs_noisy = x_obs_clean + w

	H_noisy(θ, η) = hessian(θ, η, x_obs_noisy, dictionary, xgrid)

	μmin = minimum(
		eigvals(
			Hermitian(hessian(θ_star, η_star, x_obs_noisy, dictionary, xgrid)),
		),
	)

	c_1 = 2 * J_star + c_r_0 * c_r_1 + w_norm * c_r_2
	c_2 = 1 + c_r_1 * c_r_2

	finsler_norm =
			(σ_2 * norm(η_star, 2) + σ_1) * norm(θ_star, 2) +
			σ_1 * norm(η_star, 2)

	radius = (sqrt(c_1^2 + 4 * c_2 * μmin) - c_1) / (2 * c_2 * finsler_norm)

	min_eigs = zeros(N_mc)
	convergence_success = zeros(N_mc)

	for (i, δ) in enumerate(ϵ)
		_, min_eigs[i] = monte_carlo_extr_spectrum(
			H_noisy,
			θ_star,
			η_star,
			σ_1,
			σ_2,
			δ;
			trials = mc_trials,
		)

		convergence_success[i] = monte_carlo_convergence_success(
			θ_star, η_star, δ,
			x_obs_noisy, dictionary, xgrid,
			σ_1, σ_2;
			trials = mc_trials,
		)
	end

	return (
		μmin = μmin,
		min_eigs = min_eigs,
		success_rate = convergence_success,
		radius = radius,
	)
end

# ------------------------------------------------------------
# Main experiment: progress per (p,d), pmap over realizations
# ------------------------------------------------------------
function basin_data_ranges_noisy_db_parallel_pd(
	θ_min, θ_max, ϵ_min, ϵ_max, pd_list, k_0, k_1, k_2, k_3;
	N = 1000, T = 1.0, Δ = 0.1, N_mc = 100, mc_trials = 1000,
	snr_db::Real = 30.0,
	realizations::Int = 10,
	dict_seed::Int = 1234,      # <--- use SAME dict_seed as noiseless code
	noise_seed::Int = 1,
)
	xgrid = collect(range(-T, T, length = N))
	ϵ = collect(logrange(ϵ_min, ϵ_max, length = N_mc))

	out = Dict{Tuple{Int, Int}, Any}()

	prog = Progress(length(pd_list); desc = "noisy basins per (p,d)")

	for (p, d) in pd_list
		# ----- FIXED dictionary for this (p,d) -----
		rng_dict = MersenneTwister(dict_seed + 10_000 * p + d)

		θ_star = 0.5 * (θ_min + θ_max) * ones(p)
		η_star = ones(p * d)

		dictionary = generate_spike_groups(rng_dict, -T, T, Δ, p, d)
		x_obs_clean = multi_block(k_0, θ_star, dictionary, xgrid) * η_star

		σ_0 = sqrt(p) * lipschitz_constant(k_0, dictionary, xgrid) 
		σ_1 = lipschitz_constant(k_1, dictionary, xgrid)
		σ_2 = lipschitz_constant(k_2, dictionary, xgrid)
		σ_3 = lipschitz_constant(k_3, dictionary, xgrid)

		c_star_functional(θ, group) =
			opnorm(single_block(k_2, θ, group, xgrid)) * norm(η_star, 2) +
			2 * opnorm(single_block(k_1, θ, group, xgrid))

		c_r_0 = maximum(
			c_star_functional(θ_star[i], dictionary[i]) for
			i in eachindex(θ_star)
		)

		c_r_1 = max(σ_0 / (σ_2 * norm(η_star, 2) + σ_1), inv(norm(η_star, 2)))

		c_r_2 = max(
			(σ_3 * norm(η_star, 2) + 2 * σ_2) / (σ_2 * norm(η_star, 2) + σ_1),
			2 * σ_2 / norm(η_star, 2),
		)

		J_star = opnorm(jacobian(θ_star, η_star, dictionary, xgrid))

		finsler_norm =
			(σ_2 * norm(η_star, 2) + σ_1) * norm(θ_star, 2) +
			σ_1 * norm(η_star, 2)

		x = ϵ ./ finsler_norm

		# ----- parallel over realizations (same dictionary) -----
		rs = collect(1:realizations)
		res = pmap(rs) do r
			one_noisy_realization(
				p, d, r,
				Float64.(θ_star), Float64.(η_star),
				dictionary, xgrid, ϵ,
				Float64.(x_obs_clean),
				σ_1, σ_2,
				c_r_0, c_r_1, c_r_2, J_star,
				float(snr_db), noise_seed, mc_trials,
			)
		end

		# stack: R x N_mc
		μmins = [z.μmin for z in res]
		min_eigs_mat = reduce(vcat, (reshape(z.min_eigs, 1, :) for z in res))
		conv_mat = reduce(vcat, (reshape(z.success_rate, 1, :) for z in res))
		radius_vec = [z.radius for z in res]  # scalar per realization

		out[(p, d)] = (
			x = x,
			μ_min_noisy_mean = mean(μmins),
			μ_min_noisy_std = std(μmins),
			min_eigval_noisy_mean = vec(mean(min_eigs_mat, dims = 1)),
			min_eigval_noisy_std = vec(std(min_eigs_mat, dims = 1)),
			conv_noisy_mean = vec(mean(conv_mat, dims = 1)),
			conv_noisy_std = vec(std(conv_mat, dims = 1)),
			radius_noisy_mean = mean(radius_vec),
			radius_noisy_std = std(radius_vec),
		)

		next!(prog)
	end

	return out
end

# ------------------------------------------------------------
# Save to CSV (long format: easiest for later plotting)
# ------------------------------------------------------------
function save_noisy_basins_csv(
	path::AbstractString,
	out::Dict{Tuple{Int, Int}, Any},
)
	rows = Any[]
	push!(
		rows,
		[
			"p", "d", "idx", "x",
			"mu_min_mean", "mu_min_std",
			"min_eigval_mean", "min_eigval_std",
			"conv_mean", "conv_std",
			"radius_mean", "radius_std",
		],
	)

	for ((p, d), B) in sort(collect(out); by = x -> x[1])
		for i in eachindex(B.x)
			push!(
				rows,
				[
					p, d, i, B.x[i],
					B.μ_min_noisy_mean, B.μ_min_noisy_std,
					B.min_eigval_noisy_mean[i], B.min_eigval_noisy_std[i],
					B.conv_noisy_mean[i], B.conv_noisy_std[i],  # FIX
					B.radius_noisy_mean,
					B.radius_noisy_std,
				],
			)
		end
	end

	mkpath(dirname(path))
	writedlm(path, rows, ',')
end

# ------------------- 
N = 10_000
T = 1.0
Δ = 5e-3
N_mc = 20
mc_trials = 100
snr_db = 0.0
realizations = 10

ϵ_min = 1e-4
ϵ_max = 1e5

pd_list = [(2, 1), (2, 5), (5, 1), (5, 5)]

dict_seed = 20250208
noise_seed = 1

out = basin_data_ranges_noisy_db_parallel_pd(
	θ_min, θ_max, ϵ_min, ϵ_max, pd_list,
	k_0, k_1, k_2, k_3;
	N = N, T = T, Δ = Δ, N_mc = N_mc, mc_trials = mc_trials,
	snr_db = snr_db, realizations = realizations,
	dict_seed = dict_seed, noise_seed = noise_seed,
)

save_noisy_basins_csv("results/noisy_convergence_test_dB=$(snr_db)_u=$(u).csv", out)