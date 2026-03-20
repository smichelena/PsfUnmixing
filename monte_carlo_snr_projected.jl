using Distributed
# No addprocs() – control processes with `-p` or scheduler

@everywhere begin
	using Pkg
	Pkg.activate(".")
	using PsfUnmixing, LinearAlgebra, Random, DelimitedFiles

	umin = 1e-5
	umax = 2e-1
	p = 1

	θ_star = [1e-2, 1e-2]
	Δ_star = 4e-1
	L = 3
	N = 10001
	grid = range(-1.0, 1.0, N)

	k_0 = (θ, t) -> laplace_0_n(s_laplace(θ, p, umin), t, p, umin = umin)

	T = generate_spike_groups((-1.0, 1.0), 2, L, Δ_star)

	G_star = multi_block(k_0, θ_star, T, grid)
	η_ = ones(6)
	x_ = G_star * η_
	η_star = η_ ./ norm(x_)
	x_star = G_star * η_star
end

@everywhere function run_trial(ϵ_perb, x_star, θ_star, snr_, k_0, T, grid;
	trials = 1, solver_iters = 1)
	monte_carlo_projected(
		x_star,
		θ_star,
		snr_,
		ϵ_perb,
		k_0,
		T,
		grid;
		trials = trials,
		solver_iters = solver_iters,
	)
end

ϵ_range = logrange(1e-5, 1e0, 50)
snr = 10.0

trials = 1000
solver_iters = 100

# Parallel execution
success_rate_p = pmap(
	ϵ -> run_trial(
		ϵ,
		x_star,
		θ_star,
		snr,
		k_0,
		T,
		grid,
		trials = trials,
		solver_iters = solver_iters,
	),
	ϵ_range)

# Save results
writedlm("results/tsp_monte_carlo_projected_snr_$(snr)_p_$(p).txt", success_rate_p)
