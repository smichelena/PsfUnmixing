@everywhere begin
	using Pkg
	Pkg.activate(".")
	using PsfUnmixing,
		LinearAlgebra, ForwardDiff, Random, DelimitedFiles, Statistics

	p_ = 20
	u_ = 1e-2
	Δ_ = 4e-1
	L = 3
	N = 10001
	grid = range(-1.0, 1.0, N)

	umin = 1e-5
	umax = 2e-1

	k_0 = (u, t) -> laplace_0_n(s_laplace(u, p_, umin), t, p_, umin = umin)

	T = generate_spike_groups((-1.0, 1.0), 2, L, Δ_)

	G_star = multi_block(k_0, [u_, u_], T, grid)
	η_ = ones(6)
	x_ = G_star * η_
	η_star = η_ ./ norm(x_)
	x_star = G_star * η_star

	θ_star = [u_, u_]

	function residual_projected(θ, x, T, k_0, grid)
		G = multi_block(k_0, θ, T, grid)
		return x - G * (G \ x) 
	end

	residual = θ -> residual_projected(θ, x_star, T, k_0, grid)
	J = ForwardDiff.jacobian(residual, θ_star)
	trace_inv_J = sum(eigvals(Hermitian(transpose(J) * J)) .^ (-1))
end

@everywhere function MSE_bias_projected(
	snr,
	x_star,
	θ_star,
	k_0,
	T,
	grid;
	ϵ_init = 1e-5,
	trials = 50,
	solver_iters = 100,
)
	θ_hats = []

	σ = sqrt(mean(abs2, x_star) / (10.0^(snr / 10.0)))

	for _ in 1:trials
		x_noisy =
			x_star .+ σ * randn(length(x_star))
		f(θ) = loss_projected(θ, x_noisy, T, k_0, grid)
		θ_init = random_perturbation_projected(θ_star, ϵ_init)
		θ_hat =
			gradient_descent_projected(f, θ_init, solver_iters = solver_iters)
		push!(θ_hats, θ_hat)
	end

	return mean(abs2, norm(θ - θ_star) for θ in θ_hats)
end

# Sweep parameters
ϵ_init = 1e-3
snr_samples = 20
trials = 200
solver_iters = 100
snr_range = range(-30, 10, snr_samples)

crb =
	snr_range .|>
	snr -> (mean(abs2, x_star) / (10.0^(snr / 10.0))) * trace_inv_J

# Parallel run — returns tuple for each SNR
results = pmap(
	snr -> MSE_bias_projected(
		snr,
		x_star,
		θ_star,
		k_0,
		T,
		grid,
		ϵ_init = ϵ_init,
		trials = trials,
		solver_iters = solver_iters,
	),
	snr_range,
)

mse_vals = results

writedlm("results/tsp_mse_projected_u=$(p_).txt", (snr_range, crb, mse_vals))
