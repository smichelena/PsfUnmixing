using Optim, Statistics, Base.Threads

function random_perturbation(u_star, θ_dim, ε)
	n = length(u_star)
	θ_star = @view u_star[1:θ_dim]
	η_star = @view u_star[θ_dim+1:end]

	# Split perturbation between θ and η
	r = rand()
	δθ = r * ε
	δη = (1 - r) * ε

	θ_rand = 2 * rand(length(θ_star)) .- 1.0
	η_rand = 2 * rand(length(η_star)) .- 1.0

	θ_rand *= norm(θ_rand, Inf)^(-1)
	η_rand *= norm(η_rand, Inf)^(-1)

	θ₀ = max.(θ_star .+ δθ .* θ_rand, 1.5e-5)
	η₀ = η_star .+ δη .* η_rand

	return vcat(θ₀, η₀)
end

function gradient_descent_unprojected(f, grad!, u0; solver_iters = 1000)
	n = length(u0)

	lower = vcat([1e-5, 1e-5], fill(-Inf, n - 2))
	upper = fill(Inf, n)

	opt = Optim.Options(
		iterations = solver_iters,
		show_trace = false,
		store_trace = false,
		outer_iterations = 1,
		allow_f_increases = true,  # Optional: avoids line search rejecting bad steps
	)

	result = optimize(
		f,
		grad!,
		lower,
		upper,
		u0,
		Fminbox(GradientDescent()),
		opt,
	)

	return Optim.minimizer(result)
end

function block_diag(vectors::Vector{<:AbstractVector})
	total_rows = sum(length.(vectors))
	total_cols = length(vectors)
	result = zeros(eltype(first(vectors)), total_rows, total_cols)

	row_start = 1
	for (j, vec) in enumerate(vectors)
		len = length(vec)
		result[row_start:row_start+len-1, j] = vec
		row_start += len
	end

	return result
end

function run_solver_unprojected(
	x,
	T,
	grid,
	k_0,
	k_1,
	u_init;
	solver_iters = 1000,
)
	loss(u) = begin
		θ = u[1:2]
		η = u[3:end]
		G = multi_block(k_0, θ, T, grid)
		r = G * η - x
		return 0.5 * length(x)^(-1) * sum(abs2, r)
	end

	grad!(D, u) = begin
		θ = u[1:2]
		η = u[3:end]
		block_eta = block_diag([η[1:3], η[4:end]])
		G_0 = multi_block(k_0, θ, T, grid)
		G_1 = multi_block(k_1, θ, T, grid)
		r = G_0 * η - x
		copyto!(
			D,
			length(x)^(-1) *
			vcat(transpose(G_1 * block_eta) * r, transpose(G_0) * r),
		)
		return nothing
	end

	return gradient_descent_unprojected(
		loss,
		grad!,
		u_init;
		solver_iters = solver_iters,
	)
end

function add_noise_snr(x::AbstractVector, snr_db::Real)
	signal_power = mean(abs2, x)
	snr_linear = 10.0^(snr_db / 10)
	noise_power = signal_power / snr_linear
	noise = sqrt(noise_power) * randn(length(x))
	return x .+ noise
end

function monte_carlo_unprojected(
	x_star,
	u_star,
	snr,
	T,
	grid,
	k_0,
	k_1;
	ϵ = 1.0,
	trials = 10,
	solver_iters = 1000,
	tol = 1e-3,
)
	successes = 0
	n = length(u_star)

	for _ in 1:trials
		x_noisy = add_noise_snr(x_star, snr)

		u_init = random_perturbation(u_star, 2, ϵ)

		u_hat = run_solver_unprojected(
			x_noisy,
			T,
			grid,
			k_0,
			k_1,
			u_init,
			solver_iters = solver_iters,
		)

		if all(isapprox.(u_hat, u_star, atol = tol))
			successes += 1
		end
	end

	return successes / trials
end

function monte_carlo_RMSE_unprojected(
	x_star,
	u_star,
	snr,
	T,
	grid,
	k_0,
	k_1;
	ϵ = 1.0,
	trials = 10,
	solver_iters = 1000,
)
	successes = 0
	n = length(u_star)

	for _ in 1:trials
		x_noisy = add_noise_snr(x_star, snr)

		u_init = random_perturbation(u_star, 2, ϵ)

		u_hat = run_solver_unprojected(
			x_noisy,
			T,
			grid,
			k_0,
			k_1,
			u_init,
			solver_iters = solver_iters,
		)

		sq_errors += sum((u_hat .- u_star).^2) / length(u_star)  
    end

    return sqrt(sq_errors / trials)  
end

function gradient_descent_projected(f, u_init; solver_iters = 1000)
	n = length(u_init)

	lower = 1e-5 * ones(n)
	upper = fill(Inf, n)

	opt = Optim.Options(
		iterations = solver_iters,
		show_trace = false,
		store_trace = false,
		outer_iterations = 1,
	)

	result = optimize(
		f,
		lower,
		upper,
		u_init,
		Fminbox(GradientDescent()),
		opt,
	)

	return Optim.minimizer(result)
end

function random_perturbation_projected(u_star, ε)
	n = length(u_star)

	u_rand = 2 * rand(n) .- 1

	u_rand *= ε * norm(u_rand, Inf)^(-1)

	return max.(u_star + u_rand, 1.5e-5)
end

function loss_projected(u, x, T, k_0, grid)
	G = multi_block(k_0, u, T, grid)
	x_tilde = G * (G \ x)
	return 0.5 * (length(grid))^(-1) * norm(x_tilde - x)^2
end

function monte_carlo_projected(
	x_star,
	u_star,
	snr,
	ϵ,
	k_0,
	T,
	grid;
	trials = 10,
	solver_iters = 1000,
	tol = 1e-3,
)

	successes = 0

	for _ in 1:trials
		x_noisy = add_noise_snr(x_star, snr)
		f(u) = loss_projected(u, x_noisy, T, k_0, grid)

		u_init = random_perturbation_projected(u_star, ϵ)

		u_hat =
			gradient_descent_projected(f, u_init, solver_iters = solver_iters)

		if all(isapprox.(u_hat, u_star, atol = tol))
			successes += 1
		end
	end

	return successes / trials

end

function monte_carlo_RMSE_projected(
    x_star,
    u_star,
    snr,
    ϵ,
    k_0,
    T,
    grid;
    trials = 10,
    solver_iters = 1000
)

    sq_errors = 0.0

    for _ in 1:trials
        x_noisy = add_noise_snr(x_star, snr)
        f(u) = loss_projected(u, x_noisy, T, k_0, grid)

        u_init = random_perturbation_projected(u_star, ϵ)

        u_hat = gradient_descent_projected(f, u_init, solver_iters = solver_iters)

        sq_errors += sum((u_hat .- u_star).^2) / length(u_star)  # MSE
    end

    return sqrt(sq_errors / trials)  # RMSE
end