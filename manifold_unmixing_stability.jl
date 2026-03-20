using Pkg
Pkg.activate(".")

using Distributed

if nprocs() == 1
	addprocs()
end

@everywhere begin
	using PsfUnmixing,
		LinearAlgebra, Random, Statistics, DelimitedFiles, LeastSquaresOptim

	u = 2
	s_min = 1e-3
	s_max = 1e-2

	theta_min = s_laplace(s_min, u, s_min)
	theta_max = s_laplace(s_max, u, s_min)

	k_0(theta, t) = laplace_0_n(theta, t, u, umin = s_min)
	k_1(theta, t) = laplace_1_n(theta, t, u, umin = s_min)
	k_2(theta, t) = laplace_2_n(theta, t, u, umin = s_min)
	k_3(theta, t) = laplace_3_n(theta, t, u, umin = s_min)
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

@everywhere function lipschitz_constant(k, dictionary, xgrid)
	theta_range = range(theta_min, theta_max, length = 100)
	opnorms = zeros(Float64, length(theta_range), length(dictionary))
	for (i, theta) in enumerate(theta_range)
		for (j, group) in enumerate(dictionary)
			opnorms[i, j] = opnorm(single_block(k, theta, group, xgrid))
		end
	end
	return maximum(opnorms)
end

@everywhere function residual(theta, eta, x_obs, dictionary, xgrid)
	A = multi_block(k_0, theta, dictionary, xgrid)
	return x_obs - A * eta
end

@everywhere function jacobian(theta, eta, dictionary, xgrid)
	d = length(dictionary[1])
	G0 = multi_block(k_0, theta, dictionary, xgrid)
	G1 = multi_block(k_1, theta, dictionary, xgrid)
	diag_eta = block_diag(collect(Iterators.partition(eta, d)))
	return hcat(G1 * diag_eta, G0)
end

@everywhere function full_hessian(θ, η, x_obs, dictionary, xgrid)

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

@everywhere function residual_vp(theta, x_obs, dictionary, xgrid)
	A = multi_block(k_0, theta, dictionary, xgrid)
	U, = svd(A)
	return x_obs - U * (transpose(U) * x_obs)
end

@everywhere function jacobian_vp(theta, x_obs, dictionary, xgrid)
	d = length(dictionary[1])
	G0 = multi_block(k_0, theta, dictionary, xgrid)
	G1 = multi_block(k_1, theta, dictionary, xgrid)

	eta = G0 \ x_obs
	diag_eta = block_diag(collect(Iterators.partition(eta, d)))

	grad_theta_r = G1 * diag_eta
	U, = svd(G0)

	return -grad_theta_r + U * transpose(U) * grad_theta_r
end

@everywhere function hessian_vp(θ, x_obs, dictionary, xgrid)

	@assert length(dictionary) == length(θ) # should be p
	@assert length(x_obs) == length(xgrid) # just to be same

	# once assertions are done we can get sizes
	d = length(dictionary[1])
	p = length(θ)

	G0 = multi_block(k_0, θ, dictionary, xgrid)
	G1 = multi_block(k_1, θ, dictionary, xgrid)

	r = residual_vp(θ, x_obs, dictionary, xgrid)

	η = G0 \ x_obs

	diag_eta = block_diag(collect(Iterators.partition(η, d)))

	# Curvature part
	∇_θr = G1 * diag_eta
	∇_ηr = G0

	# diagonal tensor flattened representation
	G1_tensor = Vector{Matrix{Float64}}()
	G2 = Vector{Matrix{Float64}}()
	for (group, θ_i) in zip(dictionary, θ) # theta is univariate
		push!(G1_tensor, single_block(k_1, θ_i, group, xgrid))
		push!(G2, single_block(k_2, θ_i, group, xgrid))
	end

	# Residual part
	r∇_θr = - transpose(kron(I(p), r)) * block_diag_matrices(G2) * diag_eta
	r∇_ηθr = - transpose(block_diag_matrices(G1_tensor)) * kron(I(p), r)

	∇_θL = transpose(∇_θr) * ∇_θr + r∇_θr
	∇_ηθL = transpose(∇_ηr) * ∇_θr + r∇_ηθr

	return ∇_θL - transpose(∇_ηθL) * inv(transpose(G0)*G0 + 1e-8*I) * ∇_ηθL
end

@everywhere function finsler_rho(theta_hat, eta_hat,
	theta_star, eta_star,
	sigma1, sigma2)
	c = (sigma2 * norm(eta_star, 2) + sigma1)
	return c * norm(theta_hat - theta_star, 2) +
		   sigma1 * norm(eta_hat - eta_star, 2)
end

@everywhere function snr_to_sigma(x_clean, snr_db)
	N = length(x_clean)
	snr_lin = 10.0^(snr_db / 10.0)
	return norm(x_clean, 2) / sqrt(N * snr_lin)
end

@everywhere function lm_full(
	x_obs,
	dictionary,
	xgrid,
	theta0,
	eta0;
	maxiters = 100,
)
	p = length(theta0)

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

	z0 = vcat(theta0, eta0)
	m  = length(x_obs)

    # println("z0 = $(z0)")

	prob = LeastSquaresProblem(
		x = z0,
		f! = f!,
		g! = j!,
		output_length = m,
		autodiff = :none,
	)
	optimize!(prob, LevenbergMarquardt(), iterations = maxiters, show_trace=false)

	z_hat = prob.x

    # println("z_hat after run  = $(z_hat)")

    # function f(z)
	# 	theta = @view z[1:p]
	# 	eta   = @view z[(p+1):end]
	# 	return residual(theta, eta, x_obs, dictionary, xgrid)
	# end

    # result = optimize(f, z0, LevenbergMarquardt(), iterations = maxiters)

    # z_hat = result.minimizer

	theta_hat = copy(@view z_hat[1:p])
	eta_hat   = copy(@view z_hat[(p+1):end])
	return theta_hat, eta_hat
end

@everywhere function lm_vp(x_obs, dictionary, xgrid, theta0; maxiters)
	function f!(out, theta)
		out .= residual_vp(theta, x_obs, dictionary, xgrid)
		return out
	end

	function j!(J, theta)
		J .= jacobian_vp(theta, x_obs, dictionary, xgrid)
		return J
	end

	m = length(residual_vp(theta0, x_obs, dictionary, xgrid))
	prob = LeastSquaresProblem(
		x = theta0,
		f! = f!,
		g! = j!,
		output_length = m,
		autodiff = :none,
	)

	optimize!(prob, LevenbergMarquardt(), iterations = maxiters)     # in-place

	theta_hat = copy(theta0)                         # <-- robust
	A = multi_block(k_0, theta_hat, dictionary, xgrid)
	eta_hat = A \ x_obs
	return theta_hat, eta_hat
end


# @everywhere function theory_bound(theta_star, eta_star,
# 	dictionary, xgrid, sigma_1, sigma_2, sigma_noise, w)

# 	J_star = jacobian(theta_star, eta_star, dictionary, xgrid)

# 	sigma_min_J, sigma_max_J = extrema(svdvals(J_star))

# 	N = length(xgrid)

#     c = (sigma_2 * norm(eta_star, 2) + sigma_1)

#     norm_J_w = norm(transpose(J_star) * w, 2)

#     opnorm_estimate = sigma_max_J * norm(w, 2)

#     println("norm_J_w = $(norm_J_w), opnorm_estimate = $(opnorm_estimate), sigma_min_J = $(sigma_min_J)")

#     println("norm(w) = $(norm(w ,2)), sqrt(sigma^2 * N) = $(sqrt(sigma_noise^2 * N))")

# 	return c^2 * norm_J_w^2 / sigma_min_J^4

# end

@everywhere function theory_bound(theta_star, eta_star,
    dictionary, xgrid, sigma_1, sigma_2, sigma_noise)

    J_star = jacobian(theta_star, eta_star, dictionary, xgrid)
    svals = svdvals(J_star)
    sigma_min_J, sigma_max_J = extrema(svals)

    println("kappa = $(sigma_max_J/sigma_min_J)")

    c = (sigma_2 * norm(eta_star, 2) + sigma_1)

    return c^2 * (sigma_noise^2 * norm(J_star)^2) / sigma_min_J^4
end

@everywhere function theory_bound_vp(theta_star, eta_star,
    x_obs, dictionary, xgrid, sigma_1, sigma_2, sigma_noise)

    # --- VP Jacobian at theta_star ---
    J_vp_star = jacobian_vp(theta_star, x_obs, dictionary, xgrid)
    svalsJ = svdvals(J_vp_star)
    sigma_min_J, sigma_max_J = extrema(svalsJ)

    # println("kappa_vp(J) = $(sigma_max_J / sigma_min_J)")

    # --- Gauss–Newton VP Hessian approx: H_vp ≈ JᵀJ ---
    # => ||Δθ|| ≲ ||J|| / sigma_min(J)^2 * ||w||
    theta_err_bound = (norm(J_vp_star) / sigma_min_J^2) * sigma_noise

    # --- Linear subproblem conditioning at theta_star ---
    G0_star = multi_block(k_0, theta_star, dictionary, xgrid)
    svalsG = svdvals(G0_star)
    sigma_min_G, sigma_max_G = extrema(svalsG)

    # println("cond_G(theta*) = $(sigma_max_G / sigma_min_G)")

    # --- Lifted eta error bound ---
    # Decomposition:  η̂(θ̂) - η* = [η̂(θ̂) - η̂(θ*)] + [η̂(θ*) - η*]
    #
    # (i) pure noise at θ*:   ||η̂(θ*) - η*|| ≤ ||G(θ*)†|| ||w|| = (1/sigma_min_G) ||w||
    # (ii) propagation through θ:
    #      use local Lipschitz: ||G(θ̂)-G(θ*)|| ≤ sigma_1 ||θ̂-θ*||
    #      and stability of LS in η: ||η̂(θ̂)-η̂(θ*)|| ≤ ||G(θ*)†|| ||G(θ̂)-G(θ*)|| ||η*||
    #
    eta_err_bound = (sigma_noise / sigma_min_G) +
                    ((sigma_1 * norm(eta_star, 2)) / sigma_min_G) * theta_err_bound

    # --- Full metric rho bound ---
    c_theta = (sigma_2 * norm(eta_star, 2) + sigma_1)
    rho_bound = c_theta * theta_err_bound + sigma_1 * eta_err_bound

    return rho_bound^2
end

@everywhere function one_job(
    p, d, snr_db;
    N=1000, T=1.0, Delta=0.1,
    dict_seed::Int = 0,
    seed::Int = 0,
)
    rng_dict = MersenneTwister(dict_seed)
    rng      = MersenneTwister(seed)

    xgrid = range(-T, T, length=N)

    theta_star = 0.5 * (theta_min + theta_max) * ones(p)
    eta_star   = ones(p * d)

    dictionary = generate_spike_groups(rng_dict, -T, T, Delta, p, d)

    sigma1 = lipschitz_constant(k_1, dictionary, xgrid)
    sigma2 = lipschitz_constant(k_2, dictionary, xgrid)

    norm_factor =
        (sigma2 * norm(eta_star, 2) + sigma1) * norm(theta_star, 2) +
        sigma1 * norm(eta_star, 2)

    Astar   = multi_block(k_0, theta_star, dictionary, xgrid)
    x_clean = Astar * eta_star

    sigma_noise = snr_to_sigma(x_clean, snr_db)
    w      = sigma_noise .* randn(rng, length(x_clean))
    x_obs  = x_clean + w

    theta0 = theta_min .+ (theta_max - theta_min) .* rand(rng, p)
    eta_norm = norm(eta_star, 2)
    eta0 = eta_star .+ 0.1 * eta_norm * randn(rng, p * d) / sqrt(p * d)

    theta_hat_ls, eta_hat_ls =
        lm_full(x_obs, dictionary, xgrid, theta0, eta0; maxiters=1000)

    theta_hat_vp, eta_hat_vp =
        lm_vp(x_obs, dictionary, xgrid, theta0; maxiters=1000)

    rho_ls = finsler_rho(theta_hat_ls, eta_hat_ls, theta_star, eta_star, sigma1, sigma2)^2
    rho_vp = finsler_rho(theta_hat_vp, eta_hat_vp, theta_star, eta_star, sigma1, sigma2)^2

    r_ls  = residual(theta_hat_ls, eta_hat_ls, x_obs, dictionary, xgrid)
    obj_ls = 0.5 * dot(r_ls, r_ls)

    r_vp  = residual_vp(theta_hat_vp, x_obs, dictionary, xgrid)
    obj_vp = 0.5 * dot(r_vp, r_vp)

    theory    = theory_bound(theta_star, eta_star, dictionary, xgrid, sigma1, sigma2, sigma_noise)
    theory_vp = theory_bound_vp(theta_star, eta_star, x_obs, dictionary, xgrid, sigma1, sigma2, sigma_noise)

    return (p=p, d=d, snr_db=snr_db, norm_factor=norm_factor,
            theory=theory, theory_vp=theory_vp,
            rho_ls=rho_ls, rho_vp=rho_vp, obj_ls=obj_ls, obj_vp=obj_vp)
end

function run_unmixing_stability(;
    pd_pairs::Vector{Tuple{Int,Int}},
    snr_db_range = 0:5:30,
    realizations = 50,
    N = 1000,
    T = 1.0,
    Delta = 0.1,
    seed = 1234,
)
    jobs = Tuple{Int,Int,Float64,Int,Int}[]  # (p,d,snr_db,dict_seed,seed)
    k = 0
    for (p,d) in pd_pairs
        dict_seed = seed + 1_000_000*p + 10_000*d  # fixed per (p,d)
        for snr_db in snr_db_range, rep in 1:realizations
            k += 1
            push!(jobs, (p, d, float(snr_db), dict_seed, seed + k))
        end
    end

    results = pmap(jobs) do (p, d, snr_db, dict_seed, s)
        one_job(p, d, snr_db; N=N, T=T, Delta=Delta, dict_seed=dict_seed, seed=s)
    end

    buckets = Dict{Tuple{Int,Int,Float64}, Any}()

    for r in results
        key = (r.p, r.d, r.snr_db)
        if !haskey(buckets, key)
            buckets[key] = (
                theory      = Float64[],
                theory_vp   = Float64[],
                rho_ls      = Float64[],
                rho_vp      = Float64[],
                obj_ls      = Float64[],
                obj_vp      = Float64[],
                norm_factor = Float64[],
            )
        end
        push!(buckets[key].theory,      r.theory)
        push!(buckets[key].theory_vp,   r.theory_vp)
        push!(buckets[key].rho_ls,      r.rho_ls)
        push!(buckets[key].rho_vp,      r.rho_vp)
        push!(buckets[key].obj_ls,      r.obj_ls)
        push!(buckets[key].obj_vp,      r.obj_vp)
        push!(buckets[key].norm_factor, r.norm_factor)
    end

    summary = Dict{Tuple{Int,Int,Float64}, NamedTuple}()

    for (key, v) in buckets
        nf = mean(v.norm_factor)

        mean_theory    = mean(v.theory)
        mean_theory_vp = mean(v.theory_vp)
        mean_rho_ls    = mean(v.rho_ls)
        mean_rho_vp    = mean(v.rho_vp)

        summary[key] = (
            theory      = mean_theory,
            theory_vp   = mean_theory_vp,
            mean_rho_ls = mean_rho_ls,
            mean_rho_vp = mean_rho_vp,
            mean_obj_ls = mean(v.obj_ls),
            mean_obj_vp = mean(v.obj_vp),
            norm_factor = nf,

            theory_rel      = mean_theory / (nf^2),
            theory_rel_vp   = mean_theory_vp / (nf^2),
            mean_rel_rho_ls = mean_rho_ls / (nf^2),
            mean_rel_rho_vp = mean_rho_vp / (nf^2),
        )
    end

    mkpath("results")
    rows = Any[]
    push!(rows, [
        "p","d","snr_db",
        "theory","theory_vp",
        "mean_rho_ls","mean_rho_vp",
        "mean_obj_ls","mean_obj_vp",
        "norm_factor",
        "theory_rel","theory_rel_vp",
        "mean_rel_rho_ls","mean_rel_rho_vp",
    ])

    for (key, s) in sort(collect(summary); by=x->x[1])
        (p, d, snr_db) = key
        push!(rows, [
            p, d, snr_db,
            s.theory, s.theory_vp,
            s.mean_rho_ls, s.mean_rho_vp,
            s.mean_obj_ls, s.mean_obj_vp,
            s.norm_factor,
            s.theory_rel, s.theory_rel_vp,
            s.mean_rel_rho_ls, s.mean_rel_rho_vp,
        ])
    end

    writedlm("results/unmixing_stability.csv", rows, ',')
    return summary
end

out = run_unmixing_stability(
	pd_pairs = [(2, 1), (5,5)],
	snr_db_range = -10:1:20,
	realizations = 100,
	N = 10000,
	Delta = 5e-3,
)
