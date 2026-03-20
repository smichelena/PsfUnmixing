using Random
using Base.Threads: @threads


function single_block(k, theta, locations, grid)
    return hcat([k(theta, grid .- t_l) for t_l in locations]...)
end

function multi_block(k, theta_vec, locations_vec, grid)
    blocks = [single_block(k, θ, locs, grid) for (θ, locs) in zip(theta_vec, locations_vec)]
    return hcat(blocks...)
end

using Random

function sample_from_intervals(
	rng::AbstractRNG,
	intervals::Vector{Tuple{Float64, Float64}},
)
	lengths = map(iv -> max(iv[2] - iv[1], 0.0), intervals)
	total = sum(lengths)
	total <= 0 && error("No available space left to sample from.")

	u = rand(rng) * total
	acc = 0.0
	for (iv, len) in zip(intervals, lengths)
		if u <= acc + len
			return iv[1] + (u - acc)
		end
		acc += len
	end
	return intervals[end][2]  # numerical fallback
end

function subtract_interval(
	intervals::Vector{Tuple{Float64, Float64}},
	L::Float64,
	R::Float64,
)
	L > R && return intervals
	out = Tuple{Float64, Float64}[]
	for (a, b) in intervals
		if R <= a || L >= b
			push!(out, (a, b))
		else
			if a < L
				push!(out, (a, min(L, b)))
			end
			if R < b
				push!(out, (max(R, a), b))
			end
		end
	end
	filter!(iv -> iv[2] - iv[1] > 0.0, out)
	return out
end

# --- main sampler: force one pair at distance Δ ---

"""
Generate N spikes in [a,b] with minimum pairwise distance ≥ Δ,
AND force (t0,t1) to be exactly Δ apart.

Algorithm:
- pick t0 uniform in [a,b]
- pick t1 = t0 ± Δ uniformly among feasible signs
- for all spikes, enforce min separation via exclusion radius Δ/2
- sample the remaining spikes uniformly from remaining allowed set
"""
function place_spikes(rng::AbstractRNG, a::Float64, b::Float64, Δ::Float64,
	N::Int;
	max_tries_for_pair::Int = 100_000)
	a < b || error("Require a < b.")
	Δ > 0 || error("Require Δ > 0 for a forced-distance pair.")
	N ≥ 2 || error("Need N ≥ 2 to force a pair.")

	# Quick feasibility check for just placing two points at distance Δ
	(b - a) < Δ && error(
		"Impossible: interval length (b-a) < Δ, cannot place two spikes Δ apart.",
	)

	r = Δ

	# Step 1-2: sample (t0,t1) with t1=t0±Δ, resampling t0 if needed
	t0 = 0.0
	t1 = 0.0
	for _ in 1:max_tries_for_pair
		t0 = a + rand(rng) * (b - a)

		candidates = Float64[]
		(a <= t0 - Δ <= b) && push!(candidates, t0 - Δ)
		(a <= t0 + Δ <= b) && push!(candidates, t0 + Δ)

		if !isempty(candidates)
			t1 = candidates[rand(rng, 1:length(candidates))]
			break
		end
	end
	(t1 == 0.0 && t0 == 0.0) &&
		error("Failed to sample a feasible forced pair; unexpected.")

	# Now enforce min-separation constraints by removing exclusion windows for t0 and t1
	allowed = [(a, b)]
	allowed = subtract_interval(allowed, max(a, t0 - r), min(b, t0 + r))
	allowed = subtract_interval(allowed, max(a, t1 - r), min(b, t1 + r))

	spikes = Float64[t0, t1]

	# Step 3+: sample remaining spikes uniformly from allowed set, excluding Δ/2 neighborhoods
	for k in 3:N
		isempty(allowed) && error(
			"Ran out of space before placing all spikes. Reduce N or Δ, or widen [a,b].",
		)
		t = sample_from_intervals(rng, allowed)
		push!(spikes, t)
		allowed = subtract_interval(allowed, max(a, t - r), min(b, t + r))
	end

	return spikes
end


function assign_to_groups(
	rng::AbstractRNG,
	spikes::Vector{Float64},
	p::Int,
	d::Int;
	shuffle_groups::Bool = true,
)
	length(spikes) == p*d || error(
		"Need exactly p*d spikes to assign (got $(length(spikes)), expected $(p*d)).",
	)
	perm = randperm(rng, length(spikes))
	s = spikes[perm]
	groups = [s[((i-1)*d+1):(i*d)] for i in 1:p]
	if shuffle_groups
		groups = groups[randperm(rng, p)]
	end
	return groups
end

function generate_spike_groups(
	rng::AbstractRNG,
	a::Real,
	b::Real,
	Δ::Real,
	p::Int,
	d::Int,
)
	N = p*d
	spikes = place_spikes(rng, float(a), float(b), float(Δ), N)
	groups = assign_to_groups(rng, spikes, p, d)
	return groups
end

"""
	min_separation(groups::AbstractVector{<:AbstractVector})

Compute the minimum pairwise separation across *all* spikes contained in a
vector-of-vectors `groups` (e.g. what `generate_spike_groups` returns).

Returns:
- `Inf` if there are 0 or 1 total spikes
- otherwise the minimum absolute difference between any two distinct spikes
"""
function min_separation(groups::AbstractVector{<:AbstractVector})
	all = Float64[]
	for g in groups
		append!(all, Float64.(g))
	end

	n = length(all)
	n ≤ 1 && return Inf

	sort!(all)
	mind = Inf
	@inbounds for i in 2:n
		d = all[i] - all[i-1]  # sorted => nonnegative
		if d < mind
			mind = d
		end
	end
	return mind
end