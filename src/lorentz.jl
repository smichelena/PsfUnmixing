# Generalized Stark kernel, free geometry

l_0(u, t, p) = @. u^p ./ (u^2 .+ t .^ 2) .^ (p / 2)

l_1(u, t, p) = @. (p * u^(p - 1) * t .^ 2) ./ ((u^2 .+ t .^ 2) .^ (p / 2 + 1))

l_2(u, t, p) =
	@. -(p * t .^ 2 * u^(p - 2) * (3 * u^2 - (p - 1) * t .^ 2)) ./
	(u^2 .+ t .^ 2) .^ ((p + 4) / 2)

# Generalized Stark kernel, natural geometry
using QuadGK

# Compute C_p constant numerically, no explicit formula exists (or at least can be found easily)
function cp_lorentz(p)
	integrand(x) = x^4 / (1 + x^2)^(p + 2)
	val, _ = quadgk(integrand, -Inf, Inf, rtol = 1e-8)
	return val
end

α(p) = 2p * sqrt(cp_lorentz(p))

s_l(u, p, umin) = α(p) * (sqrt(u) - sqrt(umin))

u_l(s, p, umin) = (s / α(p) + sqrt(umin))^2

du_ds(s, p, umin) = 2 * sqrt(u_l(s, p, umin)) / α(p)

du_ds_2(p) = 2 / α(p)^2 #du_ds is linear in s so this doesnt depend on it

l_0_n(s, t, p; umin = 1e-4) = l_0(u_l(s, p, umin), t, p)

l_1_n(s, t, p; umin = 1e-4) = du_ds(s, p, umin) * l_1(u_l(s, p, umin), t, p)

l_2_n(s, t, p; umin = 1e-4) =
	du_ds_2(p) * l_1(u_l(s, p, umin), t, p) +
	du_ds(s, p, umin)^2 * l_2(u_l(s, p, umin), t, p)

