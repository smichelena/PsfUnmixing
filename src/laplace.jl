function laplace_0(u, t, p)
	return @. exp(- (abs(t) / u) ^ p)
end

function laplace_1(u, t, p)
	return @. p * u^(-p - 1) * (abs(t)^p) * exp(-(abs(t) ./ u)^p)
end

function laplace_2(u, t, p)
	return @. -(
		p * abs(t)^p * ((p + 1) * u^p - p * abs(t)^p) * laplace_0(u, t, p)
	) / u^(2p + 2)

end

function laplace_3(u, t, p)
	return @. p * abs(t)^p * u^(-3 * p - 3) *
			  (
				  (p^2 + 3 * p + 2) * u^(2 * p) +
				  (-3 * p^2 - 3 * p) * abs(t)^p * u^p + p^2 * abs(t)^(2 * p)
			  ) * laplace_0(u, t, p)
end

# Arc-length reparametrization
using QuadGK

function cp_laplace(p)
	integrand(x) = abs(x)^(2p) * exp(-2 * abs(x)^p)
	val, _ = quadgk(integrand, -Inf, Inf, rtol = 1e-8)
	return val
end

α_laplace(p) = 2p * sqrt(cp_laplace(p))

f(s, p, umin) = s / α_laplace(p) + sqrt(umin)

s_laplace(u, p, umin) = α_laplace(p) * (sqrt(u) - sqrt(umin))

u_laplace(s, p, umin) = f(s, p, umin)^2

du_ds_laplace(s, p, umin) = (2 / α_laplace(p)) * f(s, p, umin)

du_ds_2_laplace(p) = 2 / α_laplace(p)^2


laplace_0_n(s, t, p; umin = 1e-4) = laplace_0(u_laplace(s, p, umin), t, p)

laplace_1_n(s, t, p; umin = 1e-4) =
	du_ds_laplace(s, p, umin) * laplace_1(u_laplace(s, p, umin), t, p)

laplace_2_n(s, t, p; umin = 1e-4) =
	du_ds_2_laplace(p) * laplace_1(u_laplace(s, p, umin), t, p) +
	du_ds_laplace(s, p, umin)^2 * laplace_2(u_laplace(s, p, umin), t, p)

laplace_3_n(s, t, p; umin = 1e-4) =
	3 * du_ds_laplace(s, p, umin) * du_ds_2_laplace(p) *
	laplace_2(u_laplace(s, p, umin), t, p) +
	du_ds_laplace(s, p, umin)^3 * laplace_3(u_laplace(s, p, umin), t, p)


