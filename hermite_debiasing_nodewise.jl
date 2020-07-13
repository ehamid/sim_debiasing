using GLMNet, Random, Statistics, Distributions, Plots, StatsPlots
using LaTeXStrings

pgfplotsx()

function hermite(x,d)
    if d == 0
        return 1
    elseif d == 1
        return x
    elseif d == 2
        return (x .^2 .- 1) ./ sqrt(2)
    elseif d == 3
        return (x .^ 3 .- 3 .* x) ./ sqrt(6)
    elseif d == 4
        return (x .^ 4 .- 6 .* x .^ 2 .+ 3) ./ sqrt(24)
    elseif d == 5
        return (x .^ 5 .- 10 .* x .^ 3 .+ 15 .* x) ./ sqrt(120)
    elseif d == 6
        return (x .^ 6 .- 15 .* x .^4 .+ 45 .* x .^ 2 .- 15) ./ sqrt(720)
    elseif d == 7
        return (x .^ 7 .- 21 .* x .^ 5 .+ 105 .* x .^ 3 .- 105 .* x) ./ sqrt(5040)
    elseif d == 8
        return (x .^ 8 .- 28 .* x .^ 6 .+ 210 .* x .^ 4 .- 420 .* x .^ 2 .+ 105) ./ sqrt(40320)
    elseif d == 9
        return (x .^ 9 .- 36 .* x .^ 7 .+ 378 .* x .^ 5 .- 1260 .* x .^ 3 .+ 945 .* x) ./ sqrt(362880)
    elseif d == 10
        return (x .^ 10 .- 45 .* x .^ 8 .+ 630 .* x .^ 6 .- 3150 .* x .^ 4 .+ 4725 .* x .^ 2 .- 945) ./ sqrt(3628800)
    end
end

function hermite_coefs(y, X, τ_pilot, d)
    return (mean(y .* hermite(X * τ_pilot, d)))
end

function dlasso(y, X, τ_pilot, μs, Σ, γ, deb_ind, hermite_deg)
    u = X[:,deb_ind] - X[:, 1:end .!=deb_ind] * γ
    h = μs[1] .* ones(length(y))

    for deg in 1:hermite_deg
        h = h .+ μs[deg+1] .* hermite(X * τ_pilot, deg)
    end
    β_dlasso = μs[2] * τ_pilot[deb_ind] + sum(u .* (y - h)) / sum(u .* X[:,deb_ind])
    return β_dlasso
end

function ss_estimates(y, X, Σ, λ, hermite_deg)
    if hermite_deg == 1
        β_pilot1 = glmnet(X,y,lambda=[λ[1]],intercept=false).betas[:,1]
        return [β_pilot1, [0,1]]
    else
        mid = Int(floor(length(y) / 2))
        y1 = y[1:mid]
        X1 = X[1:mid,:]
        y2 = y[(mid+1):end]
        X2 = X[(mid+1):end,:]


        β_pilot1 = glmnet(X2,y2,lambda=[λ[2]],intercept=false).betas[:,1]
        τ_pilot1 = β_pilot1 ./ sqrt(β_pilot1' * Σ * β_pilot1)

        μs = zeros(hermite_deg + 1)
        μs[1] = mean(y1)
        for j in 1:hermite_deg
            μs[j+1] = hermite_coefs(y1, X1, τ_pilot1, j)
        end

        return [τ_pilot1, μs]
    end
end




function ss_dlasso(y, X, λ_lasso, λ_nodewise, deb_ind, hermite_deg)
    mid = Int(floor(length(y) / 2))
    y1 = y[1:mid]
    X1 = X[1:mid,:]
    y2 = y[mid+1:end]
    X2 = X[mid+1:end,:]

    Σ_hat = X' * X ./ length(y)
    γ = glmnet(X[:,1:end .!=deb_ind], X[:,deb_ind],lambda=[λ_nodewise],intercept=false).betas[:,1]

        τ_pilot1, μs1 = ss_estimates(y1, X1, Σ_hat, λ_lasso, hermite_deg)
        β_deb1 = dlasso(y2, X2, τ_pilot1, μs1, Σ_hat, γ, deb_ind, hermite_deg)

        τ_pilot2, μs2 = ss_estimates(y2, X2, Σ_hat, λ_lasso, hermite_deg)
        β_deb2 = dlasso(y1, X1, τ_pilot2, μs2, Σ_hat, γ, deb_ind, hermite_deg)

        return mean([β_deb1 β_deb2])
end

#---

#
# function f(t)
#     return (t .- t.^2 ./ 2 .+ t.^3 ./ 3)
# end
# μ = 2
#

#
# function f(t)
#     return (5 .* (sin.(t) + cos.(t)))
# end
# μ = 5/sqrt(ℯ)



function f(t)
    return (5 .* sin.(t))
end
μ = 5/sqrt(ℯ)


plot(-4:0.1:4, f(-4:0.1:4), linewidth=2)

#---
Random.seed!(123)

p = 2000
n = 1000

s = 10
τ = [collect(s:-1:1); zeros(p - s)]
ρ = 0.5
Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p]
Σ_inv = inv(Σ)
τ = τ / sqrt(τ' * Σ * τ)
P_X = MvNormal(zeros(p), Σ)


β = μ .* τ

X = copy(rand(P_X, n)')
ϵ = rand(Normal(0,1), n)
y = f(X * τ) .+ 0.1 .* ϵ

deb_ind = 1

cv1 = glmnetcv(X[1:Int(floor(n/2)),:],y[1:Int(floor(n/2))],nfolds = 50,
    intercept=false)
λ_cv1 = cv1.lambda[argmin(cv1.meanloss)]
τ_pilot = cv1.path.betas[:, argmin(cv1.meanloss)]
τ_pilot = τ_pilot ./ √(τ_pilot' * Σ * τ_pilot)

cv2 = glmnetcv(X[1:Int(floor(n/4)),:],y[1:Int(floor(n/4))],nfolds = 50,
    intercept=false)
λ_cv2 = cv2.lambda[argmin(cv2.meanloss)]

cv_nodewise = glmnetcv(X[:,1:end .!=deb_ind], X[:,deb_ind], nfolds = 50,
    intercept=false)
λ_nodewise = cv_nodewise.lambda[argmin(cv_nodewise.meanloss)]



#---
Random.seed!(123)
nreps = 1000
hermite_deg = 10
b = ones((nreps,hermite_deg))
deb_ind = 1



for d in 1:hermite_deg
    println("\nd: $d")
    for i in 1:nreps
        if i % 100 == 0
            print(" $i, ")
        end
        X .= rand(P_X, n)'
        ϵ .= rand(Normal(0,1), n)
        y .= f(X * τ) .+ 0.1 .* ϵ
        b[i,d] = ss_dlasso(y, X, [λ_cv1, λ_cv2], λ_nodewise, deb_ind, d)
    end
end

#---

μs = [hermite_coefs(y, X, τ_pilot, d) for d in 0:5]
function f_hat(t, μs)
    u = zeros(length(t))
    for j in 1:length(μs)
        u .+= μs[j] .* hermite(t, j-1)
    end
    return u
end

l = 3
plot(-l:0.05:l, [f(-l:0.05:l), f_hat(-l:0.05:l, μs)],
    color=[:deepskyblue :darkorange],
    legend=:topleft,
    linestyle=[:dash :solid],
    xlabel=L"$t$",
    ylabel=L"$g(t)$",
    labels=[(L"$g$ : True Link Function ") L"$\hat{g}$ : Cubic Hermite Estimate"],
    title="True and Estimated Link Functions")
ylabel!(L"$g(t)$")
xlabel!(L"$t$")
# savefig("linkplot.png")


#---
println("\nbias: \t\t", round.( sqrt(n) .* (mean(b, dims = 1) .-  β[deb_ind]), digits=3))
println("\nstd dev: \t", round.(sqrt.( n.* var(b, dims = 1,corrected=false)), digits=3))
print("\nRMSE: \t\t", round.(sqrt.( n.* mean((b .- β[deb_ind]).^2, dims = 1)), digits=3))


boxplot(b, legend=:best)
hline!([β[1]], linestyle=:dot, label="True β[1]",
    linewidth = 2, linecolor = :red)


histogram(b[:,[1,5]] .- β[1], normed=true,
            alpha=0.3, legend=:topright)
density!(b[:,[1,5]] .- β[1])
vline!([0], linestyle=:dot)
xlabel!(L"$\tilde{\beta}_1 - \beta_1$")
title!("Histogram of Centered Estimates")
ylabel!("Frequency")
savefig("histogram.png")




#---
b_sc = copy(b)
b_s = copy(b)



#---
#Jacknife

Random.seed!(123)

p = 2000
n = 1000

s = 10
τ = [collect(s:-1:1); zeros(p - s)]
ρ = 0.5
Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p]
Σ_inv = inv(Σ)
τ = τ / sqrt(τ' * Σ * τ)
P_X = MvNormal(zeros(p), Σ)


β = μ .* τ

X .= copy(rand(P_X, n)')
ϵ .= rand(Normal(0,1), n)
y .= f(X * τ) .+ 0.1 .* ϵ


folds = 100
leaveout = Int(floor(n / folds))

n_i = Int((folds - 1) * leaveout)
X_i = zeros((n_i, p))
y_i = zeros(n_i)

b_jack = zeros((folds, hermite_deg))
indices=collect(1:n)
repl=20
v = zeros((repl, hermite_deg))
for rep in 1:repl
    X .= copy(rand(P_X, n)')
    ϵ .= rand(Normal(0,1), n)
    y .= f(X * τ) .+ 0.1 .* ϵ

    for d in 1:hermite_deg
        println(d)
        for i in 1:folds
                if i%100 == 0
                    print(" , $i ")
                end
            indices = [j for j in 1:n if j ∉ (i-1)*leaveout+1:i*leaveout]
            # indices .= sample(1:n, n, replace=true)
            # X_i .= X[indices,:]
            # y_i .= y[indices]
            b_jack[i,d] = ss_dlasso(y[indices], X[indices,:], sqrt(n/n_i).*[λ_cv1, λ_cv2], sqrt(n/n_i)* λ_nodewise, deb_ind, d)
        end
    end
    v[rep,:] = var(b_jack, dims=1)
end

print(round.((var(b_jack, dims=1)), digits=6))

#---
plot(mean((b .- β[1]).^2, dims=1)')

print(round.(mean((b .- β[1]).^2, dims=1), digits=6))
print(round.(std((b .- β[1]).^2, dims=1) ./ sqrt(nreps), digits=6))

#---
colors = [:deepskyblue :darkorange]
plot([(mean((b .- β[deb_ind]).^2, dims = 1))'],
 markershape=:diamond, label="Mean Squared Error", legend=:topright,
  color=colors[1],linewidth=1.5, yerror=std((b .- β[deb_ind]).^2, dims = 1)'./sqrt(nreps))

plot(mean(v,dims=1)',
   markershape=:diamond, label="Bootstrap Estimate of Variance", legend=:topright,
    color=colors[2],linewidth=1.5,yerror=std(v,dims=1)'./ sqrt(10))

# plot!([0.002944 0.003642 0.00128 0.001374 0.001248 0.001484 0.001673 0.001453 0.001275 0.001344]',
#    markershape=:diamond, label="MSE", legend=:topright,
#     color=colors[2],linewidth=1.5,
#      yerror=[0.000128 0.000155 6.6e-5 6.8e-5 7.4e-5 0.000133 0.000185 0.000101 6.3e-5 0.000123]')


plot!((var(b_jack, dims = 1))',
   markershape=:diamond, label="Bootstrap Estimate of Variance", legend=:topright,
    color=colors[2],linewidth=1.5, alpha = 0.5)
xlabel!("Degree of Hermite Expansion")
title!(L"Accuracy of Estimators (known $\Sigma$)")
savefig("knownSigma.png")
