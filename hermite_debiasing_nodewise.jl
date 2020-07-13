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
hermite_deg = 5
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
println("\nbias: \t\t", round.( (mean(b, dims = 1) .-  β[deb_ind]), digits=6))
println("\nstd dev: \t", round.(sqrt.( var(b, dims = 1)), digits=6))
print("\nRMSE: \t\t", round.(sqrt.(mean((b .- β[deb_ind]).^2, dims = 1)), digits=6))


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

p = 3000
n = 2000

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


folds = 500
leaveout = Int(floor(n / folds))

n_i = Int((folds - 1) * leaveout)
X_i = zeros((n_i, p))
y_i = zeros(n_i)

b_jack = zeros((folds, hermite_deg))
indices=collect(1:n)

for d in 1:hermite_deg
    println(d)
    for i in 1:folds
            if i%100 == 0
                print(" , $i ")
            end
        # indices = [j for j in 1:n if j ∉ (i-1)*leaveout+1:i*leaveout]
        indices .= sample(1:n, n, replace=true)
        # X_i .= X[indices,:]
        # y_i .= y[indices]
        b_jack[i,d] = ss_dlasso(y[indices], X[indices,:], sqrt(n / n_i) .* [λ_cv1, λ_cv2], λ_nodewise, deb_ind, d)
    end
end

print(round.(sqrt.(var(b_jack, dims=1)), digits=6))
