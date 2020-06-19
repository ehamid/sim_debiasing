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
    end
end

function hermite_coefs(y, X, τ_pilot, d)
    return (mean(y .* hermite(X * τ_pilot, d)))
end

function dlasso(y, X, τ_pilot, λ_lasso, Σ, Σ_inv, deb_ind, hermite_deg)
    β_lasso = glmnet(X,y,lambda=[λ_lasso],intercept=false).betas[:,1]
    τ_lasso = β_lasso ./ sqrt(β_lasso' * Σ * β_lasso)
    u = X * Σ_inv[:,deb_ind]
    # u = (Σ \ X')'[:,deb_ind]
    y_mean = mean(y)

    h = zeros(length(y))
    if hermite_deg > 1
        h = y_mean * ones(length(y))
        for deg in 2:hermite_deg
            h = h .+ (hermite_coefs(y, X, τ_pilot, deg) .* hermite(X * τ_pilot, deg))
        end
    end
    β_dlasso = β_lasso[deb_ind] + sum(u .* (y - X * β_lasso - h)) / sum(u .* X[:,deb_ind])
    return β_dlasso
end

function ss_dlasso(y, X, λ_lasso, Σ, Σ_inv, deb_ind, hermite_deg)
    mid = Int(floor(length(y) / 2))
    y1 = y[1:mid]
    X1 = X[1:mid,:]
    y2 = y[mid+1 : end]
    X2 = X[mid+1 : end, :]

    if hermite_deg == 1
        τ_pilot = 0
        return dlasso(y, X, τ_pilot, λ_lasso[1], Σ, Σ_inv, deb_ind, hermite_deg)
    else
        τ_pilot1 = glmnet(X2,y2,lambda=[λ_lasso[2]],intercept=false).betas[:,1]
        τ_pilot1 = τ_pilot1 ./ sqrt(τ_pilot1' * Σ * τ_pilot1)
        β_deb1 = dlasso(y1, X1, τ_pilot1, λ_lasso[2], Σ, Σ_inv, deb_ind, hermite_deg)

        τ_pilot2 = glmnet(X1,y1,lambda=[λ_lasso[2]],intercept=false).betas[:,1]
        τ_pilot2 = τ_pilot2 ./ sqrt(τ_pilot2' * Σ * τ_pilot2)
        β_deb2 = dlasso(y2, X2, τ_pilot2, λ_lasso[2], Σ, Σ_inv, deb_ind, hermite_deg)
        return mean([β_deb1 β_deb2])
    end
end

#
# function f(t)
#     return (t .- t.^2 ./ 2 .+ t.^3 ./ 3)
# end
# μ = 2



function f(t)
    return (5 * sin.(t))
end
μ = 5/sqrt(ℯ)

plot(-4:0.1:4, f(-4:0.1:4), linewidth=2)

#---
Random.seed!(1)

p = 2500
n = 2000

s = 5
τ = [collect(1:s); zeros(p - s)]
ρ = 0.5
Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p]
Σ_inv = inv(Σ)
τ = τ / sqrt(τ' * Σ * τ)
P_X = MvNormal(zeros(p), Σ)


β = μ .* τ

X = rand(P_X, n)'
ϵ = rand(Normal(0,1), n)
y = f(X * τ) .+ 0.1 .* ϵ



cv = glmnetcv(X,y, nfolds = 50)
λ_cv = cv.lambda[argmin(cv.meanloss)]
β_pilot = cv.path.betas[:, argmin(cv.meanloss)]
τ_pilot = β_pilot ./ sqrt(β_pilot' * Σ * β_pilot)
cv2 = glmnetcv(X[1:Int(n/2),:],y[1:Int(n/2)],nfolds = 50)
λ_cv2 = cv2.lambda[argmin(cv2.meanloss)]


#---
Random.seed!(123)
nreps = 500
hermite_deg = 5
b = ones((nreps,hermite_deg))
b_ss = ones((nreps,hermite_deg))
deb_ind = 1



for d in 1:hermite_deg
    for i in 1:nreps
        X = rand(P_X, n)'
        ϵ = rand(Normal(0,1), n)
        y = f(X * τ) .+ 0.1 .* ϵ
        # b[i,d] = dlasso(y,X,τ, λ_cv, Σ, 1, d)
        b_ss[i,d] = ss_dlasso(y, X, [λ_cv λ_cv2], Σ, Σ_inv, deb_ind, d)
    end
    print("\nd: $d")
end


# println(mean(b, dims = 1) .-  β[deb_ind])
# println(var(b, dims = 1))
# histogram(b, alpha = 0.7)

#---

μs = [hermite_coefs(y, X, τ_pilot, d) for d in 0:5]
function f_hat(t, μs)
    u = zeros(length(t))
    for j in 1:length(μs)
        u .+= μs[j] .* hermite(t, j-1)
    end
    return u
end

l = 4
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
println("bias: ", mean(b_ss, dims = 1) .-  β[deb_ind])
println("std dev: ", sqrt.(var(b_ss, dims = 1)))
histogram(b_ss, alpha = 0.2)

boxplot(b_ss .- β[1], legend=:bottomleft)
hline!([0], linestyle=:dot, label="True β[1]",
    linewidth = 2, linecolor = :red)

print("RMSE: ", sqrt.(mean((b_ss .- β[deb_ind]).^2, dims = 1)))

histogram(b_ss[:,1] )
plot(kde(b_ss[:,1]))

histogram(b_ss .- β[1], normed=true, alpha=0.3,
 color=[:deepskyblue :orange], label=[], legend=:topright)
density(b_ss .- β[1], label=["Order $j Hermite Expansion" for j in 1:5])
xlabel!(L"$\tilde{\beta}_1 - \beta_1$")
title!("Histogram of Centered Estimates")
ylabel!("Frequency")
savefig("histogram.png")




#---
function dfactorial(n)
    df::Float64 = 1
    for j in n:-2:1
        df *= j
    end
    return df
end


function expected_val(n)
    S = 1
    for j in 1:n
        S += (-1)^j / dfactorial(2*j)
    end
    return S
end
