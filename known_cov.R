require(mvtnorm)
require(glmnet)
require(flare)
require(parallel)
require(xtable)
require(EQL)

set.seed(1)

n_list = c(200, 500)
s_list <- c(5, 10)
kappa_list <- c(0, 0.5)
nrep = 200

cov_results <- data.frame()
test_results <- data.frame()
start_time <- proc.time()
case_time <- proc.time()
case = 0
m_h = floor(log(n) ^ (2 / 3))

f = function(u) {
  return(sign(u))
}

# f = function(u){
#   U <- rexp(n = length(u),rate = 1)
#   return(U * exp(u))
# }


for (n in n_list) {
  p <- 2 * n
  for (kappa in kappa_list) {
    Sigma = toeplitz(kappa ^ seq(0, p - 1))
    Omega = solve(Sigma)
    sigma = 1
    for (s in s_list) {
      deb_index = c(1:s, sample(x = (1 + s):p, size = 10))  # The indices of covariates to be debiased
      tau = c(s:1, rep(0, p - s))   #Strong Sparsity
      tau = tau / as.double(sqrt(t(tau) %*% Sigma %*% tau))
      
      
      est = matrix(0, nrep, length(deb_index))
      std_error = matrix(0, nrep, length(deb_index))
      bias_bound = matrix(0, nrep, length(deb_index))
      mu = 0
      
      for (i in 1:nrep) {
        # X = rmvnorm(n, mean = rep(0, p), sigma = Sigma)
        X = scale(rmvnorm(n, mean = rep(1, p), sigma = Sigma), scale = FALSE)
        e = rnorm(n)
        y = f(X %*% tau) + sigma * e
        # y = f(X %*% tau)
        
        cvfit = cv.glmnet(X, y, intercept = FALSE, parallel = TRUE)
        pilot_est = coef(cvfit, s = cvfit$lambda.min)[-1]
        
        #computing the hermite polynomial estimate
        # mu_hat = rep(0, m_h)
        # mu_hat[1] = sqrt(t(pilot_est) %*% Sigma %*% pilot_est)
        # H = rep(0, n)
        # for(l in 2:m_h){
        #   h = hermite(X %*% pilot_est / mu_hat[1], l, prob =TRUE)/ sqrt(factorial(l))
        #   mu_hat[l] = mean(y * h)
        #   H = H + mu_hat[l] * h
        # }
        
        z = y - X %*% pilot_est
        
        for (j in 1:length(deb_index)) {
          ind <- deb_index[j]
          R = X %*% Omega[, ind]
          est[i, j] =  sum(R * (y - X[,-ind] %*%  pilot_est[-ind])) / sum(R * X[, ind])
          std_error[i, j] <-
            sqrt(sum(z ^ 2 * R ^ 2)) / sum(R * X[, ind])
          # bias_bound[i, j] <-0
        }
        
        mu = mu + sum(f(X %*% tau) * (X %*% tau)) / n
      }
      
      mu = mu / nrep
      
      mu_hat = as.double(sqrt(t(y) %*% X %*% pilot_est / n))
      beta = mu * tau
      
      
      
      std_dev <- sqrt(colMeans(est ^ 2) - colMeans(est) ^ 2)
      
      nq <- qnorm(0.975)
      
      
      coverage <-
        colMeans((est -  std_error * nq - bias_bound  - (
          matrix(
            rep(beta[deb_index], nrow(est)),
            ncol = length(deb_index),
            byrow = TRUE
          )
        ) <= 0) *
          ((est + std_error * nq + bias_bound) - (
            matrix(
              rep(beta[deb_index], nrow(est)),
              ncol = length(deb_index),
              byrow = TRUE
            )
          ) >= 0))
      
      
      std_dev
      colMeans(std_error)
      
      covS <- mean(coverage[1:s])
      covSc <- mean(coverage[(s + 1):(s + 10)])
      avglen <- 2 * colMeans(bias_bound + std_error * nq)
      lenS <- mean(avglen[1:s])
      lenSc <- mean(avglen[(s + 1):(s + 10)])
      
      TPR <-
        colMeans((est[, 1:s] -  std_error[, 1:s] * nq - bias_bound[, 1:s] > 0) +
                   ((est[, 1:s] + std_error[, 1:s] * nq + bias_bound[, 1:s]) < 0))
      FPR <-
        colMeans((est[, (1 + s):(s + 10)] -  std_error[, (1 + s):(s +
                                                                    10)] * nq - bias_bound[, (1 +
                                                                                                s):(s +
                                                                                                      10)] > 0) +
                   ((est[, (1 + s):(s + 10)] + std_error[, (1 + s):(s +
                                                                      10)] * nq + bias_bound[, (1 +
                                                                                                  s):(s +
                                                                                                        10)]) < 0))
      
      cov_output <-
        data.frame(
          "Configuration" = sprintf("(%s, %s, %s)", n, kappa, s),
          "Cov S" = covS,
          "Cov Sc" = covSc,
          "l(S)" = lenS,
          "l(Sc)" = lenSc
        )
      test_output <-
        data.frame(
          "Configuration" = sprintf("(%s, %s, %s)", n, kappa, s),
          "FPR" = mean(FPR),
          "TPR" = mean(TPR),
          "TPR1" = TPR[1],
          "TPR2" = TPR[2],
          "TPR3" = TPR[3],
          "TPR4" = TPR[4],
          "TPR5" = TPR[5]
        )
      case <- case + 1
      print(proc.time() - case_time)
      case_time <- proc.time()
      print(sprintf("Case: %s", case))
      print(cov_output)
      print(test_output)
      cov_results <- rbind(cov_results, cov_output)
      test_results <- rbind(test_results, test_output)
      
    }
  }
}

proc.time() - start_time

write.csv(cov_results, "exp_coverage_known.csv")
write.csv(test_results, "exp_test_known.csv")

#Generate LaTeX tables from results
xtable(cov_results)
xtable(test_results)
beta = tau * sqrt(2 / pi)

data = rbind(est, unknown_est)
require(tidyverse)

tib = as.tibble(data)
tib['Covariance'] = "Known"
tib[201:400, 'Covariance'] = "Unknown"
summary(tib['cov'])

tidydata = tib[, c(1:15, 21)] %>% gather('Variable',  'Estimate', 1:15)
tidydata$Variable = as.factor(tidydata$Variable)
tidydata$cov = as.factor(tidydata$cov)
tidydata$Variable = factor(
  tidydata$Variable,
  levels = c(
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15"
  ),
  ordered = TRUE
)

ggplot() + geom_boxplot(data = tidydata, aes(x = Variable, y = Estimate, fill = Covariance)) +
  geom_point(
    data = data.frame(x = unique(tidydata$Variable), y = beta[1:15]),
    aes(x = x, y = y),
    color = 'yellow',
    size = 3
  )
