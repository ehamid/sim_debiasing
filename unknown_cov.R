require(mvtnorm)
require(glmnet)
require(flare)
require(parallel)
require(xtable)

set.seed(1)

n_list = c(200,500)
kappa_list <- c(0,0.5)
s_list <- c(5,10)
nrep = 200

cov_results <- data.frame()
test_results <- data.frame()
start_time <- proc.time()
case_time <- proc.time()
case = 0

f = function(u){
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
    rho <- max(diag(Sigma))
    sigma = 1
    for (s in s_list) {
      deb_index = c(1:s, sample(x = (1+s):p, size = 10))  # The indices of covariates to be debiased
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
        
        z = y - X %*% pilot_est
        
        for (j in 1:length(deb_index)) {
          ind <- deb_index[j]
          
          
          # ### Without Sample Splitting
          # nodewise_fit <- glmnet(X[,-ind], X[, ind], intercept = FALSE)
          # Gamma <- coef(nodewise_fit)[-1,]
          # RR <- matrix(X[,ind], nrow = n, ncol = dim(Gamma)[2]) - X[,-ind] %*% Gamma
          # eta = apply(abs((t(RR) %*% X[,-ind])/(sqrt(colSums(RR^2)))), 1, max)
          # R = RR[,min(which(eta <  sqrt(log(p))))]
          # 
          ### With sample splitting
          I1 <- 1:floor(n/2)
          I2 <- (floor(n/2)+1):n

          nodewise_fit1 <- glmnet(X[I1,-ind], X[I1, ind], intercept = FALSE)
          Gamma1 <- coef(nodewise_fit1)[-1,]
          RR1 <- matrix(X[I2,ind], nrow = length(I2), ncol = dim(Gamma1)[2]) - X[I2,-ind] %*% Gamma1
          eta1 = apply(abs((t(RR1) %*% X[I2,-ind])/(sqrt(colSums(RR1^2)))), 1, max)
          R1 = RR1[,max(which(eta1 >=  sqrt(log(p))))]
          
           
          nodewise_fit2 <- glmnet(X[I2,-ind], X[I2, ind], intercept = FALSE)
          Gamma2 <- coef(nodewise_fit2)[-1,]
          RR2 <- matrix(X[I1,ind], nrow = length(I1), ncol = dim(Gamma2)[2]) - X[I1,-ind] %*% Gamma2
          eta2 = apply(abs((t(RR2) %*% X[I1,-ind])/(sqrt(colSums(RR2^2)))), 1, max)
          R2 = RR2[,max(which(eta2 >=  sqrt(log(p))))]
          R = c(R2, R1)
          
          est[i, j] =  sum(R * (y - X[, -ind] %*%  pilot_est[-ind])) / sum(R * X[, ind])
          std_error[i, j] <-
            sqrt(sum(z ^ 2 * R ^ 2)) / sum(R * X[, ind])
          # bias_bound[i, j] <-
          #   max(abs(t(R) %*% X[,-ind])) * cvfit$lambda.min / sum(R * X[, ind])
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
      covSc <- mean(coverage[(s + 1):(s+10)])
      avglen <- 2 * colMeans(bias_bound + std_error * nq)
      lenS <- mean(avglen[1:s])
      lenSc <- mean(avglen[(s + 1):(s+10)])
      
      TPR <-
        colMeans((est[, 1:s] -  std_error[, 1:s] * nq - bias_bound[, 1:s] > 0) +
                   ((est[, 1:s] + std_error[, 1:s] * nq + bias_bound[, 1:s]) < 0))
      FPR <-
        colMeans((est[, (1 + s):(s+10)] -  std_error[, (1 + s):(s+10)] * nq - bias_bound[, (1 +
                                                                                              s):(s+10)] > 0) +
                   ((est[, (1 + s):(s+10)] + std_error[, (1 + s):(s+10)] * nq + bias_bound[, (1 +
                                                                                                s):(s+10)]) < 0))
      
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

write.csv(cov_results, "exp_coverage_unknown.csv")
write.csv(test_results, "exp_test_unknown.csv")

#Generate LaTeX tables from results
xtable(cov_results)
xtable(test_results)
