#' Extension to the Model class to use Markov Chain Monte Carlo Maximum Likelihood
#' 
#' An R6 class representing a GLMM and study design
#' @details
#' For the generalised linear mixed model 
#' 
#' \deqn{Y \sim F(\mu,\sigma)}
#' \deqn{\mu = h^-1(X\beta + Z\gamma)}
#' \deqn{\gamma \sim MVN(0,D)}
#' 
#' where h is the link function. A Model in comprised of a \link[glmmr]{MeanFunction} object, which defines the family F, 
#' link function h, and fixed effects design matrix X, and a \link[glmmr]{Covariance} object, which defines Z and D. The class provides
#' methods for analysis and simulation with these models.
#' 
#' This class provides methods for: data simulation (`sim_data()` and `fitted()`), model fitting using Markov Chain 
#' Monte Carlo Maximum Likelihood (MCML) methods (`MCML()`), design analysis via simulation including power (`analysis()`),
#' deletion diagnostics (`dfbeta()`), and permutation tests including p-values and confidence intervals (`permutation()`).
#' 
#' The class by default calculates the covariance matrix of the observations as:
#' 
#' \deqn{\Sigma = W^{-1} + ZDZ^T}
#' 
#' where _W_ is a diagonal matrix with the WLS iterated weights for each observation equal
#' to, for individual _i_ \eqn{\phi a_i v(\mu_i)[h'(\mu_i)]^2} (see Table 2.1 in McCullagh 
#' and Nelder (1989) <ISBN:9780412317606>). For very large designs, this can be disabled as
#' the memory requirements can be prohibitive.
#' @references 
#' Braun and Feng
#' McCullagh
#' Stan
#' McCullagh and Nelder
#' Approx GLMMs paper
#' Watson confidence interval
#' @importFrom Matrix Matrix
#' @export 
ModelMCML <- R6::R6Class("ModelMCML",
                         inherit = Model,
                         public = list(
                           #'@description
                           #'Markov Chain Monte Carlo Maximum Likelihood  model fitting
                           #'
                           #'@details
                           #'**MCMCML**
                           #'Fits generalised linear mixed models using one of three algorithms: Markov Chain Newton
                           #'Raphson (MCNR), Markov Chain Expectation Maximisation (MCEM), or Maximum simulated
                           #'likelihood (MSL). All the algorithms are described by McCullagh (1997). For each iteration
                           #'of the algorithm the unobserved random effect terms (\eqn{\gamma}) are simulated
                           #'using Markov Chain Monte Carlo (MCMC) methods (we use Hamiltonian Monte Carlo through Stan),
                           #'and then these values are conditioned on in the subsequent steps to estimate the covariance
                           #'parameters and the mean function parameters (\eqn{\beta}). For all the algorithms, 
                           #'the covariance parameter estimates are updated using an expectation maximisation step.
                           #'For the mean function parameters you can either use a Newton Raphson step (MCNR) or
                           #'an expectation maximisation step (MCEM). A simulated likelihood step can be added at the 
                           #'end of either MCNR or MCEM, which uses an importance sampling technique to refine the 
                           #'parameter estimates. 
                           #'
                           #'The accuracy of the algorithm depends on the user specified tolerance. For higher levels of
                           #'tolerance, larger numbers of MCMC samples are likely need to sufficiently reduce Monte Carlo error.
                           #'
                           #'The function also offers different methods of obtaining standard errors. First, one can generate
                           #'estimates from the estimated Hessian matrix (`se.method = 'lik'`). Second, there are robust standard 
                           #'errors using a sandwich estimator based on White (1982) (`se.method = 'robust'`). 
                           #'Third, there are use approximate generalised least squares estimates based on the maximum likelihood 
                           #'estimates of the covariance
                           #'parameters (`se.method = 'approx'`), or use a permutation test approach (`se.method = 'perm'`).
                           #'Note that the permutation test can be accessed separately with the function `permutation_test()`.
                           #'
                           #'There are several options that can be specified to the function using the `options` argument. 
                           #'The options should be provided as a list, e.g. `options = list(method="mcnr")`. The possible options are:
                           #'* `b_se_only` TRUE (calculate standard errors of the mean function parameters only) or FALSE (calculate
                           #'all standard errors), default it FALSE.
                           #'* `use_cmdstanr` TRUE (uses `cmdstanr` for the MCMC sampling, requires cmdstanr), or FALSE (uses `rstan`). Default is FALSE.
                           #'* `skip_cov_optim` TRUE (skips the covariance parameter estimation step, and uses the values covariance$parameters), or 
                           #'FALSE (run the whole algorithm)], default is FALSE
                           #'* `sim_lik_step` TRUE (conduct a simulated likelihood step at the end of the algorithm), or FALSE (does
                           #'not do this step), defaults to FALSE.
                           #'* `no_warnings` TRUE (do not report any warnings) or FALSE (report warnings), default to FALSE
                           #'* `perm_type` Either `cov` (use weighted test statistic in permutation test) or `unw` (use unweighted
                           #' test statistic), defaults to `cov`. See `permutation_test()`.
                           #' * `perm_iter` Number of iterations for the permutation test, default is 100.
                           #' * `perm_parallel` TRUE (run permuation test in parallel) or FALSE (runs on a single thread), default to TRUE
                           #' * `warmup_iter` Number of warmup iterations on each iteration for the MCMC sampler, default is 500
                           #' * `perm_ci_steps` Number of steps for the confidence interval search procedure if using the permutation
                           #' test, default is 1000. See `permutation_test()`.
                           #' * `fd_tol` The tolerance of the first difference method to estimate the Hessian and Gradient, default 
                           #' is 1e-4.
                           #'
                           #'@param y A numeric vector of outcome data
                           #'@param start Optional. A numeric vector indicating starting values for the MCML algorithm iterations. 
                           #'If this is not specified then the parameter values stored in the linked mean function object will be used.
                           #'@param se.method One of either `'lik'`, `'approx'`, `'perm'`, or `'none'`, see Details.
                           #'@param method The MCML algorithm to use, either `mcem` or `mcnr`, see Details. Default is `mcem`.
                           #'@param permutation.par Optional. Integer specifing the index of the parameter if permutation tests are being used.
                           #'@param verbose Logical indicating whether to provide detailed output, defaults to TRUE.
                           #'@param tol Numeric value, tolerance of the MCML algorithm, the maximum difference in parameter estimates 
                           #'between iterations at which to stop the algorithm.
                           #'@param m Integer. The number of MCMC draws of the random effects on each iteration.
                           #'@param max.iter Integer. The maximum number of iterations of the MCML algorithm.
                           #'@param options An optional list providing options to the algorithm, see Details.
                           #'@return A `mcml` object
                           #'@examples
                           #'\dontrun{
                           #'df <- nelder(~(cl(10)*t(5)) > ind(10))
                           #' df$int <- 0
                           #' df[df$cl > 5, 'int'] <- 1
                           #' des <- Design$new(
                           #'   covariance = list(
                           #'     data = df,
                           #'     formula = ~ (1|gr(cl)*ar1(t)),
                           #'     parameters = c(0.25,0.8)),
                           #'   mean.function = list(
                           #'     formula = ~ factor(t) + int - 1,
                           #'     data=df,
                           #'     parameters = c(rep(0,5),0.6),
                           #'     family = binomial())
                           #' )
                           #' ysim <- des$sim_data()
                           #' # fits the models using MCEM but does not estimate standard errors
                           #' fit1 <- des$MCML(y = ysim,
                           #'   se.method = "none")
                           #' #fits the models using MCNR but does not estimate standard errors
                           #' fit2 <- des$MCML(y = ysim,
                           #'   se.method = "none",
                           #'   method="mcnr")
                           #' #fits the models and uses permutation tests for parameter of interest
                           #' fit3 <- des$MCML(y = ysim,
                           #'   se.method = "perm",
                           #'   permutation.par = 6,
                           #'   options = list(
                           #'     perm_type="unw",
                           #'     perm_iter=1000,
                           #'     perm_parallel=FALSE,
                           #'     perm_ci_steps=1000
                           #'   ))
                           #'  #adds a simulated likelihood step after the MCEM algorithm
                           #' fit4 <- des$MCML(y = des$sim_data(),
                           #'   se.method = "none",
                           #'   options = list(
                           #'     sim_lik_step=TRUE
                           #'   ))  
                           #'}
                           #'@md
                           MCML = function(y,
                                           start,
                                           se.method = "lik",
                                           method = "mcnr",
                                           permutation.par,
                                           verbose=TRUE,
                                           tol = 1e-2,
                                           m=100,
                                           max.iter = 30,
                                           mcmc = TRUE,
                                           options = list()){
                             
                             # checks
                             if(!se.method%in%c("perm","lik","none","robust","approx"))stop("se.method should be 'perm', 'lik', 'robust', 'approx', or 'none'")
                             if(se.method=="perm" & missing(permutation.par))stop("if using permutational based
inference, set permuation.par")
                             if(se.method=="perm" & is.null(self$mean_function$randomise))stop("random allocations
are created using the function in self$mean_function$randomise, but this has not been set. Please see help(MeanFunction)
for more details")
                             #set options
                             if(!is(options,"list"))stop("options should be a list")
                             b_se_only <- ifelse("b_se_only"%in%names(options),options$b_se_only,FALSE)
                             use_cmdstanr <- ifelse("use_cmdstanr"%in%names(options),options$use_cmdstanr,FALSE)
                             skip_cov_optim <- ifelse("skip_cov_optim"%in%names(options),options$skip_cov_optim,FALSE)
                             #method <- ifelse("method"%in%names(options),options$method,"mcnr")
                             sim_lik_step <- ifelse("sim_lik_step"%in%names(options),options$sim_lik_step,FALSE)
                             no_warnings <- ifelse("no_warnings"%in%names(options),options$no_warnings,FALSE)
                             perm_type <- ifelse("perm_type"%in%names(options),options$perm_type,"cov")
                             perm_iter <- ifelse("perm_iter"%in%names(options),options$perm_iter,100)
                             perm_parallel <- ifelse("perm_parallel"%in%names(options),options$perm_iter,TRUE)
                             warmup_iter <- ifelse("warmup_iter"%in%names(options),options$warmup_iter,500)
                             perm_ci_steps <- ifelse("perm_ci_steps"%in%names(options),options$perm_ci_steps,1000)
                             fd_tol <- ifelse("fd_tol"%in%names(options),options$fd_tol,1e-4)
                             trace <- ifelse("trace"%in%names(options),options$trace,0)
                             
                             P <- ncol(self$mean_function$X)
                             R <- length(unlist(self$covariance$parameters))
                             
                             family <- self$mean_function$family[[1]]
                             
                             parInds <- list(b = 1:P,
                                             cov = (P+1):(P+R),
                                             sig = P+R+1)
                             
                             if(family%in%c("gaussian")){
                               mf_parInd <- c(parInds$b,parInds$sig)
                             } else {
                               mf_parInd <- c(parInds$b)
                             }
                             
                             orig_par_b <- self$mean_function$parameters
                             orig_par_cov <- self$covariance$parameters
                             
                             #check starting values
                             if(family%in%c("gaussian")){
                               if(missing(start)){
                                 if(verbose)message("starting values not set, setting defaults")
                                 start <- c(self$mean_function$parameters,self$covariance$parameters,self$var_par)
                               }
                               if(length(start)!=(P+R+1))stop("wrong number of starting values")
                               all_pars <- 1:(P+R+1)
                             }
                             
                             if(family%in%c("binomial","poisson")){
                               if(!missing(start)){
                                 if(length(start)!=(P+R)){
                                   stop("wrong number of starting values")
                                 }
                               } else {
                                 if(verbose)message("starting values not set, setting defaults")
                                 start <- c(self$mean_function$parameters,self$covariance$parameters)
                               }
                               start <- c(start,1)
                               all_pars <- 1:(P+R)
                             }
                             
                             theta <- start
                             thetanew <- rep(1,length(theta))
                             
                             if(verbose)message(paste0("using method: ",method))
                             if(verbose)cat("\nStart: ",start[all_pars],"\n")
                             
                             iter <- 0
                             niter <- m
                             Q = ncol(self$covariance$Z)
                             
                             #parse family
                             file_type <- mcnr_family(self$mean_function$family)
                             invfunc <- self$mean_function$family$linkinv
                             
                             ## set up sampler
                             if(!requireNamespace("cmdstanr"))stop("cmdstanr not available")
                             model_file <- system.file("stan",
                                                       file_type$file,
                                                       package = "glmmrMCML",
                                                       mustWork = TRUE)
                             mod <- cmdstanr::cmdstan_model(model_file)
                             ## ALGORITHMS
                             while(any(abs(theta-thetanew)>tol)&iter <= max.iter){
                               iter <- iter + 1
                               if(verbose)cat("\nIter: ",iter,": ")
                               thetanew <- theta
                               Xb <- Matrix::drop(self$mean_function$X %*% thetanew[parInds$b])
                               L <- as.matrix(blockMat(self$covariance$get_chol_D(thetanew[parInds$cov])))
                               
                               data <- list(
                                 N = self$n(),
                                 Q = Q,
                                 Xb = Xb,
                                 Z = as.matrix(des$covariance$Z)%*%L,
                                 y = y,
                                 sigma = thetanew[parInds$sig],
                                 type=as.numeric(file_type$type)
                               )
                               
                               capture.output(fit <- mod$sample(data = data,
                                                                chains = 1,
                                                                iter_warmup = warmup_iter,
                                                                iter_sampling = m,
                                                                refresh = 0),
                                              file=tempfile())
                               dsamps <- fit$draws("gamma",format = "matrix")
                               dsamps <- t(dsamps %*% L)
                               #dsamps <- matrix(dsamps[,1,],ncol=Q)%*%L
                              # if(mcmc){
                              #   capture.output(suppressWarnings(fit <- rstan::sampling(stanmodels[[gsub(".stan","",file_type$file)]],
                              #                                                    data = data,warmup = 100,
                              #                                                    iter = 100 + m,chains=1)))
                              # } else {
                              #   capture.output(suppressWarnings(fit <- rstan::vb(stanmodels[[gsub(".stan","",file_type$file)]],
                              #                                                    data = data,
                              #                                                    algorithm="meanfield",
                              #                                                    keep_every=10,
                              #                                                    iter=1000,
                              #                                                    output_samples = m)))
                              # }
                               
                               # dsamps <- rstan::extract(fit,"gamma")
                               # #dsamps <- matrix(dsamps[,1,],ncol=Q)
                               # dsamps <- t(dsamps$gamma %*% L)
                               
                               # if(method == "mcnr"){
                               #   start_pars <- theta[c(parInds$b,parInds$cov)]
                               #   lower_b <- rep(-Inf, length(parInds$b))
                               # } else if(method == "mcem"){
                               #   start_pars <- theta[all_pars]
                               #   lower_b <- c(rep(-Inf, length(parInds$b)),1e-6)
                               # }
                               fit_pars <- do.call(mcml_optim,append(self$covariance$.__enclos_env__$private$D_data,
                                                                     list(as.matrix(self$covariance$Z),
                                                                          as.matrix(self$mean_function$X),
                                                                          y,
                                                                          dsamps,
                                                                          theta[parInds$cov],
                                                                          family=self$mean_function$family[[1]],
                                                                          link=self$mean_function$family[[2]],
                                                                          start = theta,
                                                                          trace=trace,
                                                                          mcnr = method=="mcnr",
                                                                          importance = sim_lik_step)))
                               
                               theta[parInds$b] <-  drop(fit_pars$beta)
                               if(self$mean_function$family[[1]] == "gaussian")theta[parInds$sig] <- fit_pars$sigma
                               theta[parInds$cov] <- drop(fit_pars$theta)
                                 
                               if(verbose)cat("\ntheta:",theta[all_pars])
                             }
                             
                             not_conv <- iter >= max.iter|any(abs(theta-thetanew)>tol)
                             if(not_conv&!no_warnings)warning(paste0("algorithm not converged. Max. difference between iterations :",max(abs(theta-thetanew)),". Suggest 
                                                 increasing m, or trying a different algorithm."))
                             
                             # if(sim_lik_step){
                             #   if(verbose)cat("\n\n")
                             #   if(verbose)message("Optimising simulated likelihood")
                             #   newtheta <- do.call(f_lik_optim,append(self$covariance$.__enclos_env__$private$D_data,
                             #                                          list(as.matrix(self$covariance$Z),
                             #                                               as.matrix(self$mean_function$X),
                             #                                               y,
                             #                                               dsamps,
                             #                                               theta[parInds$cov],
                             #                                               family=self$mean_function$family[[1]],
                             #                                               link=self$mean_function$family[[2]],
                             #                                               start = theta[all_pars],
                             #                                               lower = c(rep(-Inf,P),rep(1e-6,length(all_pars)-P)),
                             #                                               upper = c(rep(Inf,P),upper),
                             #                                               trace=trace)))
                             #   theta[all_pars] <- newtheta
                             # }
                             
                             if(verbose)cat("\n\nCalculating standard errors...")
                             
                             if(family%in%c("gaussian")){
                               mf_pars <- theta[c(parInds$b,parInds$sig)]
                               mf_pars_names <- c(colnames(self$mean_function$X),"sigma")
                               #upper <- c(upper,Inf)
                             } else {
                               mf_pars <- theta[c(parInds$b)]
                               mf_pars_names <- colnames(self$mean_function$X)
                             }
                             
                             
                             cov_nms <- as.character(unlist(rev(self$covariance$.__enclos_env__$private$flist)))
                             cov_idx <- unique(self$covariance$.__enclos_env__$private$D_data$N_par)
                             cov_pars_freq <- rep(0,length(cov_nms))
                             for(b in 1:length(cov_nms)){
                               if(b < length(cov_nms)){
                                 cov_pars_freq[b] <- cov_idx[b+1,1] - cov_idx[b,1]
                               } else {
                                 cov_pars_freq[b] <- length(parInds$cov) - cov_idx[b,1]
                               }
                               
                             }
                             cov_pars_names <- rep(cov_nms,cov_pars_freq)
                             permutation <- FALSE
                             robust <- FALSE
                             if(se.method=="lik"|se.method=="robust"|se.method=="approx"){
                               if(se.method=="lik"|se.method=="robust"){
                                 if(verbose&!robust)cat("using Hessian\n")
                                 if(verbose&robust)cat("using robust sandwich estimator\n")
                                 # hess <- tryCatch(do.call(f_lik_hess,append(self$covariance$.__enclos_env__$private$D_data,
                                 #                                            list(as.matrix(self$covariance$Z),
                                 #                                                 as.matrix(self$mean_function$X),
                                 #                                                 y,
                                 #                                                 dsamps,
                                 #                                                 theta[parInds$cov],
                                 #                                                 family=self$mean_function$family[[1]],
                                 #                                                 link=self$mean_function$family[[2]],
                                 #                                                 start = theta[all_pars],
                                 #                                                 lower = c(rep(-Inf,P),rep(1e-6,length(all_pars)-P)),
                                 #                                                 upper = c(rep(Inf,P),upper),
                                 #                                                 tol=fd_tol,importance = TRUE))),
                                 #                  error=function(e)NULL)
                                 
                                 hess <- tryCatch(do.call(mcml_hess,append(self$covariance$.__enclos_env__$private$D_data,
                                                                           list(as.matrix(self$covariance$Z),
                                                                                as.matrix(self$mean_function$X),
                                                                                y,
                                                                                dsamps,
                                                                                theta[parInds$cov],
                                                                                family=self$mean_function$family[[1]],
                                                                                link=self$mean_function$family[[2]],
                                                                                start = theta,
                                                                                trace=trace,
                                                                                mcnr = method=="mcnr",
                                                                                importance = sim_lik_step))),
                                                  error=function(e)NULL)
                                 
                                 hessused <- TRUE
                                 semat <- tryCatch(Matrix::solve(-hess),error=function(e)NULL)
                                 
                                 if(se.method == "robust"&!is.null(semat)){
                                   hlist <- list()
                                   #identify the clustering and sum over independent clusters
                                   D_data <- self$covariance$.__enclos_env__$private$D_data
                                   gr_var <- apply(D_data$func_def,1,function(x)any(x==1))
                                   gr_count <- D_data$N_dim
                                   gr_id <- which(gr_count == min(gr_count[gr_var]))
                                   gr_cov_var <- D_data$cov_data[1:D_data$N_dim[gr_id],
                                                                 1:D_data$N_var_func[gr_id,which(D_data$func_def[gr_id,]==1)],gr_id,drop=FALSE]
                                   gr_cov_var <- as.data.frame(gr_cov_var)
                                   gr_var_id <- which(rev(self$covariance$.__enclos_env__$private$flistvars)[[gr_id]]$funs=="gr")
                                   gr_cov_names <- rev(self$covariance$.__enclos_env__$private$flistvars)[[gr_id]]$rhs[
                                     rev(self$covariance$.__enclos_env__$private$flistvars)[[gr_id]]$groups==gr_var_id]
                                   colnames(gr_cov_var) <- gr_cov_names
                                   Z_in <- match_rows(self$covariance$data,as.data.frame(gr_cov_var),by=colnames(gr_cov_var))
                                   
                                   for(i in 1:ncol(Z_in)){
                                     id_in <- which(Z_in[,i]==1)
                                     g1 <- matrix(0,nrow=length(all_pars),ncol=1)
                                     # g1 <- do.call(f_lik_grad,append(self$covariance$.__enclos_env__$private$D_data,
                                     #                                 list(as.matrix(self$covariance$Z)[id_in,,drop=FALSE],
                                     #                                      as.matrix(self$mean_function$X)[id_in,,drop=FALSE],
                                     #                                      y[id_in],
                                     #                                      dsamps,
                                     #                                      theta[parInds$cov],
                                     #                                      family=self$mean_function$family[[1]],
                                     #                                      link=self$mean_function$family[[2]],
                                     #                                      start = theta[all_pars],
                                     #                                      lower = c(rep(-Inf,P),rep(1e-5,length(all_pars)-P)),
                                     #                                      upper = c(rep(Inf,P),upper),
                                     #                                      tol=fd_tol)))
                                     g1 <- do.call(f_hess,append(self$covariance$.__enclos_env__$private$D_data,
                                                                 list(as.matrix(self$covariance$Z)[id_in,,drop=FALSE],
                                                                      as.matrix(self$mean_function$X)[id_in,,drop=FALSE],
                                                                      y[id_in],
                                                                      dsamps,
                                                                      theta[parInds$cov],
                                                                      family=self$mean_function$family[[1]],
                                                                      link=self$mean_function$family[[2]],
                                                                      start = theta[all_pars],
                                                                      lower = c(rep(-Inf,P),rep(1e-5,length(all_pars)-P)),
                                                                      upper = c(rep(Inf,P),upper),
                                                                      tol=fd_tol)))
                                     
                                     hlist[[i]] <- g1%*%t(g1)
                                   }
                                   g0 <- Reduce('+',hlist)
                                   semat <- semat%*%g0%*%semat
                                   robust <- TRUE
                                 }
                                 
                                 if(!is.null(semat)){
                                   SE <- tryCatch(sqrt(Matrix::diag(semat)),
                                                  error=function(e)rep(NA,length(mf_pars)+length(cov_pars_names)),
                                                  warning=function(e)rep(NA,length(mf_pars)+length(cov_pars_names)))
                                 } else {
                                   SE <- rep(NA,length(mf_pars)+length(cov_pars_names))
                                 }
                               }
                               
                               if(se.method=="approx" || any(is.na(SE[1:P]))){
                                 SE <- rep(NA,length(mf_pars)+length(cov_pars_names))
                                 #if(!no_warnings&se.method!="approx")warning("Hessian was not positive definite, using approximation")
                                 #if(verbose&se.method=="approx")cat("using approximation\n")
                                 hessused <- FALSE
                                 self$check(verbose=FALSE)
                                 invM <- Matrix::solve(self$information_matrix())
                                 if(!robust){
                                   SE[1:P] <- sqrt(Matrix::diag(invM))
                                 } else {
                                   xb <-self$mean_function$X%*%theta[parInds$b] 
                                   XSyXb <- Matrix::t(self$mean_function$X)%*%Matrix::solve(self$Sigma)%*%(y - xb)
                                   robSE <- invM %*% XSyXb %*% invM
                                   SE[1:P] <- sqrt(Matrix::diag(robSE))
                                 }
                               }
                               
                               res <- data.frame(par = c(mf_pars_names,cov_pars_names,paste0("d",1:Q)),
                                                 est = c(mf_pars,theta[parInds$cov],rowMeans(dsamps)),
                                                 SE=c(SE,apply(dsamps,1,sd)))
                               
                               res$lower <- res$est - qnorm(1-0.05/2)*res$SE
                               res$upper <- res$est + qnorm(1-0.05/2)*res$SE
                               
                             } else {
                               res <- data.frame(par = c(mf_pars_names,cov_pars_names),
                                                 est = c(mf_pars,theta[parInds$cov]),
                                                 SE=NA,
                                                 lower = NA,
                                                 upper =NA)
                               hessused <- FALSE
                               robust <- FALSE
                             }
                             
                             # if(se.method=="perm") {
                             #   if(verbose)cat("using permutational method\n")
                             #   permutation = TRUE
                             #   #get null model
                             #   # use parameters from fit above rather than null marginal model
                             #   perm_out <- self$permutation_test(y,
                             #                                     permutation.par,
                             #                                     start = theta[parInds$b][permutation.par],
                             #                                     nsteps = perm_ci_steps,
                             #                                     iter = perm_iter,
                             #                                     type = perm_type,
                             #                                     verbose= verbose,
                             #                                     parallel = perm_parallel)
                             #   tval <- qnorm(1-perm_out$p/2)
                             #   par <- theta[parInds$b][permutation.par]
                             #   se <- abs(par/tval)
                             #   se1 <- rep(NA,length(mf_pars))
                             #   se1[permutation.par] <- se
                             #   se2 <- rep(NA,length(parInds$cov))
                             #   ci1l <- ci1u <- rep(NA,length(mf_pars))
                             #   ci2l <- ci2u <- rep(NA,length(parInds$cov))
                             #   ci1l[permutation.par] <- perm_out$lower
                             #   ci1u[permutation.par] <- perm_out$upper
                             #   
                             #   res <- data.frame(par = c(mf_pars_names,cov_pars_names),
                             #                     est = c(mf_pars,theta[parInds$cov]),
                             #                     SE=c(se1,se2),
                             #                     lower=c(ci1l,ci2l),
                             #                     upper=c(ci1u,ci2u))
                             #   hessused <- FALSE
                             #   robust <- FALSE
                             # } else 
                             
                             rownames(dsamps) <- Reduce(c,rev(self$covariance$.__enclos_env__$private$flistlabs))
                             
                             ## model summary statistics
                             aic_data <- append(list(Z = as.matrix(self$covariance$Z),
                                                     X = as.matrix(self$mean_function$X),
                                                     y = y,
                                                     u = dsamps,
                                                     family = self$mean_function$family[[1]],
                                                     link=self$mean_function$family[[2]]), 
                                                self$covariance$.__enclos_env__$private$D_data)
                             aic <- do.call(aic_mcml,append(aic_data,list(beta_par = mf_pars,
                                                                          cov_par = theta[parInds$cov])))
                             
                             xb <- self$mean_function$X %*% theta[parInds$b]
                             zd <- self$covariance$Z %*% rowMeans(dsamps)
                             
                             wdiag <- gen_dhdmu(Matrix::drop(xb),
                                                family=self$mean_function$family[[1]],
                                                link = self$mean_function$family[[2]])
                             
                             if(self$mean_function$family[[1]]%in%c("gaussian","gamma")){
                               wdiag <- theta[parInds$sig] * wdiag
                             }
                             
                             total_var <- var(Matrix::drop(xb)) + var(Matrix::drop(zd)) + mean(wdiag)
                             condR2 <- (var(Matrix::drop(xb)) + var(Matrix::drop(zd)))/total_var
                             margR2 <- var(Matrix::drop(xb))/total_var
                             
                             
                             out <- list(coefficients = res,
                                         converged = !not_conv,
                                         method = method,
                                         hessian = hessused,
                                         robust = robust,
                                         permutation = permutation,
                                         m = m,
                                         tol = tol,
                                         sim_lik = sim_lik_step,
                                         aic = aic,
                                         Rsq = c(cond = condR2,marg=margR2),
                                         mean_form = as.character(self$mean_function$formula),
                                         cov_form = as.character(self$covariance$formula),
                                         family = self$mean_function$family[[1]],
                                         link = self$mean_function$family[[2]],
                                         re.samps = dsamps)
                             
                             class(out) <- "mcml"
                             
                             self$mean_function$parameters <- orig_par_b 
                             self$covariance$parameters <- orig_par_cov
                             #self$check(verbose=FALSE)
                             
                             return(out)
                           }
                         ))

#' Returns the file name and type for MCNR function
#' 
#' Returns the file name and type for MCNR function
#' 
#' @param family family object
#' @return list with filename and type
mcnr_family <- function(family){
  f1 <- family[[1]]
  link <- family[[2]]
  gaussian_list <- c("identity")
  binomial_list <- c("logit","log","identity","probit")
  poisson_list <- c("log")
  type <- which(get(paste0(f1,"_list"))==link)
  return(list(file = paste0("mcml_",f1,".stan"),type=type))
}