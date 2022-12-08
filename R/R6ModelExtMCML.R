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
#' where h is the link function. A Model in comprised of a \link[glmmrBase]{MeanFunction} object, which defines the family F, 
#' link function h, and fixed effects design matrix X, and a \link[glmmrBase]{Covariance} object, which defines Z and D. 
#' 
#' This class extends the \link[glmmrBase]{Model} class of the `glmmrBase` package by adding the member function `MCML()`, which 
#' provides Markov Chain Monte Carlo Maximum Likelihood model fitting. See \href{https://github.com/samuel-watson/glmmrBase/blob/master/README.md}{glmmrBase} for a 
#' detailed guide on model specification.
#' 
#' @return No return value, called for side effects.
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
                           #'The function also offers different methods of obtaining standard errors. One can generate
                           #'estimates from the estimated Hessian matrix (`se.method = 'lik'`). 
                           #'Or use approximate generalised least squares estimates based on the maximum likelihood 
                           #'estimates of the covariance parameters (`se.method = 'approx'`).
                           #'
                           #' Options for the MCMC sampler are set by changing the values in `self$mcmc_options`
                           #'
                           #'There are several options that can be specified to the function using the `options` argument. 
                           #'The options should be provided as a list, e.g. `options = list(method="mcnr")`. The possible options are:
                           #'* `no_warnings` TRUE (do not report any warnings) or FALSE (report warnings), default to FALSE
                           #' * `fd_tol` The tolerance of the first difference method to estimate the Hessian and Gradient, default 
                           #' is 1e-4.
                           #' * `trace` Level of detail to report in output: 0 = no detail (default), 1 & 2 = detailed return from BOBYQA
                           #'
                           #'@param y A numeric vector of outcome data
                           #'@param start Optional. A numeric vector indicating starting values for the MCML algorithm iterations. 
                           #'If this is not specified then the parameter values stored in the linked mean function object will be used.
                           #'@param se.method One of either `'lik'`, `'approx'`, `'perm'`, or `'none'`, see Details.
                           #'@param method The MCML algorithm to use, either `mcem` or `mcnr`, see Details. Default is `mcem`.
                           #'@param sim.lik.step Logical. Either TRUE (conduct a simulated likelihood step at the end of the algorithm), or FALSE (does
                           #'not do this step), defaults to FALSE.
                           #'@param verbose Logical indicating whether to provide detailed output, defaults to TRUE.
                           #'@param tol Numeric value, tolerance of the MCML algorithm, the maximum difference in parameter estimates 
                           #'between iterations at which to stop the algorithm.
                           #'@param max.iter Integer. The maximum number of iterations of the MCML algorithm.
                           #'@param sparse Logical indicating whether to use sparse matrix methods
                           #'@param usestan Logical whether to use Stan (through the package `cmdstanr`) for the MCMC sampling. If FALSE then 
                           #'the internal Hamiltonian Monte Carlo sampler will be used instead. We recommend Stan over the internal sampler as 
                           #'it generally produces a larger number of effective samplers per unit time, especially for more complex 
                           #'covariance functions.
                           #'@param options An optional list providing options to the algorithm, see Details.
                           #'@return A `mcml` object
                           #' @seealso \link[glmmrBase]{Model}, \link[glmmrBase]{Covariance}, \link[glmmrBase]{MeanFunction} 
                           #'@examples
                           #'\dontrun{
                           #'df <- nelder(~(cl(10)*t(5)) > ind(10))
                           #' df$int <- 0
                           #' df[df$cl > 5, 'int'] <- 1
                           #' # specify parameter values in the call for the data simulation below
                           #' des <- ModelMCML$new(
                           #'   covariance = list( formula = ~ (1|gr(cl)*ar1(t)),
                           #'                      parameters = c(0.25,0.7)), 
                           #'   mean = list(formula = ~ factor(t) + int - 1,
                           #'                         parameters = c(rep(0,5),-0.2)),
                           #'   data = df,
                           #'   family = stats::binomial()
                           #' )
                           #' ysim <- des$sim_data() # simulate some data from the model
                           #' fit1 <- des$MCML(y = ysim,method="mcnr",usestan=FALSE,verbose=FALSE)
                           #' #fits the models using MCEM and report detailed output
                           #' fit2 <- des$MCML(y = ysim,method="mcem")
                           #'  #adds a simulated likelihood step after the MCEM algorithm
                           #' fit4 <- des$MCML(y = ysim, sim.lik.step = TRUE)  
                           #' 
                           #' # to set better starting values we can use the update_parameters function 
                           #' des$update_parameters(mean = rep(0.5,6), cov = c(0.3,0.7))
                           #'}
                           #'@md
                           MCML = function(y,
                                           start,
                                           se.method = "approx",
                                           method = "mcnr",
                                           sim.lik.step = FALSE,
                                           verbose=TRUE,
                                           tol = 1e-2,
                                           max.iter = 30,
                                           sparse = FALSE,
                                           usestan = TRUE,
                                           options = list()){
                             
                             # checks
                             if(!se.method%in%c("lik","robust","approx"))stop("se.method should be 'perm', 'lik', 'robust', 'approx', or 'none'")
                             
                             #set options
                             if(!is(options,"list"))stop("options should be a list")
                             no_warnings <- ifelse("no_warnings"%in%names(options),options$no_warnings,FALSE)
                             #warmup_iter <- ifelse("warmup_iter"%in%names(options),options$warmup_iter,500)
                             fd_tol <- ifelse("fd_tol"%in%names(options),options$fd_tol,1e-4)
                             trace <- ifelse("trace"%in%names(options),options$trace,0)
                             #step_size <- ifelse("trace"%in%names(options),options$trace,0.015)
                             
                             ### DO SOME BASIC CHECKS ON Y TO MAKE SURE IT DOESN'T CAUSE ERROR!
                             if(self$mean_function$family[[1]]=="binomial"){
                               if(!all(y==0 | y==1))stop("y must be 0 or 1")
                             } else if(self$mean_function$family[[1]]=="poisson"){
                               if(any(y <0) || any(y%%1 != 0))stop("y must be integer >= 0")
                             } else if(self$mean_function$family[[1]]=="beta"){
                               if(any(y<0 || y>1))stop("y must be between 0 and 1")
                             } else if(self$mean_function$family[[1]]=="Gamma") {
                               if(any(y<=0))stop("y must be positive")
                             } else if(self$mean_function$family[[1]]=="gaussian" & self$mean_function$family[[2]]=="log"){
                               if(any(y<=0))stop("y must be positive")
                             }
                             
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
                             if(family%in%c("gaussian","Gamma","beta")){
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
                             niter <- self$mcmc_options$samps
                             Q = ncol(self$covariance$Z)
                             invfunc <- self$mean_function$family$linkinv
                             ## generate sparse matrix if sparse option
                             if(sparse){
                               D <- as(self$covariance$D,"dsCMatrix")
                               Ap <- D@p
                               Ai <- D@i
                               L <- Matrix::t(Matrix::chol(D))#SparseChol::LL_Cholesky(D)##
                               # print(L)
                             } else {
                               L <- Matrix::Matrix(self$covariance$get_chol_D())
                             }
                             
                             #parse family
                             file_type <- mcnr_family(self$mean_function$family)
                             
                             ## set up sampler
                             if(usestan){
                               if(!requireNamespace("cmdstanr")){
                                 stop("cmdstanr is required to use Stan for sampling. See https://mc-stan.org/cmdstanr/ for details on how to install.\n
                                    Set option usestan=FALSE to use the in-built MCMC sampler.")
                               } else {
                                 if(verbose)message("If this is the first time running this model, it will be compiled by cmdstan.")
                                 model_file <- system.file("stan",
                                                           file_type$file,
                                                           package = "glmmrMCML",
                                                           mustWork = TRUE)
                                 mod <- suppressMessages(cmdstanr::cmdstan_model(model_file))
                               }
                             }
                             
                             
                             Xb <- Matrix::drop(self$mean_function$X %*% theta[parInds$b])
                             data <- list(
                               N = self$n(),
                               Q = Q,
                               Xb = Xb,
                               Z = as.matrix(self$covariance$Z%*%L),
                               y = y,
                               sigma = theta[parInds$sig],
                               type=as.numeric(file_type$type)
                             )
                             ## ADD IN INTERNAL MALA SAMPLER
                             ## ADD SPARSE MALA SAMPLER
                             
                             if(usestan){
                               ## ALGORITHMS
                               while(any(abs(theta-thetanew)>tol)&iter <= max.iter){
                                 iter <- iter + 1
                                 if(verbose)cat("\nIter: ",iter,"\n",Reduce(paste0,rep("-",40)))
                                 if(trace==2)t1 <- Sys.time()
                                 thetanew <- theta
                                 data$Xb <-  Matrix::drop(self$mean_function$X %*% thetanew[parInds$b])
                                 data$Z <- as.matrix(self$covariance$Z%*%L)
                                 data$sigma <- thetanew[parInds$sig]
                                 
                                 
                                 capture.output(fit <- mod$sample(data = data,
                                                                  chains = 1,
                                                                  iter_warmup = self$mcmc_options$warmup,
                                                                  iter_sampling = self$mcmc_options$samps,
                                                                  refresh = 0),
                                                file=tempfile())
                                 dsamps <- fit$draws("gamma",format = "matrix")
                                 class(dsamps) <- "matrix"
                                 dsamps <- Matrix::Matrix(L %*% Matrix::t(dsamps)) #check this
                                 if(trace==2)t2 <- Sys.time()
                                 if(trace==2)cat("\nMCMC sampling took: ",t2-t1)
                                 
                                 ## ADD IN RSTAN FUNCTIONALITY ONCE PARALLEL METHODS AVAILABLE IN RSTAN
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
                                 
                                 if(sparse){
                                   fit_pars <- do.call(mcml_optim_sparse,append(self$covariance$get_D_data(),
                                                                                list(
                                                                                  eff_range = self$covariance$eff_range,
                                                                                  Ap=Ap,
                                                                                  Ai=Ai,
                                                                                  Z=as.matrix(self$covariance$Z),
                                                                                  X=as.matrix(self$mean_function$X),
                                                                                  y=y,
                                                                                  u=as.matrix(dsamps),
                                                                                  family=self$mean_function$family[[1]],
                                                                                  link=self$mean_function$family[[2]],
                                                                                  start = theta,
                                                                                  trace=trace,
                                                                                  mcnr = method=="mcnr"
                                                                                )))
                                 } else {
                                   fit_pars <- do.call(mcml_optim,append(self$covariance$get_D_data(),
                                                                         list(
                                                                           eff_range = self$covariance$eff_range,
                                                                           Z=as.matrix(self$covariance$Z),
                                                                           X=as.matrix(self$mean_function$X),
                                                                           y=y,
                                                                           u=as.matrix(dsamps),
                                                                           family=self$mean_function$family[[1]],
                                                                           link=self$mean_function$family[[2]],
                                                                           start = theta,
                                                                           trace=trace,
                                                                           mcnr = method=="mcnr"
                                                                         )))
                                 }
                                 
                                 theta[parInds$b] <-  drop(fit_pars$beta)
                                 if(self$mean_function$family[[1]] %in%c("gaussian","Gamma","beta"))theta[parInds$sig] <- fit_pars$sigma
                                 theta[parInds$cov] <- drop(fit_pars$theta)
                                 
                                 if(sparse){
                                   L <- SparseChol::sparse_L(fit_pars)
                                   L <- L%*%Matrix::Diagonal(x=sqrt(fit_pars$D))
                                 } else {
                                   L <- Matrix::Matrix(self$covariance$get_chol_D(thetanew[parInds$cov]))
                                 }
                                 if(trace==2)t3 <- Sys.time()
                                 if(trace==2)cat("\nModel fitting took: ",t3-t2)
                                 if(verbose){
                                   #cat("\ntheta:",theta[all_pars])
                                   cat("\nBeta: ", theta[parInds$b])
                                   cat("\nTheta: ", theta[parInds$cov])
                                   if(self$mean_function$family[[1]]%in%c("gaussian","Gamma","beta"))cat("\nSigma: ",theta[parInds$sig])
                                   cat("\nMax. diff: ", max(abs(theta-thetanew)))
                                   cat("\n",Reduce(paste0,rep("-",40)))
                                }
                               }
                               
                               not_conv <- iter >= max.iter|any(abs(theta-thetanew)>tol)
                               if(not_conv&!no_warnings)warning(paste0("algorithm not converged. Max. difference between iterations :",max(abs(theta-thetanew)),". Suggest 
                                                 increasing m, or trying a different algorithm."))
                               
                               if(sim.lik.step){
                                 if(verbose)cat("\n\n")
                                 if(verbose)message("Optimising simulated likelihood")
                                 if(sparse){
                                   newtheta <- do.call(mcml_simlik_sparse,append(self$covariance$get_D_data(),
                                                                                 list(
                                                                                   eff_range = self$covariance$eff_range,
                                                                                   Ap=Ap,
                                                                                   Ai=Ai,
                                                                                   Z=as.matrix(self$covariance$Z),
                                                                                   X=as.matrix(self$mean_function$X),
                                                                                   y=y,
                                                                                   u=as.matrix(dsamps),
                                                                                   family=self$mean_function$family[[1]],
                                                                                   link=self$mean_function$family[[2]],
                                                                                   start = theta,
                                                                                   trace=trace,
                                                                                   mcnr = method=="mcnr"
                                                                                 )))
                                 } else {
                                   newtheta <- do.call(mcml_simlik,append(self$covariance$get_D_data(),
                                                                          list(
                                                                            eff_range = self$covariance$eff_range,
                                                                            Z=as.matrix(self$covariance$Z),
                                                                            X=as.matrix(self$mean_function$X),
                                                                            y=y,
                                                                            u=as.matrix(dsamps),
                                                                            family=self$mean_function$family[[1]],
                                                                            link=self$mean_function$family[[2]],
                                                                            start = theta,
                                                                            trace=trace,
                                                                            mcnr = method=="mcnr"
                                                                          )))
                                 }
                                 
                                 theta[all_pars] <- newtheta
                               }
                               
                               
                             } else {
                               # use internal mcmc sampler
                               if(verbose)message("Using in-built MCMC sampler.")
                               # used for debugging
                               # args1 <<- append(self$covariance$get_D_data(),
                               #                  list(
                               #                    eff_range = self$covariance$eff_range,
                               #                    Z=as.matrix(self$covariance$Z),
                               #                    X=as.matrix(self$mean_function$X),
                               #                    y=y,
                               #                    family=self$mean_function$family[[1]],
                               #                    link=self$mean_function$family[[2]],
                               #                    start = theta,
                               #                    mcnr = method=="mcnr",
                               #                    m = self$mcmc_options$samps,
                               #                    maxiter = max.iter,
                               #                    warmup = self$mcmc_options$warmup,
                               #                    tol=tol,
                               #                    verbose = verbose,
                               #                    lambda = self$mcmc_options$lambda,
                               #                    trace=trace,
                               #                    refresh = self$mcmc_options$refresh,
                               #                    maxsteps = self$mcmc_options$maxsteps,
                               #                    target_accept = self$mcmc_options$target_accept
                               #                  ))
                               res <- do.call(mcml_full,append(self$covariance$get_D_data(),
                                                               list(
                                                                 eff_range = self$covariance$eff_range,
                                                                 Z=as.matrix(self$covariance$Z),
                                                                 X=as.matrix(self$mean_function$X),
                                                                 y=y,
                                                                 family=self$mean_function$family[[1]],
                                                                 link=self$mean_function$family[[2]],
                                                                 start = theta,
                                                                 mcnr = method=="mcnr",
                                                                 m = self$mcmc_options$samps,
                                                                 maxiter = max.iter,
                                                                 warmup = self$mcmc_options$warmup,
                                                                 tol=tol,
                                                                 verbose = verbose,
                                                                 lambda = self$mcmc_options$lambda,
                                                                 trace=trace,
                                                                 refresh = self$mcmc_options$refresh,
                                                                 maxsteps = self$mcmc_options$maxsteps,
                                                                 target_accept = self$mcmc_options$target_accept
                                                               )))
                               theta[parInds$b] <-  res$beta
                               if(self$mean_function$family[[1]] %in% c("gaussian","Gamma","beta"))theta[parInds$sig] <- res$sigma
                               theta[parInds$cov] <- res$theta
                               not_conv <- !res$converged
                               dsamps <- res$u
                             }
                             
                             
                             
                             if(verbose)cat("\n\nCalculating standard errors...")
                             fnpar <- c(1,1,1,2,2,1,2,2,2,2,2,2,2,1)
                             cov_nms <- as.character(unlist(rev(self$covariance$.__enclos_env__$private$flist)))
                             cov_pars_freq <- rep(0,length(cov_nms))
                             ddata <- self$covariance$get_D_data()
                             pidx <- 0
                             for(i in 1:length(cov_nms)){
                               bid <- min(ddata$cov[ddata$cov[,5]==pidx,1])
                               cov_pars_freq[i] <- sum(fnpar[ddata$cov[ddata$cov[,1]==bid,3]])
                               bid <- bid + cov_pars_freq[i]
                             }
                             
                             cov_pars_names <- rep(cov_nms,cov_pars_freq)
                             robust <- FALSE
                             if(se.method=="lik"|se.method=="robust"|se.method=="approx"){
                               if(se.method=="lik"|se.method=="robust"){
                                 if(verbose&!robust)cat("using Hessian\n")
                                 if(verbose&robust)cat("using robust sandwich estimator\n")
                                 if(sparse){
                                   hess <- do.call(mcml_hess_sparse,append(self$covariance$get_D_data(),
                                                                                 list(
                                                                                   eff_range = self$covariance$eff_range,
                                                                                   Ap=Ap,
                                                                                   Ai=Ai,
                                                                                   Z=as.matrix(self$covariance$Z),
                                                                                   X=as.matrix(self$mean_function$X),
                                                                                   y=y,
                                                                                   u=as.matrix(dsamps),
                                                                                   family=self$mean_function$family[[1]],
                                                                                   link=self$mean_function$family[[2]],
                                                                                   start = theta,
                                                                                   trace=trace
                                                                                 )))
                                 } else {
                                   newtheta <- do.call(mcml_hess,append(self$covariance$get_D_data(),
                                                                          list(
                                                                            eff_range = self$covariance$eff_range,
                                                                            Z=as.matrix(self$covariance$Z),
                                                                            X=as.matrix(self$mean_function$X),
                                                                            y=y,
                                                                            u=as.matrix(dsamps),
                                                                            family=self$mean_function$family[[1]],
                                                                            link=self$mean_function$family[[2]],
                                                                            start = theta,
                                                                            trace=trace
                                                                          )))
                                 }
                                 hessused <- TRUE
                                 semat <- tryCatch(Matrix::solve(hess),error=function(e)NULL)
                                 
                                 if(!is.null(semat)){
                                   SE <- tryCatch(sqrt(Matrix::diag(semat)),
                                                  error=function(e)rep(NA,P+R),
                                                  warning=function(e)rep(NA,P+R))
                                 } else {
                                   SE <- rep(NA,P+R)
                                 }
                               }
                                 
                              
                               if(se.method=="approx" || any(is.na(SE[1:P]))){
                                 SE <- rep(NA,P+R)
                                 #if(!no_warnings&se.method!="approx")warning("Hessian was not positive definite, using approximation")
                                 #if(verbose&se.method=="approx")cat("using approximation\n")
                                 hessused <- FALSE
                                 self$covariance$parameters <- theta[parInds$cov]
                                 if(family%in%c("gaussian")){
                                   orig_sigma <- self$var_par
                                   self$var_par <- theta[parInds$sig]
                                 }
                                 self$check(verbose=FALSE)
                                 invM <- Matrix::solve(self$information_matrix())
                                 self$covariance$parameters <- orig_par_cov
                                 if(family%in%c("gaussian")){
                                   self$var_par <- orig_sigma
                                 }
                                 if(!robust){
                                   SE[1:P] <- sqrt(Matrix::diag(invM))
                                 } else {
                                   xb <-self$mean_function$X%*%theta[parInds$b] 
                                   XSyXb <- Matrix::t(self$mean_function$X)%*%Matrix::solve(self$Sigma)%*%(y - xb)
                                   robSE <- invM %*% XSyXb %*% invM
                                   SE[1:P] <- sqrt(Matrix::diag(robSE))
                                 }
                               }
                               
                               if(family%in%c("gaussian")){
                                 #mf_pars <- theta[c(parInds$b,parInds$sig)]
                                 mf_pars_names <- c(colnames(self$mean_function$X),cov_pars_names,"sigma")
                                 SE <- c(SE,NA)
                               } else {
                                 #mf_pars <- theta[c(parInds$b)]
                                 mf_pars_names <- c(colnames(self$mean_function$X),cov_pars_names)
                               }
                               res <- data.frame(par = c(mf_pars_names,paste0("d",1:Q)),
                                                 est = c(theta[all_pars],rowMeans(dsamps)),
                                                 SE=c(SE,apply(dsamps,1,sd)))
                               
                               res$lower <- res$est - qnorm(1-0.05/2)*res$SE
                               res$upper <- res$est + qnorm(1-0.05/2)*res$SE
                               
                             } else {
                               res <- data.frame(par = c(mf_pars_names),
                                                 est = c(theta),
                                                 SE=NA,
                                                 lower = NA,
                                                 upper =NA)
                               hessused <- FALSE
                               robust <- FALSE
                             }
                             
                             
                             rownames(dsamps) <- Reduce(c,rev(self$covariance$.__enclos_env__$private$flistlabs))
                             ## model summary statistics
                             aic <- do.call(aic_mcml,append(self$covariance$get_D_data(),
                                                            list(
                                                              eff_range = self$covariance$eff_range,
                                                              Z = as.matrix(self$covariance$Z),
                                                              X = as.matrix(self$mean_function$X),
                                                              y = y,
                                                              u = as.matrix(dsamps),
                                                              family = self$mean_function$family[[1]],
                                                              link=self$mean_function$family[[2]],
                                                              beta_par = theta[mf_parInd],
                                                              cov_par = theta[parInds$cov]
                                                            )))
                             
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
                                         m = self$mean_function$samps,
                                         tol = tol,
                                         sim_lik = sim.lik.step,
                                         aic = aic,
                                         Rsq = c(cond = condR2,marg=margR2),
                                         mean_form = as.character(self$mean_function$formula),
                                         cov_form = as.character(self$covariance$formula),
                                         family = self$mean_function$family[[1]],
                                         link = self$mean_function$family[[2]],
                                         re.samps = dsamps,
                                         iter = iter)
                             
                             class(out) <- "mcml"
                             
                             self$mean_function$parameters <- orig_par_b 
                             self$covariance$parameters <- orig_par_cov
                             #self$check(verbose=FALSE)
                             
                             return(out)
                           },
                           #'@description
                           #'Maximum Likelihood model fitting with Laplace Approximation
                           #'
                           #'@details
                           #'**Laplace approximation**
                           #'Fits generalised linear mixed models using Laplace approximation. 
                           #'
                           #'@param y A numeric vector of outcome data
                           #'@param start Optional. A numeric vector indicating starting values for the model parameters. 
                           #'@param method String. Either "nloptim" for non-linear optimisation, or "nr" for Newton-Raphson algorithm
                           #'@param use.hess Logical indicating whether to estimate the standard errors using the Hessian (TRUE) or (approximate) information matrix (FALSE)
                           #'@param verbose logical indicating whether to provide detailed algorithm feedback.
                           #'@return A `mcml` object
                           #' @seealso \link[glmmrBase]{Model}, \link[glmmrBase]{Covariance}, \link[glmmrBase]{MeanFunction} 
                           #'@examples
                           #'\dontrun{
                           #'df <- nelder(~(cl(10)*t(5)) > ind(10))
                           #' df$int <- 0
                           #' df[df$cl > 5, 'int'] <- 1
                           #' # specify parameter values in the call for the data simulation below
                           #' des <- ModelMCML$new(
                           #'   covariance = list( formula = ~ (1|gr(cl)*ar1(t)),
                           #'                      parameters = c(0.25,0.7)), 
                           #'   mean = list(formula = ~ factor(t) + int - 1,
                           #'                         parameters = c(rep(0,5),-0.2)),
                           #'   data = df,
                           #'   family = stats::binomial()
                           #' )
                           #' ysim <- des$sim_data() # simulate some data from the model
                           #' fit1 <- des$LA(y = ysim)
                           #'}
                           #'@md
                           LA = function(y,
                                         start,
                                         method = "nloptim",
                                         use.hess = FALSE,
                                         verbose = FALSE){
                             
                             if(!method%in%c("nloptim","nr"))stop("method should be either nr or nloptim")
                             trace <- ifelse(verbose,1,0)
                             ### DO SOME BASIC CHECKS ON Y TO MAKE SURE IT DOESN'T CAUSE ERROR!
                             if(self$mean_function$family[[1]]=="binomial"){
                               if(!all(y==0 | y==1))stop("y must be 0 or 1")
                             } else if(self$mean_function$family[[1]]=="poisson"){
                               if(any(y <0) || any(y%%1 != 0))stop("y must be integer >= 0")
                             } else if(self$mean_function$family[[1]]=="beta"){
                               if(any(y<0 || y>1))stop("y must be between 0 and 1")
                             } else if(self$mean_function$family[[1]]=="Gamma") {
                               if(any(y<=0))stop("y must be positive")
                             } else if(self$mean_function$family[[1]]=="gaussian" & self$mean_function$family[[2]]=="log"){
                               if(any(y<=0))stop("y must be positive")
                             }
                             
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
                             if(family%in%c("gaussian","Gamma","beta")){
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
                             
                             if(verbose)cat("\nStart: ",start[all_pars],"\n")
                             
                             Q = ncol(self$covariance$Z)
                             invfunc <- self$mean_function$family$linkinv
                             if(method=="nloptim"){
                               resb <- do.call(mcml_la,append(self$covariance$get_D_data(),
                                                              list(
                                                                eff_range = self$covariance$eff_range,
                                                                Z=as.matrix(self$covariance$Z),
                                                                X=as.matrix(self$mean_function$X),
                                                                y=y,
                                                                family=self$mean_function$family[[1]],
                                                                link=self$mean_function$family[[2]],
                                                                start = theta,
                                                                usehess = TRUE,
                                                                tol = 1e-2,
                                                                verbose = verbose,
                                                                trace = trace
                                                              )))
                             } else {
                               resb <- do.call(mcml_la_nr,append(self$covariance$get_D_data(),
                                                              list(
                                                                eff_range = self$covariance$eff_range,
                                                                Z=as.matrix(self$covariance$Z),
                                                                X=as.matrix(self$mean_function$X),
                                                                y=y,
                                                                family=self$mean_function$family[[1]],
                                                                link=self$mean_function$family[[2]],
                                                                start = theta,
                                                                usehess = FALSE,
                                                                tol = 1e-2,
                                                                verbose = verbose,
                                                                trace = trace
                                                              )))
                             }
                             
                             
                             theta[parInds$b] <-  resb$beta
                             if(self$mean_function$family[[1]] %in% c("gaussian","Gamma","beta"))theta[parInds$sig] <- resb$sigma
                             theta[parInds$cov] <- resb$theta
                             
                             if(verbose)cat("\n\nCalculating standard errors...")
                             fnpar <- c(1,1,1,2,2,1,2,2,2,2,2,2,2,1)
                             cov_nms <- as.character(unlist(rev(self$covariance$.__enclos_env__$private$flist)))
                             cov_pars_freq <- rep(0,length(cov_nms))
                             ddata <- self$covariance$get_D_data()
                             pidx <- 0
                             for(i in 1:length(cov_nms)){
                               bid <- min(ddata$cov[ddata$cov[,5]==pidx,1])
                               cov_pars_freq[i] <- sum(fnpar[ddata$cov[ddata$cov[,1]==bid,3]])
                               bid <- bid + cov_pars_freq[i]
                             }
                             
                             cov_pars_names <- rep(cov_nms,cov_pars_freq)
                             robust <- FALSE
                             
                             SE <- rep(NA,P+R)
                             #if(!no_warnings&se.method!="approx")warning("Hessian was not positive definite, using approximation")
                             #if(verbose&se.method=="approx")cat("using approximation\n")
                             hessused <- usehess#FALSE
                             if(!use.hess){
                               self$covariance$parameters <- theta[parInds$cov]
                               if(family%in%c("gaussian")){
                                 orig_sigma <- self$var_par
                                 self$var_par <- theta[parInds$sig]
                               }
                               self$check(verbose=FALSE)
                               invM <- Matrix::solve(self$information_matrix())
                               self$covariance$parameters <- orig_par_cov
                               if(family%in%c("gaussian")){
                                 self$var_par <- orig_sigma
                               }
                               if(!robust){
                                 SE[1:P] <- sqrt(Matrix::diag(invM))
                               } else {
                                 xb <-self$mean_function$X%*%theta[parInds$b] 
                                 XSyXb <- Matrix::t(self$mean_function$X)%*%Matrix::solve(self$Sigma)%*%(y - xb)
                                 robSE <- invM %*% XSyXb %*% invM
                                 SE[1:P] <- sqrt(Matrix::diag(robSE))
                               }
                               
                               if(family%in%c("gaussian","Gamma","beta")){
                                 #mf_pars <- theta[c(parInds$b,parInds$sig)]
                                 mf_pars_names <- c(colnames(self$mean_function$X),cov_pars_names,"sigma")
                                 SE <- c(SE,NA)
                               } else {
                                 #mf_pars <- theta[c(parInds$b)]
                                 mf_pars_names <- c(colnames(self$mean_function$X),cov_pars_names)
                               }
                               
                               res <- data.frame(par = c(mf_pars_names,paste0("d",1:Q)),
                                                 est = c(theta[all_pars],resb$u),
                                                 SE=c(SE,rep(NA,length(resb$u))))
                             } else {
                               res <- data.frame(par = c(mf_pars_names,paste0("d",1:Q)),
                                                 est = c(theta[all_pars],resb$u),
                                                 SE=c(resb$se,rep(NA,length(resb$u))))
                             }
                             
                             
                             res$lower <- res$est - qnorm(1-0.05/2)*res$SE
                             res$upper <- res$est + qnorm(1-0.05/2)*res$SE
                             
                             
                             rownames(resb$u) <- Reduce(c,rev(self$covariance$.__enclos_env__$private$flistlabs))
                             ## model summary statistics
                             aic <- do.call(aic_mcml,append(self$covariance$get_D_data(),
                                                            list(
                                                              eff_range = self$covariance$eff_range,
                                                              Z = as.matrix(self$covariance$Z),
                                                              X = as.matrix(self$mean_function$X),
                                                              y = y,
                                                              u = as.matrix(resb$u),
                                                              family = self$mean_function$family[[1]],
                                                              link=self$mean_function$family[[2]],
                                                              beta_par = theta[mf_parInd],
                                                              cov_par = theta[parInds$cov]
                                                            )))
                             
                             xb <- self$mean_function$X %*% theta[parInds$b]
                             zd <- self$covariance$Z %*% resb$u
                             
                             wdiag <- gen_dhdmu(Matrix::drop(xb),
                                                family=self$mean_function$family[[1]],
                                                link = self$mean_function$family[[2]])
                             
                             if(self$mean_function$family[[1]]%in%c("gaussian","Gamma","beta")){
                               wdiag <- theta[parInds$sig] * wdiag
                             }
                             
                             total_var <- var(Matrix::drop(xb)) + var(Matrix::drop(zd)) + mean(wdiag)
                             condR2 <- (var(Matrix::drop(xb)) + var(Matrix::drop(zd)))/total_var
                             margR2 <- var(Matrix::drop(xb))/total_var
                             
                             
                             out <- list(coefficients = res,
                                         converged = TRUE,
                                         method = method,
                                         hessian = FALSE,
                                         m = NA,
                                         tol = NA,
                                         sim_lik = FALSE,
                                         aic = aic,
                                         Rsq = c(cond = condR2,marg=margR2),
                                         mean_form = as.character(self$mean_function$formula),
                                         cov_form = as.character(self$covariance$formula),
                                         family = self$mean_function$family[[1]],
                                         link = self$mean_function$family[[2]],
                                         re.samps = resb$u,
                                         iter = 0)
                             
                             class(out) <- "mcml"
                             
                             self$mean_function$parameters <- orig_par_b 
                             self$covariance$parameters <- orig_par_cov
                             #self$check(verbose=FALSE)
                             
                             return(out)
                             
                           },
                           #' @field mcmc_options There are five options for MCMC sampling that are specified in this list:
                           #' * `warmup` The number of warmup iterations. Note that if using the internal HMC
                           #' sampler, this only applies to the first iteration of the MCML algorithm, as the 
                           #' values from the previous iteration are carried over.
                           #' * `samps` The number of MCMC samples drawn in the MCML algorithm. For 
                           #' smaller tolerance values larger numbers of samples are required. For the internal
                           #' HMC sampler, larger numbers of samples are generally required than if using Stan since
                           #' the samples generally exhibit higher autocorrealtion, especially for more complex 
                           #' covariance structures.
                           #' * `lambda` (Only relevant for the internal HMC sampler) Value of the trajectory length of the leapfrog integrator in Hamiltonian Monte Carlo
                           #'  (equal to number of steps times the step length). Larger values result in lower correlation in samples, but
                           #'  require larger numbers of steps and so is slower. Smaller numbers are likely required for non-linear GLMMs.
                           #'  * `refresh` How frequently to print to console MCMC progress if displaying verbose output.
                           #'  * `maxsteps` (Only relevant for the internal HMC sampler) Integer. The maximum number of steps of the leapfrom integrator
                           mcmc_options = list(warmup = 500, 
                                               samps = 250, 
                                               lambda = 5,
                                               refresh = 500,
                                               maxsteps = 100,
                                               target_accept = 0.95)
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