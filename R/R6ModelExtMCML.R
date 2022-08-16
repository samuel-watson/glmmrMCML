library(glmmrBase)
Method$set("public","MCML",function(y,
                                    start,
                                    se.method = "lik",
                                    method = "mcnr",
                                    permutation.par,
                                    verbose=TRUE,
                                    tol = 1e-2,
                                    m=100,
                                    max.iter = 30,
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
      start <- c(self$mean_function$parameters,unlist(self$covariance$parameters),self$var_par)
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
      start <- c(self$mean_function$parameters,unlist(self$covariance$parameters))
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
  if(use_cmdstanr){
    if(!requireNamespace("cmdstanr"))stop("cmdstanr not available")
    model_file <- system.file("stan",
                              file_type$file,
                              package = "glmmr",
                              mustWork = TRUE)
    mod <- cmdstanr::cmdstan_model(model_file)
    
  }
  
  ## ALGORITHMS
  while(any(abs(theta-thetanew)>tol)&iter <= max.iter){
    iter <- iter + 1
    if(verbose)cat("\nIter: ",iter,": ")
    thetanew <- theta
    
    
    Xb <- Matrix::drop(self$mean_function$X %*% thetanew[parInds$b])
    
    data <- list(
      N = self$n(),
      P = P,
      Q = Q,
      Xb = Xb,
      L = as.matrix(Matrix::t(Matrix::chol(self$covariance$D))),
      Z = as.matrix(self$covariance$Z),
      y = y,
      sigma = thetanew[parInds$sig],
      type=as.numeric(file_type$type)
    )
    
    if(use_cmdstanr){
      capture.output(fit <- mod$sample(data = data,
                                       chains = 1,
                                       iter_warmup = warmup_iter,
                                       iter_sampling = m,
                                       refresh = 0),
                     file=tempfile())
      dsamps <- fit$draws("gamma")
      dsamps <- matrix(dsamps[,1,],ncol=Q)
    } else {
      capture.output(suppressWarnings(fit <- rstan::sampling(stanmodels[[gsub(".stan","",file_type$file)]],
                                                             data = data,
                                                             chains = 1,
                                                             warmup = warmup_iter,
                                                             iter = warmup_iter+m)))
      dsamps <- rstan::extract(fit,"gamma",permuted=FALSE)
      dsamps <- matrix(dsamps[,1,],ncol=Q)
      dsamps <- t(dsamps)
    }
    
    if(method == "mcnr"){
      start_pars <- theta[c(parInds$b,parInds$cov)]
    } else if(method == "mcem"){
      start_pars <- theta[all_pars]
    }
    
    fit_pars <- do.call(f_lik_optim,append(self$covariance$.__enclos_env__$private$D_data,
                                           list(as.matrix(self$covariance$Z),
                                                as.matrix(self$mean_function$X),
                                                y,
                                                dsamps,
                                                theta[parInds$cov],
                                                family=self$mean_function$family[[1]],
                                                link=self$mean_function$family[[2]],
                                                start = start_pars,
                                                lower_b = rep(-Inf,P+1),
                                                upper_b = rep(Inf,P+1),
                                                lower_t = rep(1e-6,length(all_pars)-P),
                                                upper_t = upper,
                                                trace=trace,
                                                mcnr = method=="mcnr",
                                                importance = sim_lik_step)))
    # BETA PARAMETERS STEP
    if(method == "mcnr"){
      theta[parInds$b] <-  drop(fit_pars$beta)
      theta[parInds$sig] <- fit_pars$sigma
    } else if(method == "mcem"){
      theta[mf_parInd] <- drop(fit_pars$beta)
    }
    theta[parInds$cov] <- drop(fit_pars$theta)
    
    # # BETA PARAMETERS STEP
    # if(method == "mcnr"){
    #   beta_step <- mcnr_step(y = y,
    #                          X= as.matrix(self$mean_function$X),
    #                          Z = as.matrix(self$covariance$Z),
    #                          beta = theta[parInds$b],
    #                          u = dsamps,
    #                          family = self$mean_function$family[[1]],
    #                          link = self$mean_function$family[[2]])
    #   
    #   theta[parInds$b] <-  theta[parInds$b] + beta_step$beta_step
    #   theta[parInds$sig] <- beta_step$sigmahat
    #   
    #   
    # } else if(method == "mcem"){
    #   theta[mf_parInd] <- drop(l_lik_optim(as.matrix(self$covariance$Z),
    #                                        as.matrix(self$mean_function$X),
    #                                        y,
    #                                        dsamps,
    #                                        family=self$mean_function$family[[1]],
    #                                        link=self$mean_function$family[[2]],
    #                                        start = theta[mf_parInd],
    #                                        lower = rep(-Inf,length(mf_parInd)),
    #                                        upper = rep(Inf,length(mf_parInd)),
    #                                        trace= trace))
    #   
    # }
    
    
    # COVARIANCE PARAMETERS STEP
    # if(!skip_cov_optim){
    #   upper <- rep(Inf,length(parInds$cov))
    #   # if any are ar1, then need to set upper limit of 1
    #   if(any(c(t(self$covariance$.__enclos_env__$private$D_data$func_def))==3)){
    #     id3 <- which(rep(c(t(self$covariance$.__enclos_env__$private$D_data$func_def)),c(t(self$covariance$.__enclos_env__$private$D_data$N_par)))==3)
    #     upper[id3] <- 1
    #   }
    #   
    #   newtheta <- do.call(d_lik_optim,append(self$covariance$.__enclos_env__$private$D_data,
    #                                          list(u = dsamps,
    #                                               start = c(theta[parInds$cov]),
    #                                               lower= rep(1e-6,length(parInds$cov)),
    #                                               upper= upper,
    #                                               trace=trace)))
    #   theta[parInds$cov] <- drop(newtheta)
    # }
    
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
    upper <- c(upper,Inf)
  } else {
    mf_pars <- theta[c(parInds$b)]
    mf_pars_names <- colnames(self$mean_function$X)
  }
  
  cov_pars_names <- rep(as.character(unlist(rev(self$covariance$.__enclos_env__$private$flist))),
                        rowSums(self$covariance$.__enclos_env__$private$D_data$N_par))#paste0("cov",1:R)
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
                                                       start = start_pars,
                                                       lower_b = rep(-Inf,P+1),
                                                       upper_b = rep(Inf,P+1),
                                                       lower_t = rep(1e-6,length(all_pars)-P),
                                                       upper_t = upper,
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
      invM <- Matrix::solve(private$information_matrix())
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
    
  } else if(se.method=="perm") {
    if(verbose)cat("using permutational method\n")
    permutation = TRUE
    #get null model
    # use parameters from fit above rather than null marginal model
    perm_out <- self$permutation_test(y,
                                      permutation.par,
                                      start = theta[parInds$b][permutation.par],
                                      nsteps = perm_ci_steps,
                                      iter = perm_iter,
                                      type = perm_type,
                                      verbose= verbose,
                                      parallel = perm_parallel)
    tval <- qnorm(1-perm_out$p/2)
    par <- theta[parInds$b][permutation.par]
    se <- abs(par/tval)
    se1 <- rep(NA,length(mf_pars))
    se1[permutation.par] <- se
    se2 <- rep(NA,length(parInds$cov))
    ci1l <- ci1u <- rep(NA,length(mf_pars))
    ci2l <- ci2u <- rep(NA,length(parInds$cov))
    ci1l[permutation.par] <- perm_out$lower
    ci1u[permutation.par] <- perm_out$upper
    
    res <- data.frame(par = c(mf_pars_names,cov_pars_names),
                      est = c(mf_pars,theta[parInds$cov]),
                      SE=c(se1,se2),
                      lower=c(ci1l,ci2l),
                      upper=c(ci1u,ci2u))
    hessused <- FALSE
    robust <- FALSE
  } else {
    res <- data.frame(par = c(mf_pars_names,cov_pars_names),
                      est = c(mf_pars,theta[parInds$cov]),
                      SE=NA,
                      lower = NA,
                      upper =NA)
    hessused <- FALSE
    robust <- FALSE
  }
  
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
})

Method$set("public","permutation_test",function(y,
                                                permutation.par,
                                                start,
                                                iter = 1000,
                                                nsteps=1000,
                                                type="cov",
                                                parallel = TRUE,
                                                verbose=TRUE){
  if(is.null(self$mean_function$randomise))
    stop("random allocations are created using the function
                                                  in self$mean_function$randomise, but this has not
                                                  been set. Please see help(MeanFunction) for more
                                                  details")
  
  if(verbose&type=="cov")
    message("using covariance weighted statistic, to change
                                                    permutation statistic set option perm_type, see
                                                    details in help(Design)")
  if(verbose&type=="unw")
    message("using unweighted statistic, to change permutation
                                                     statistic set option perm_type, see details in
                                                     help(Design)")
  
  if(verbose){
    pbapply::pboptions(type="timer")
  } else {
    pbapply::pboptions(type="none")
  }
  
  Xnull <- as.matrix(self$mean_function$X)
  Xnull <- Xnull[,-permutation.par]
  
  family <- self$mean_function$family
  unless.null <- function(x, if.null) if(is.null(x)) if.null else x
  valideta    <- unless.null(family$valideta, function(eta) TRUE)
  validmu     <- unless.null(family$validmu,  function(mu)  TRUE)
  
  weights  = rep(1, NROW(y))
  offset   = rep(0, NROW(y))
  null_fit <- myglm(Xnull,y, weights, offset, family)
  xb <- null_fit$linear.predictors
  
  tr <- self$mean_function$X[, permutation.par]
  if(any(!tr%in%c(0,1)))stop("permuational inference only available for dichotomous treatments")
  #tr[tr==0] <- -1
  
  w.opt <- type=="cov"
  if(w.opt) {
    invS <- Matrix::solve(self$Sigma)
  } else {
    invS <- 1
  }
  
  tr_mat <- pbapply::pbsapply(1:iter,function(i){
    self$mean_function$randomise()
  })
  
  ############
  # Will the resid ever change?
  ypred <- self$mean_function$family$linkinv(Matrix::drop(xb))
  resids <- Matrix::Matrix(y-ypred)
  family2 <- self$mean_function$family[[2]]
  dtr <- tr
  dtr[dtr==0] <- -1
  qstat <- qscore_impl(as.vector(resids),dtr,xb,as.matrix(invS),family2,w.opt)
  qtest <- as.vector(permutation_test_impl(as.vector(resids),
                                           tr_mat, xb, as.matrix(invS),
                                           family2, w.opt, iter, verbose))
  ############
  
  #permutation confidence intervals
  if(verbose)
    cat("Starting permutational confidence intervals\n")
  pval <- length(qtest[qtest>qstat])/iter
  #print(pval)
  if(pval==0)pval <- 0.5/iter
  tval <- qnorm(1-pval/2)
  par <- start#theta[parInds$b][permutation.par]
  se <- abs(par/tval)
  
  tr_mat <- pbapply::pbsapply(1:nsteps,function(i){
    self$mean_function$randomise()
  })
  
  if(verbose)cat("Lower\n")
  lower <- confint_search(start = par - 2*0.2,
                          b = start,
                          Xnull,
                          y,
                          tr,
                          new_tr_mat = as.matrix(tr_mat),
                          xb,
                          as.matrix(invS),
                          family,
                          family2,
                          nsteps,
                          w.opt,
                          alpha = 0.05,
                          verbose = verbose)
  if(verbose)cat("\nUpper\n")
  upper <- confint_search(start = par + 2*0.2,
                          b = start,
                          Xnull,
                          y,
                          tr,
                          new_tr_mat = tr_mat,
                          xb,
                          as.matrix(invS),
                          family,
                          family2,
                          nsteps,
                          w.opt,
                          alpha = 0.05,
                          verbose = verbose)
  
  return(list(p=pval,lower=lower,upper=upper))
})

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
#'@export
MCML <- function(y,
                 start,
                 se.method = "lik",
                 method = "mcnr",
                 permutation.par,
                 verbose=TRUE,
                 tol = 1e-2,
                 m=100,
                 max.iter = 30,
                 options = list()){
  return(NULL)
}

#'Conducts a permuation test
#'
#' Estimates p-values and confidence intervals using a permutation test
#'
#'@details
#'**Permutation tests**
#' If the user provided a re-randomisation function to the linked mean function object (see \link[glmmr]{MeanFunction}),
#' then a permuation test can be conducted. A new random assignment is generated on each iteration of the permutation test.
#' The test statistic can be either a quasi-score statistic, weighting the observations using the covariance matrix (`type="cov"`),
#' or an unweighted statistic that weights each observation in each cluster equally (`type="unw"`). The 1-alpha% 
#' confidence interval limits are estimated using an efficient iterative stochastic search procedure. On each step of the algorithm
#' a single permuation and test statistic is generated, and the current estimate of the confidence interval limit either 
#' increased or decreased depedning on its value. The procedure converges in probability to the true limits, see Watson et al (2021)
#' and Garthwaite (1996).
#' 
#'@param y Numeric vector of outcome data
#'@param permutation.par Integer indicator which parameter to conduct a permutation
#'test for. Refers to a column of the X matrix.
#'@param start Value of the parameter. Used both as a starting value for the algorithms
#'and as a best estimate for the confidence interval search procedure.
#'@param iter Integer. Number of iterations of the permuation test to conduct
#'@param nsteps Integer. Number of steps of the confidence interval search procedure
#'@param type Either `cov` for a test statistic weighted by the covariance matrix, or 
#'`unw` for an unweighted test statistic. See Details.
#'@param parallel Logical indicating whether to run the tests in parallel
#'@param verbose Logical indicating whether to report detailed output
#'@return A list with the estimated p-value and the estimated lower and upper 95% confidence interval
#' @references 
#' Watson et al. Arxiv
#' Braun and Feng
#' Gail
#' Garthwaite
#'@examples
#'\dontrun{
#'  df <- nelder(~(cl(6)*t(5)) > ind(5))
#'  df$int <- 0
#'  df[df$cl > 3, 'int'] <- 1
#'  #generate function that produces random allocations
#'  treatf <- function(){
#'              tr <- sample(rep(c(0,1),each=3),6,replace = FALSE)
#'              rep(tr,each=25)
#'              }
#'  mf1 <- MeanFunction$new(
#'    formula = ~ factor(t) + int - 1,
#'    data=df,
#'    parameters = c(rep(0,5),0.6),
#'    family =gaussian(),
#'    treat_var = "int",
#'    random_function = treatf)
#'  cov1 <- Covariance$new(
#'    data = df,
#'    formula = ~ (1|gr(cl)),
#'    parameters = c(0.25))
#'  des <- Design$new(
#'    covariance = cov1,
#'    mean.function = mf1,
#'    var_par = 1)
#'  #run MCML to get parameter estimate:
#'  fit1 <- des$MCML(y = ysim,
#'   se.method = "none")
#'  perm1 <- des$permutation_test(
#'    y=ysim,
#'    permutation.par=6,
#'    start = fit1$coefficients$est[6],
#'    type="unw",
#'    iter = 1000,
#'    nsteps = 1000) 
#' }
#' @export
permutation_test <- function(y,
                             permutation.par,
                             start,
                             iter = 1000,
                             nsteps=1000,
                             type="cov",
                             parallel = TRUE,
                             verbose=TRUE){
  return(NULL)
}

