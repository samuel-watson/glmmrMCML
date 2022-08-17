#include "../inst/include/glmmrmcml.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

//' Likelihood maximisation for the GLMM 
//' 
//' Likelihood maximisation for the GLMM
//' @param B Integer specifying the number of blocks in the matrix
//' @param N_dim Vector of integers, which each value specifying the dimension of each block
//' @param N_func Vector of integers specifying the number of functions in the covariance function 
//' for each block.
//' @param func_def Matrix of integers where each column specifies the function definition for each function in each block. 
//' @param N_var_func Matrix of integers of same size as `func_def` with each column specying the number 
//' of variables in the argument to each function in each block
//' @param col_id 3D array (cube) of integers of dimension length(func_def) x max(N_var_func) x B 
//' where each slice the respective column indexes of `cov_data` for each function in the block
//' @param N_par Matrix of integers of same size as `func_def` with each column specifying the number
//' of parameters in the function in each block
//' @param cov_data 3D array (cube) holding the data for the covariance matrix where each of the B slices
//' is the data required for each block
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param cov_par_fix A vector of covariance parameters for importance sampling
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param lower Vector of lower bounds for the model parameters
//' @param upper Vector of upper bounds for the model parameters
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return A vector of the parameters that maximise the simulated likelihood
// [[Rcpp::export]]
Rcpp::List mcml_optim(const arma::uword &B,
                      const arma::uvec &N_dim,
                      const arma::uvec &N_func,
                      const arma::umat &func_def,
                      const arma::umat &N_var_func,
                      const arma::ucube &col_id,
                      const arma::umat &N_par,
                      const arma::uword &sum_N_par,
                      const arma::cube &cov_data,
                      const arma::mat &Z, 
                      const arma::mat &X,
                      const arma::vec &y, 
                      const arma::mat &u,
                      const arma::vec &cov_par_fix,
                      std::string family, 
                      std::string link,
                      arma::vec start,
                      const arma::vec &lower_b,
                      const arma::vec &upper_b,
                      const arma::vec &lower_t,
                      const arma::vec &upper_t,
                      int trace,
                      bool mcnr = false,
                      bool importance = false){
  mcmloptim mc(B,N_dim,
               N_func,
               func_def,N_var_func,
               col_id,N_par,sum_N_par,
               cov_data,Z,X,y,u,
               cov_par_fix,family,
               link, start,lower_b,upper_b,
               lower_t,upper_t,trace);
  
  
  if(!mcnr){
    mc.l_optim();
  } else {
    mc.mcnr();
  }
  mc.d_optim();
  if(importance)mc.f_optim();
  
  arma::vec beta = mc.get_beta();
  arma::vec theta = mc.get_theta();
  double sigma = mc.get_sigma();
  
  Rcpp::List L = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma);
  return L;
}

//' Likelihood maximisation for the GLMM 
//' 
//' Likelihood maximisation for the GLMM
//' @param B Integer specifying the number of blocks in the matrix
//' @param N_dim Vector of integers, which each value specifying the dimension of each block
//' @param N_func Vector of integers specifying the number of functions in the covariance function 
//' for each block.
//' @param func_def Matrix of integers where each column specifies the function definition for each function in each block. 
//' @param N_var_func Matrix of integers of same size as `func_def` with each column specying the number 
//' of variables in the argument to each function in each block
//' @param col_id 3D array (cube) of integers of dimension length(func_def) x max(N_var_func) x B 
//' where each slice the respective column indexes of `cov_data` for each function in the block
//' @param N_par Matrix of integers of same size as `func_def` with each column specifying the number
//' of parameters in the function in each block
//' @param cov_data 3D array (cube) holding the data for the covariance matrix where each of the B slices
//' is the data required for each block
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param cov_par_fix A vector of covariance parameters for importance sampling
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param lower Vector of lower bounds for the model parameters
//' @param upper Vector of upper bounds for the model parameters
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return A vector of the parameters that maximise the simulated likelihood
// [[Rcpp::export]]
arma::mat mcml_hess(const arma::uword &B,
                      const arma::uvec &N_dim,
                      const arma::uvec &N_func,
                      const arma::umat &func_def,
                      const arma::umat &N_var_func,
                      const arma::ucube &col_id,
                      const arma::umat &N_par,
                      const arma::uword &sum_N_par,
                      const arma::cube &cov_data,
                      const arma::mat &Z, 
                      const arma::mat &X,
                      const arma::vec &y, 
                      const arma::mat &u,
                      const arma::vec &cov_par_fix,
                      std::string family, 
                      std::string link,
                      arma::vec start,
                      const arma::vec &lower_b,
                      const arma::vec &upper_b,
                      const arma::vec &lower_t,
                      const arma::vec &upper_t,
                      int trace,
                      bool importance = false){
  
  mcmloptim mc(B,N_dim,
               N_func,
               func_def,N_var_func,
               col_id,N_par,sum_N_par,
               cov_data,Z,X,y,u,
               cov_par_fix,family,
               link, start,lower_b,upper_b,
               lower_t,upper_t,trace);
  
  arma::mat hess = mc.f_hess();
  return hess;
}

//' Calculates the Akaike Information Criterion for the GLMM
//' 
//' Calculates the Akaike Information Criterion for the GLMM 
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param B Integer specifying the number of blocks in the matrix
//' @param N_dim Vector of integers, which each value specifying the dimension of each block
//' @param N_func Vector of integers specifying the number of functions in the covariance function 
//' for each block.
//' @param func_def Matrix of integers where each column specifies the function definition for each function in each block. 
//' @param N_var_func Matrix of integers of same size as `func_def` with each column specying the number 
//' of variables in the argument to each function in each block
//' @param col_id 3D array (cube) of integers of dimension length(func_def) x max(N_var_func) x B 
//' where each slice the respective column indexes of `cov_data` for each function in the block
//' @param N_par Matrix of integers of same size as `func_def` with each column specifying the number
//' of parameters in the function in each block
//' @param cov_data 3D array (cube) holding the data for the covariance matrix where each of the B slices
//' is the data required for each block
//' @param beta_par Vector specifying the values of the mean function parameters
//' @param cov_par Vector specifying the values of the covariance parameters
//' @return A matrix of the Hessian for each parameter
// [[Rcpp::export]]
double aic_mcml(const arma::mat &Z, 
                const arma::mat &X,
                const arma::vec &y, 
                const arma::mat &u, 
                std::string family, 
                std::string link,
                const arma::uword &B,
                const arma::uvec &N_dim,
                const arma::uvec &N_func,
                const arma::umat &func_def,
                const arma::umat &N_var_func,
                const arma::ucube &col_id,
                const arma::umat &N_par,
                const arma::uword &sum_N_par,
                const arma::cube &cov_data,
                const arma::vec& beta_par,
                const arma::vec& cov_par){
  arma::uword niter = u.n_cols;
  arma::uword n = y.n_elem;
  //arma::vec zd(n);
  arma::uword P = beta_par.n_elem;
  arma::vec xb(n);
  double var_par;
  arma::uword dof = beta_par.n_elem + cov_par.n_elem;
  
  if(family=="gaussian"){
    var_par = beta_par(P-1);
    xb = X*beta_par.subvec(0,P-2);
  } else {
    var_par = 0;
    xb = X*beta_par;
  }
  
  DMatrix dmat(B,N_dim,N_func,func_def,N_var_func,col_id,N_par,sum_N_par,cov_data,cov_par);
  arma::field<arma::mat> Dfield = dmat.genD();
  // arma::field<arma::mat> Dfield = genD(B,N_dim,
  //                                      N_func,
  //                                      func_def,N_var_func,
  //                                      col_id,N_par,sum_N_par,
  //                                      cov_data,cov_par);
  arma::vec dmvvec(niter,fill::zeros);
  double logdetD;
  arma::uword ndim_idx = 0;
  for(arma::uword b=0;b<B;b++){
    if(all(func_def.row(b)==1)){
#pragma omp parallel for collapse(2)
      for(arma::uword j=0;j<niter;j++){
        for(arma::uword k=0; k<Dfield[b].n_rows; k++){
          dmvvec(j) += -0.5*log(Dfield[b](k,k)) -0.5*log(2*arma::datum::pi) -
            0.5*pow(u(ndim_idx+k,j),2)/Dfield[b](k,k);
        }
      }
      
    } else {
      arma::mat invD = inv_sympd(Dfield[b]);
      logdetD = arma::log_det_sympd(Dfield[b]);
#pragma omp parallel for
      for(arma::uword j=0;j<niter;j++){
        dmvvec(j) += log_mv_gaussian_pdf(u.col(j).subvec(ndim_idx,ndim_idx+N_dim(b)-1),
               invD,logdetD);
      }
    }
    ndim_idx += N_dim(b);
  }
  
  arma::vec ll(niter,fill::zeros);
  arma::mat zd = Z * u;
#pragma omp parallel for
  for(arma::uword j=0; j<niter ; j++){
    ll(j) += log_likelihood(y,
       xb + zd.col(j),
       var_par,
       family,
       link);
  }
  
  return (-2*( mean(ll) + mean(dmvvec) ) + 2*arma::as_scalar(dof)); 
  
}

