#include "../inst/include/glmmrmcmlclass.h"
using namespace Rcpp;
using namespace arma;

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

//' Likelihood maximisation for the GLMM 
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
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
//' @param sum_N_par Total number of parameters
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
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @param mcnr Logical indicating whether to use Newton-Raphson (TRUE) or Expectation Maximisation (FALSE)
//' @param importance Logical indicating whether to use importance sampling step
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
                      int trace,
                      bool mcnr = false,
                      bool importance = false){
  DMatrix dmat(B,N_dim,N_func,func_def,N_var_func,col_id,N_par,sum_N_par,cov_data,cov_par_fix);
  mcmloptim mc(&dmat,Z,X,y,u,cov_par_fix,family,link, start,trace);
  
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

//' Likelihood maximisation for the GLMM s
//' 
//' Likelihood maximisation for the GLMM
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
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
//' @param sum_N_par Total number of parameters
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
                      int trace){
  
  DMatrix dmat(B,N_dim,N_func,func_def,N_var_func,col_id,N_par,sum_N_par,cov_data,cov_par_fix);
  mcmloptim mc(&dmat,Z,X,y,u,cov_par_fix,family,link, start,trace);
  
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
//' @param sum_N_par Total number of parameters
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
  dmat.gen_blocks_byfunc();
  arma::vec dmvvec(niter,fill::zeros);
  //#pragma omp parallel for
  for(arma::uword j=0;j<niter;j++){
    dmvvec(j) += dmat.loglik(u.col(j));
  }
  
  arma::vec ll(niter,fill::zeros);
  arma::mat zd = Z * u;
#pragma omp parallel for
  for(arma::uword j=0; j<niter ; j++){
    ll(j) += log_likelihood(y,xb + zd.col(j),var_par,family,link);
  }
  
  return (-2*( mean(ll) + mean(dmvvec) ) + 2*arma::as_scalar(dof)); 
  
}