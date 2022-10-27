#include "../inst/include/glmmrmcmlclass.h"
using namespace Rcpp;
using namespace arma;

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

//' Likelihood maximisation for the GLMM 
//' 
//' Given model data and random effects samples `u`, this function will run the MCML steps to generate new estimates of 
//' mean function and covariance parameters
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
//' @param D_data Named list specifying the covariance matrix D. Usually the return from the member function `get_D_data()` of the 
//' covariance class
//' @param Q Integer specifying the number of covariance parameters
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @param mcnr Logical indicating whether to use Newton-Raphson (TRUE) or Expectation Maximisation (FALSE)
//' @return A vector of the parameters that maximise the likelihood functions
// [[Rcpp::export]]
Rcpp::List mcml_optim(Rcpp::List D_data,
                      int Q,
                      const arma::mat &Z, 
                      const arma::mat &X,
                      const arma::vec &y, 
                      const arma::mat &u,
                      std::string family, 
                      std::string link,
                      arma::vec start,
                      int trace,
                      bool mcnr = false){
  DMatrix dmat(D_data,start.subvec(X.n_cols,X.n_cols+Q-1));
  mcmloptim<DMatrix> mc(&dmat,Z,X,y,u,Q,family,link, start,trace);
  
  if(!mcnr){
    mc.l_optim();
  } else {
    mc.mcnr();
  }
  mc.d_optim();
  
  arma::vec beta = mc.get_beta();
  arma::vec theta = mc.get_theta();
  double sigma = mc.get_sigma();
  
  Rcpp::List L = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma);
  return L;
}

//' Simulated likelihood optimisation step for MCML
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
//' Likelihood maximisation for the GLMM
//' @param  D_data Named list specifying the covariance matrix D. Usually the return from the member function `get_D_data()` of the 
//' covariance class
//' @param Q Integer specifying the number of covariance parameters
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return A vector of the parameters that maximise the simulated likelihood
// [[Rcpp::export]]
Rcpp::List mcml_simlik(Rcpp::List D_data,
                      int Q,
                      const arma::mat &Z, 
                      const arma::mat &X,
                      const arma::vec &y, 
                      const arma::mat &u,
                      std::string family, 
                      std::string link,
                      arma::vec start,
                      int trace){
  DMatrix dmat(D_data,start.subvec(X.n_cols,X.n_cols+Q-1));
  mcmloptim<DMatrix> mc(&dmat,Z,X,y,u,Q,family,link, start,trace);
  
  mc.f_optim();
  
  arma::vec beta = mc.get_beta();
  arma::vec theta = mc.get_theta();
  double sigma = mc.get_sigma();
  
  Rcpp::List L = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma);
  return L;
}

//' Likelihood maximisation for the GLMM using sparse matrix methods
//' 
//' Given model data and random effects samples `u`, this function will run the MCML steps to generate new estimates of 
//' mean function and covariance parameters
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
//' Likelihood maximisation for the GLMM
//' @param  D_data Named list specifying the covariance matrix D. Usually the return from the member function `get_D_data()` of the 
//' covariance class
//' @param Q Integer specifying the number of covariance parameters
//' @param Ap Integer vector of pointers, one for each column, specifying the initial (zero-based) index of elements in the column. Slot `p`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Ai Integer vector specifying the row indices of the non-zero elements of the matrix. Slot `i`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @param mcnr Logical indicating whether to use Newton-Raphson (TRUE) or Expectation Maximisation (FALSE)
//' @return A vector of the parameters that maximise the simulated likelihood
// [[Rcpp::export]]
Rcpp::List mcml_optim_sparse(Rcpp::List D_data,
                             int Q,
                             const arma::uvec &Ap,
                             const arma::uvec &Ai,
                             const arma::mat &Z, 
                             const arma::mat &X,
                             const arma::vec &y, 
                             const arma::mat &u,
                             std::string family, 
                             std::string link,
                             arma::vec start,
                             int trace,
                             bool mcnr = false){
                      
  SparseDMatrix dmat(D_data,start.subvec(X.n_cols,X.n_cols+Q-1),Ap,Ai);
  mcmloptim<SparseDMatrix> mc(&dmat,Z,X,y,u,Q,family,link, start,trace);
  if(!mcnr){
    mc.l_optim();
  } else {
    mc.mcnr();
  }
  mc.d_optim();

  arma::vec beta = mc.get_beta();
  arma::vec theta = mc.get_theta();
  double sigma = mc.get_sigma();

  Rcpp::List L = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma,
                                    _["Ap"] = dmat.chol_->L->Ap,_["Ai"] = dmat.chol_->L->Ai,
                                    _["Ax"] = dmat.chol_->L->Ax,_["D"] = dmat.chol_->D);
  return L;
}

//' Simulated likelihood optimisation step for MCML using sparse matrix methods
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
//' Likelihood maximisation for the GLMM
//' @param  D_data Named list specifying the covariance matrix D. Usually the return from the member function `get_D_data()` of the 
//' covariance class
//' @param Q Integer specifying the number of covariance parameters
//' @param Ap Integer vector of pointers, one for each column, specifying the initial (zero-based) index of elements in the column. Slot `p`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Ai Integer vector specifying the row indices of the non-zero elements of the matrix. Slot `i`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return A vector of the parameters that maximise the simulated likelihood
// [[Rcpp::export]]
Rcpp::List mcml_simlik_sparse(Rcpp::List D_data,
                       int Q,
                       const arma::uvec &Ap,
                       const arma::uvec &Ai,
                       const arma::mat &Z, 
                       const arma::mat &X,
                       const arma::vec &y, 
                       const arma::mat &u,
                       std::string family, 
                       std::string link,
                       arma::vec start,
                       int trace){
  SparseDMatrix dmat(D_data,start.subvec(X.n_cols,X.n_cols+Q-1),Ap,Ai);
  mcmloptim<SparseDMatrix> mc(&dmat,Z,X,y,u,Q,family,link, start,trace);
  
  mc.f_optim();
  
  arma::vec beta = mc.get_beta();
  arma::vec theta = mc.get_theta();
  double sigma = mc.get_sigma();
  
  Rcpp::List L = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma);
  return L;
}

//' Generate Hessian matrix of GLMM
//' 
//' Generate Hessian matrix of GLMM using numerical differentiation
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
//' @param  D_data Named list specifying the covariance matrix D. Usually the return from the member function `get_D_data()` of the 
//' covariance class
//' @param Q Integer specifying the number of covariance parameters
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param tol The tolerance of the numerical differentiation routine
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return The estimated Hessian matrix
// [[Rcpp::export]]
arma::mat mcml_hess(Rcpp::List D_data,
                      int Q,
                      const arma::mat &Z, 
                      const arma::mat &X,
                      const arma::vec &y, 
                      const arma::mat &u,
                      std::string family, 
                      std::string link,
                      arma::vec start,
                      double tol = 1e-5,
                      int trace = 0){
  
  DMatrix dmat(D_data,start.subvec(X.n_cols,X.n_cols+Q-1));
  mcmloptim<DMatrix> mc(&dmat,Z,X,y,u,Q,family,link, start,trace);
  
  arma::mat hess = mc.f_hess(tol);
  return hess;
}

//' Calculates the conditional Akaike Information Criterion for the GLMM
//' 
//' Calculates the conditional Akaike Information Criterion for the GLMM 
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param  D_data Named list specifying the covariance matrix D. Usually the return from the member function `get_D_data()` of the 
//' covariance class
//' @param beta_par Vector specifying the values of the mean function parameters to estimate the AIC at
//' @param cov_par Vector specifying the values of the covariance function parameters to estimate the AIC at
//' @return Estimated conditional AIC
// [[Rcpp::export]]
double aic_mcml(const arma::mat &Z, 
                const arma::mat &X,
                const arma::vec &y, 
                const arma::mat &u, 
                std::string family, 
                std::string link,
                Rcpp::List D_data,
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
  
  DMatrix dmat(D_data,cov_par);
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