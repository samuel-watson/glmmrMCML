#include <glmmr.h>
#include "../inst/include/glmmrMCML.h"
#include <RcppEigen.h>
using namespace Rcpp;

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppEigen)]]

//' Likelihood maximisation for the GLMM 
//' 
//' Given model data and random effects samples `u`, this function will run the MCML steps to generate new estimates of 
//' mean function and covariance parameters
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
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
Rcpp::List mcml_optim(const Eigen::ArrayXXi &cov,
                      const Eigen::ArrayXd &data,
                      const Eigen::ArrayXd &eff_range,
                      const Eigen::MatrixXd &Z, 
                      const Eigen::MatrixXd &X,
                      const Eigen::VectorXd &y, 
                      const Eigen::MatrixXd &u,
                      std::string family, 
                      std::string link,
                      Eigen::ArrayXd start,
                      int trace,
                      bool mcnr = false){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd thetapars = start.segment(X.cols(),dat.n_cov_pars());
  glmmr::MCMLDmatrix dmat(&dat, thetapars);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,Z,X,y,u,family,link, start,trace);
  
  if(!mcnr){
    mc.l_optim();
  } else {
    mc.mcnr();
  }
  mc.d_optim();
  
  Eigen::VectorXd beta = mc.get_beta();
  Eigen::VectorXd theta = mc.get_theta();
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
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
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
Rcpp::List mcml_simlik(const Eigen::ArrayXXi &cov,
                       const Eigen::ArrayXd &data,
                       const Eigen::ArrayXd &eff_range,
                       const Eigen::MatrixXd &Z, 
                       const Eigen::MatrixXd &X,
                       const Eigen::VectorXd &y, 
                       const Eigen::MatrixXd &u,
                       std::string family, 
                       std::string link,
                       Eigen::ArrayXd start,
                       int trace){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd thetapars = start.segment(X.cols(),dat.n_cov_pars());
  glmmr::MCMLDmatrix dmat(&dat, thetapars);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,Z,X,y,u,family,link, start,trace);
  
  mc.f_optim();
  
  Eigen::VectorXd beta = mc.get_beta();
  Eigen::VectorXd theta = mc.get_theta();
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
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
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
Rcpp::List mcml_optim_sparse(const Eigen::ArrayXXi &cov,
                             const Eigen::ArrayXd &data,
                             const Eigen::ArrayXd &eff_range,
                             const Eigen::ArrayXi &Ap,
                             const Eigen::ArrayXi &Ai,
                             const Eigen::MatrixXd &Z, 
                             const Eigen::MatrixXd &X,
                             const Eigen::VectorXd &y, 
                             const Eigen::MatrixXd &u,
                             std::string family, 
                             std::string link,
                             Eigen::ArrayXd start,
                             int trace,
                             bool mcnr = false){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd thetapars = start.segment(X.cols(),dat.n_cov_pars());
  glmmr::SparseDMatrix dmat(&dat, thetapars,Ap,Ai);
  glmmr::mcmloptim<glmmr::SparseDMatrix> mc(&dmat,Z,X,y,u,family,link, start,trace);
  
  if(!mcnr){
    mc.l_optim();
  } else {
    mc.mcnr();
  }
  mc.d_optim();

  Eigen::VectorXd beta = mc.get_beta();
  Eigen::VectorXd theta = mc.get_theta();
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
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
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
Rcpp::List mcml_simlik_sparse(const Eigen::ArrayXXi &cov,
                              const Eigen::ArrayXd &data,
                              const Eigen::ArrayXd &eff_range,
                              const Eigen::ArrayXi &Ap,
                              const Eigen::ArrayXi &Ai,
                              const Eigen::MatrixXd &Z, 
                              const Eigen::MatrixXd &X,
                              const Eigen::VectorXd &y, 
                              const Eigen::MatrixXd &u,
                              std::string family, 
                              std::string link,
                              Eigen::ArrayXd start,
                              int trace){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd thetapars = start.segment(X.cols(),dat.n_cov_pars());
  glmmr::SparseDMatrix dmat(&dat, thetapars,Ap,Ai);
  glmmr::mcmloptim<glmmr::SparseDMatrix> mc(&dmat,Z,X,y,u,family,link, start,trace);
  
  mc.f_optim();
  
  Eigen::VectorXd beta = mc.get_beta();
  Eigen::VectorXd theta = mc.get_theta();
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
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
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
Eigen::MatrixXd mcml_hess(const Eigen::ArrayXXi &cov,
                    const Eigen::ArrayXd &data,
                    const Eigen::ArrayXd &eff_range,
                    const Eigen::MatrixXd &Z, 
                    const Eigen::MatrixXd &X,
                    const Eigen::VectorXd &y, 
                    const Eigen::MatrixXd &u,
                    std::string family, 
                    std::string link,
                    Eigen::ArrayXd start,
                      double tol = 1e-5,
                      int trace = 0){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd theta = start.segment(X.cols(),dat.n_cov_pars());
  glmmr::MCMLDmatrix dmat(&dat, theta);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,Z,X,y,u,family,link, start,trace);
  
  Eigen::MatrixXd hess = mc.f_hess(tol);
  return hess;
}

//' Generate Hessian matrix of GLMM using sparse matrix methods
//' 
//' Generate Hessian matrix of GLMM with numerical differentiation using sparse matrix methods
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
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
//' @param tol The tolerance of the numerical differentiation routine
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return The estimated Hessian matrix
// [[Rcpp::export]]
Eigen::MatrixXd mcml_hess_sparse(const Eigen::ArrayXXi &cov,
                          const Eigen::ArrayXd &data,
                          const Eigen::ArrayXd &eff_range,
                          const Eigen::ArrayXi &Ap,
                          const Eigen::ArrayXi &Ai,
                          const Eigen::MatrixXd &Z, 
                          const Eigen::MatrixXd &X,
                          const Eigen::VectorXd &y, 
                          const Eigen::MatrixXd &u,
                          std::string family, 
                          std::string link,
                          Eigen::ArrayXd start,
                          double tol = 1e-5,
                          int trace = 0){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd theta = start.segment(X.cols(),dat.n_cov_pars());
  glmmr::SparseDMatrix dmat(&dat, theta,Ap,Ai);
  glmmr::mcmloptim<glmmr::SparseDMatrix> mc(&dmat,Z,X,y,u,family,link, start,trace);
  
  Eigen::MatrixXd hess = mc.f_hess(tol);
  return hess;
}

//' Calculates the conditional Akaike Information Criterion for the GLMM
//' 
//' Calculates the conditional Akaike Information Criterion for the GLMM 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param beta_par Vector specifying the values of the mean function parameters to estimate the AIC at
//' @param cov_par Vector specifying the values of the covariance function parameters to estimate the AIC at
//' @return Estimated conditional AIC
// [[Rcpp::export]]
double aic_mcml(const Eigen::ArrayXXi &cov,
                const Eigen::ArrayXd &data,
                const Eigen::ArrayXd &eff_range,
                const Eigen::MatrixXd &Z, 
                const Eigen::MatrixXd &X,
                const Eigen::VectorXd &y, 
                const Eigen::MatrixXd &u, 
                std::string family, 
                std::string link,
                const Eigen::VectorXd& beta_par,
                const Eigen::VectorXd& cov_par){
  
  int niter = u.cols();
  int n = y.size();
  //arma::vec zd(n);
  int P = X.cols();
  Eigen::VectorXd xb(n);
  double var_par;
  int dof = beta_par.size() + cov_par.size();
  
  if(family=="gaussian"){
    var_par = beta_par(P-1);
    xb = X*beta_par.segment(0,P-1);
  } else {
    var_par = 0;
    xb = X*beta_par;
  }
  
  glmmr::DData dat(cov,data,eff_range);
  glmmr::MCMLDmatrix dmat(&dat, cov_par);
  
  // Eigen::ArrayXd dmvvec = Eigen::ArrayXd::Zero(niter);
  // //#pragma omp parallel for
  // for(int j=0;j<niter;j++){
  //   dmvvec(j) += dmat.loglik(u.col(j));
  // }
  double dmvvec = dmat.loglik(u);
  
  Eigen::ArrayXd ll = Eigen::ArrayXd::Zero(niter);
  Eigen::MatrixXd zd = Z * u;
#pragma omp parallel for
  for(int j=0; j<niter ; j++){
    ll(j) += glmmr::maths::log_likelihood(y,xb + zd.col(j),var_par,family,link);
  }
  
  return (-2*( ll.mean() + dmvvec ) + 2*dof); 
  
}

//' Multivariate normal log likelihood
//' 
//' Calculates the log likelihood of the multivariate normal distribution using `glmmr` covariance representation
//' 

//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param gamma Vector specifying the values of the covariance function parameters
//' @param u Matrix (or vector) of observed values
//' @return Scalar value 
// [[Rcpp::export]]
double mvn_ll(const Eigen::ArrayXXi &cov,
              const Eigen::ArrayXd &data,
              const Eigen::ArrayXd &eff_range,
              const Eigen::ArrayXd &gamma,
              const Eigen::MatrixXd &u){
  glmmr::DData dat(cov,data,eff_range);
  glmmr::MCMLDmatrix dmat(&dat, gamma);
  return dmat.loglik(u);
}

