#include <glmmr.h>
#include "../inst/include/glmmrMCML.h"
#include <RcppEigen.h>
using namespace Rcpp;

#ifdef _OPENMP
#include <omp.h>
#endif


// [[Rcpp::depends(RcppEigen)]]

//' Markov Chain Monte Carlo Maximum Likelihood Algorithm 
//' 
//' Full Markov Chain Monte Carlo Maximum Likelihood Algorithm using c++ code and the inbuilt Hamiltonian Monte Carlo MCMC sampler.
//' 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=ever more detailed output
//' @param mcnr Logical indicating whether to use Newton-Raphson (TRUE) or Expectation Maximisation (FALSE)
//' @param m Integer. Total number of MCMC samples to draw on each iteration
//' @param maxiter Integer. The maximum number of MCML iterations
//' @param warmup Integer. The number of warmup iterations for the MCMC sampler. Note, this reduces to 10 after
//' the first iteration as the sampler starts from the last set of values and adaptive step size
//' @param tol Value of the tolerance. The algorithm termninates if differences in values of parameters between
//' iterations are all less than this value.
//' @param verbose Logical indicating whether to provide output to the console
//' @param lambda Value of the trajectory length of the leapfrog integrator in Hamiltonian Monte Carlo
//'  (equal to number of steps times the step length). Larger values result in lower correlation in samples, but
//'  require larger numbers of steps and so is slower.
//' @param refresh Integer. Number of MCMC iterations to print progress to the console (requires verbose=TRUE)
//' @param maxsteps Integer. The maximum number of steps of the leapfrom integrator
//' @param target_accept The target acceptance rate of HMC proposals (default 0.9)
//' @return A list with the maximum likelihood estimates of the model parameters, the final set of MCMC samples, and
//' and indciator for whether the algorithm converged.
// [[Rcpp::export]]
Rcpp::List mcml_full(const Eigen::ArrayXXi &cov,
                     const Eigen::ArrayXd &data,
                     const Eigen::ArrayXd &eff_range,
                     const Eigen::MatrixXd &Z, 
                     const Eigen::MatrixXd &X,
                     const Eigen::VectorXd &y, 
                     std::string family, 
                     std::string link,
                     Eigen::ArrayXd start,
                     bool mcnr = false,
                     int m = 500,
                     int maxiter = 30,
                     int warmup = 500,
                     double tol = 1e-3,
                     bool verbose = true,
                     double lambda = 0.05,
                     int trace = 0,
                     int refresh = 500,
                     int maxsteps = 100,
                     double target_accept = 0.9){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::VectorXd theta = start.segment(X.cols(),dat.n_cov_pars()).matrix();
  Eigen::VectorXd beta = start.segment(0,X.cols()).matrix();
  double var_par = family=="gaussian"||family=="gamma" ? start(start.size()-1) : 1;
  glmmr::MCMLDmatrix dmat(&dat, theta);
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(Z.cols(),m);
  Eigen::MatrixXd L = dmat.genD(0,true,false);
  glmmr::mcmlModel model(Z,&L,X,y,&u,beta,var_par,family,link);
  glmmr::mcmc::mcmcRunHMC mcmc(&model,trace,lambda, refresh, maxsteps, target_accept);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,&model, start,trace);
  
  Eigen::ArrayXd diff(start.size());
  double maxdiff = 1;
  int iter = 1;
  
  Eigen::VectorXd newbeta = Eigen::VectorXd::Zero(beta.size());
  Eigen::VectorXd newtheta = Eigen::VectorXd::Zero(theta.size());
  double new_var_par = 1;
  bool converged = false;
  if(trace > 0 ) Rcpp::Rcout << "\n STARTING MCMCML \n " ;
  
  while(maxdiff > tol && iter <= maxiter){
    if(verbose)Rcpp::Rcout << "\n\nIter " << iter;
    if(trace > 0 ) Rcpp::Rcout << "\n MCMC sampling \n" ;
    // skip the warmup after first iteration as it will start from the previous iteration
    // if(iter == 1){
    //   u = mcmc.sample(warmup,m);
    // } else {
    //   u = mcmc.sample(10,m,10);
    // }
    u = mcmc.sample(warmup,m);
    
    if(trace > 0 ) Rcpp::Rcout << "\n Estimating beta " ;
    if(!mcnr){
      mc.l_optim();
    } else {
      mc.mcnr();
    }
    if(trace > 0 ) Rcpp::Rcout << "\n Estimating theta " ;
    mc.d_optim();
    
    newbeta = mc.get_beta();
    newtheta = mc.get_theta();
    if(family=="gaussian"||family=="gamma") new_var_par = mc.get_sigma();
    
    // check the differences
    diff.segment(0,beta.size()) = (beta - newbeta).cwiseAbs();
    diff.segment(beta.size(),theta.size()) = (theta - newtheta).cwiseAbs();
    diff(beta.size()+theta.size()) = abs(var_par - new_var_par);
    maxdiff = diff.maxCoeff();
    
    if(maxdiff < tol) converged = true;
    
    //update all the parameters
    beta = newbeta;
    theta = newtheta;
    var_par = new_var_par;
    if(!converged){
      dmat.update_parameters(newtheta);
      L = dmat.genD(0,true,false);
      model.update_beta(beta);
      model.var_par_ = new_var_par;
      model.update_L();
    }
    
    iter++;
    
    if(verbose){
      Rcpp::Rcout << "\nbeta: " << beta.transpose() << " theta: " << theta.transpose();
      if(family=="gaussian"||family=="gamma") Rcpp::Rcout << " sigma: " << var_par;
      Rcpp::Rcout << "\n Max. diff: " << maxdiff;
      if(converged)Rcpp::Rcout << " CONVERGED!";
    }
    
    
  }
  
  if(verbose && !converged) Rcpp::Rcout << " \n Warning: algorithm not converged and reached maximum iterations" << std::endl;
  
  Rcpp::List res = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = var_par,
                                      _["converged"] = converged, _["u"] = u);
  return res;
  
}

// sparse function - not yet available as no function genD in sparse class
// //' Markov Chain Monte Carlo Maximum Likelihood Algorithm using sparse matrix methods
// //' 
// //' Full Markov Chain Monte Carlo Maximum Likelihood Algorithm using c++ code and the inbuilt Hamiltonian 
// //' Monte Carlo MCMC sampler and using sparse matrix methods.
// //' 
// //' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
// //' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
// //' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
// //' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
// //' @param Z Matrix Z of the GLMM
// //' @param X Matrix X of the GLMM
// //' @param y Vector of observations
// //' @param family Character specifying the family
// //' @param link Character specifying the link function
// //' @param start Vector of starting values for the optimisation
// //' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=ever more detailed output
// //' @param mcnr Logical indicating whether to use Newton-Raphson (TRUE) or Expectation Maximisation (FALSE)
// //' @param m Integer. Total number of MCMC samples to draw on each iteration
// //' @param maxiter Integer. The maximum number of MCML iterations
// //' @param warmup Integer. The number of warmup iterations for the MCMC sampler. Note, this reduces to 10 after
// //' the first iteration as the sampler starts from the last set of values and adaptive step size
// //' @param tol Value of the tolerance. The algorithm termninates if differences in values of parameters between
// //' iterations are all less than this value.
// //' @param verbose Logical indicating whether to provide output to the console
// //' @param lambda Value of the trajectory length of the leapfrog integrator in Hamiltonian Monte Carlo
// //'  (equal to number of steps times the step length). Larger values result in lower correlation in samples, but
// //'  require larger numbers of steps and so is slower.
// //' @param refresh Integer. Number of MCMC iterations to print progress to the console (requires verbose=TRUE)
// //' @param maxsteps Integer. The maximum number of steps of the leapfrom integrator
// //' @return A list with the maximum likelihood estimates of the model parameters, the final set of MCMC samples, and
// //' and indciator for whether the algorithm converged.
// // [[Rcpp::export]]
// Rcpp::List mcml_full(const Eigen::ArrayXXi &cov,
//                      const Eigen::ArrayXd &data,
//                      const Eigen::ArrayXd &eff_range,
//                      const Eigen::ArrayXi &Ap,
//                      const Eigen::ArrayXi &Ai,
//                      const Eigen::MatrixXd &Z, 
//                      const Eigen::MatrixXd &X,
//                      const Eigen::VectorXd &y, 
//                      std::string family, 
//                      std::string link,
//                      Eigen::ArrayXd start,
//                      bool mcnr = false,
//                      int m = 500,
//                      int maxiter = 30,
//                      int warmup = 500,
//                      double tol = 1e-3,
//                      bool verbose = true,
//                      double lambda = 0.05,
//                      int trace = 0,
//                      int refresh = 500,
//                      int maxsteps = 100){
//   
//   glmmr::DData dat(cov,data,eff_range);
//   Eigen::VectorXd theta = start.segment(X.cols(),dat.n_cov_pars()).matrix();
//   Eigen::VectorXd beta = start.segment(0,X.cols()).matrix();
//   double var_par = family=="gaussian"||family=="gamma" ? start(start.size()-1) : 1;
//   glmmr::SparseDMatrix dmat(&dat, theta,Ap,Ai);
//   Eigen::MatrixXd u = Eigen::MatrixXd::Zero(Z.cols(),m);
//   Eigen::MatrixXd L = dmat.genD(0,true,false);
//   glmmr::mcmlModel model(Z,&L,X,y,&u,beta,var_par,family,link);
//   glmmr::mcmc::mcmcRunHMC mcmc(&model,trace,lambda, refresh, maxsteps);
//   glmmr::mcmloptim<glmmr::SparseDmatrix> mc(&dmat,&model, start,trace);
//   
//   Eigen::ArrayXd diff(start.size());
//   double maxdiff = 1;
//   int iter = 1;
//   
//   Eigen::VectorXd newbeta = Eigen::VectorXd::Zero(beta.size());
//   Eigen::VectorXd newtheta = Eigen::VectorXd::Zero(theta.size());
//   double new_var_par = 1;
//   bool converged = false;
//   if(trace > 0 ) Rcpp::Rcout << "\n STARTING MCMCML \n " ;
//   
//   while(maxdiff > tol && iter <= maxiter){
//     if(verbose)Rcpp::Rcout << "\n\nIter " << iter;
//     if(trace > 0 ) Rcpp::Rcout << "\n MCMC sampling \n" ;
//     // skip the warmup after first iteration as it will start from the previous iteration
//     if(iter == 1){
//       u = mcmc.sample(warmup,m);
//     } else {
//       u = mcmc.sample(10,m,10);
//     }
//     
//     
//     if(trace > 0 ) Rcpp::Rcout << "\n Estimating beta " ;
//     if(!mcnr){
//       mc.l_optim();
//     } else {
//       mc.mcnr();
//     }
//     if(trace > 0 ) Rcpp::Rcout << "\n Estimating theta " ;
//     mc.d_optim();
//     
//     newbeta = mc.get_beta();
//     newtheta = mc.get_theta();
//     if(family=="gaussian"||family=="gamma") new_var_par = mc.get_sigma();
//     
//     // check the differences
//     diff.segment(0,beta.size()) = (beta - newbeta).cwiseAbs();
//     diff.segment(beta.size(),theta.size()) = (theta - newtheta).cwiseAbs();
//     diff(beta.size()+theta.size()) = abs(var_par - new_var_par);
//     maxdiff = diff.maxCoeff();
//     
//     if(maxdiff < tol) converged = true;
//     
//     //update all the parameters
//     beta = newbeta;
//     theta = newtheta;
//     var_par = new_var_par;
//     if(!converged){
//       dmat.update_parameters(newtheta);
//       L = dmat.genD(0,true,false);
//       model.update_beta(beta);
//       model.var_par_ = new_var_par;
//       model.update_L();
//     }
//     
//     iter++;
//     
//     if(verbose){
//       Rcpp::Rcout << "\nbeta: " << beta.transpose() << " theta: " << theta.transpose();
//       if(family=="gaussian"||family=="gamma") Rcpp::Rcout << " sigma: " << var_par;
//       Rcpp::Rcout << "\n Max. diff: " << maxdiff;
//       if(converged)Rcpp::Rcout << " CONVERGED!";
//     }
//     
//     
//   }
//   
//   if(verbose && !converged) Rcpp::Rcout << " \n Warning: algorithm not converged and reached maximum iterations" << std::endl;
//   
//   Rcpp::List res = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = var_par,
//                                       _["converged"] = converged, _["u"] = u);
//   return res;
//   
// }

//' Hamiltonian Monte Carlo Sampler for Model Random Effects
//' 
//' Hamiltonian Monte Carlo Sampler for Model Random Effects
//' 
//' @param Z Matrix Z of the GLMM
//' @param L Matrix L, the Cholesky decomposition of the random effect covariance (returned with `gen_chol_D()` function of covariance)
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param beta Vector of parameters multiplying the X matrix
//' @param var_par (Optional) Value of the scale parameter where required (Gaussian and gamma distributions) 
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=ever more detailed output
//' @param nsamp Integer. Total number of MCMC samples to draw on each iteration
//' @param warmup Integer. The number of warmup iterations for the MCMC sampler. Note, this reduces to 10 after
//' the first iteration as the sampler starts from the last set of values and adaptive step size
//' @param lambda Value of the trajectory length of the leapfrog integrator in Hamiltonian Monte Carlo
//'  (equal to number of steps times the step length). Larger values result in lower correlation in samples, but
//'  require larger numbers of steps and so is slower.
//' @param refresh Integer. Number of MCMC iterations to print progress to the console (requires verbose=TRUE)
//' @param maxsteps Integer. The maximum number of steps of the leapfrom integrator
//' @param target_accept The target acceptance rate of HMC proposals (default 0.9)
//' @return A matrix (of dimension number of random effects * number of samples)
// [[Rcpp::export]]
Eigen::ArrayXXd mcmc_sample(const Eigen::MatrixXd &Z,
                            const Eigen::MatrixXd &L,
                            const Eigen::MatrixXd &X,
                            const Eigen::VectorXd &y,
                            const Eigen::VectorXd &beta,
                            std::string family,
                            std::string link,
                            int warmup, 
                            int nsamp, 
                            double lambda, 
                            double var_par = 1,
                            int trace = 0, 
                            int refresh = 500,
                            int maxsteps = 100,
                            double target_accept = 0.9){
  
  //int niter = thin <= 1 ? nsamp : (int)floor(nsamp/thin);
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(Z.cols(),nsamp);
  Eigen::MatrixXd L_ = L;
  glmmr::mcmlModel model(Z,&L_,X,y,&u,beta,var_par,family,link);
  glmmr::mcmc::mcmcRunHMC mcmc(&model,trace,lambda,refresh,maxsteps,target_accept);
  if(trace > 0 ) Rcpp::Rcout << " \n STARTING SAMPLING" << std::endl;
  Eigen::ArrayXXd samples = mcmc.sample(warmup,nsamp);
  return samples;
}