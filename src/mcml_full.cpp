#include <glmmr.h>
#include "../inst/include/glmmrMCML.h"
#include <RcppEigen.h>
using namespace Rcpp;

#ifdef _OPENMP
#include <omp.h>
#endif

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
                     int thin = 5,
                     int maxiter = 30,
                     int warmup = 500,
                     double tol = 1e-3,
                     bool verbose = true,
                     int trace = 0){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::VectorXd theta = start.segment(X.cols(),dat.n_cov_pars()).matrix();
  Eigen::VectorXd beta = start.segment(0,X.cols()).matrix();
  double var_par = family=="gaussian"||family=="gamma" ? start(start.size()-1) : 1;
  glmmr::MCMLDmatrix dmat(&dat, theta);
  Eigen::MatrixXd L = dmat.genD(0,true,false);
  glmmr::mcmc::mcmcModel model(Z,L,X,y,beta,var_par,family,link);
  glmmr::mcmc::mcmcRun mcmc(&model);
  int miter = thin > 1 ? (int)floor(m/thin) : m;
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(Z.cols(),miter);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,Z,X,y,u,family,link, start,trace);
  
  Eigen::ArrayXd diff(start.size());
  double maxdiff = 1;
  int iter = 1;
  
  Eigen::VectorXd newbeta = Eigen::VectorXd::Zero(beta.size());
  Eigen::VectorXd newtheta = Eigen::VectorXd::Zero(theta.size());
  double new_var_par = 1;
  bool converged = false;
  
  while(maxdiff > tol && iter <= maxiter){
    if(verbose)Rcpp::Rcout << "\n\nIter " << iter;
    u = mcmc.sample(warmup,m,thin);
    mc.u_ = u;
    
    if(!mcnr){
      mc.l_optim();
    } else {
      mc.mcnr();
    }
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
      model.update_beta(newbeta);
      model.var_par_ = new_var_par;
      model.L_ = L;
      mc.u_ = u;
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

// [[Rcpp::export]]
Eigen::MatrixXd mcmc_sample(const Eigen::MatrixXd &Z,
                            const Eigen::MatrixXd &L,
                            const Eigen::MatrixXd &X,
                            const Eigen::VectorXd &y,
                            const Eigen::VectorXd &beta,
                            double var_par,
                            std::string family,
                            std::string link,
                            int warmup, int nsamp, int thin){
  glmmr::mcmc::mcmcModel model(Z,L,X,y,beta,var_par,family,link);
  glmmr::mcmc::mcmcRun mcmc(&model);
  Eigen::ArrayXXd samples = mcmc.sample(warmup,nsamp,thin);
  return samples;
}