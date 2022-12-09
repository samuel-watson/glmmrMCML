#include <glmmr.h>
#include "../inst/include/glmmrMCML.h"
#include <RcppEigen.h>
using namespace Rcpp;


// [[Rcpp::depends(RcppEigen)]]

//' Maximum Likelihood with Laplace Approximation and Derivative Free Optimisation
//' 
//' Maximum Likelihood with Laplace Approximation and Derivative Free Optimisation
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
//' @param tol Value of the tolerance. The algorithm termninates if differences in values of parameters between
//' iterations are all less than this value.
//' @param verbose Logical indicating whether to provide output to the console
// [[Rcpp::export]]
Rcpp::List mcml_la(const Eigen::ArrayXXi &cov,
            const Eigen::ArrayXd &data,
            const Eigen::ArrayXd &eff_range,
            const Eigen::MatrixXd &Z, 
            const Eigen::MatrixXd &X,
            const Eigen::VectorXd &y, 
            std::string family, 
            std::string link,
            Eigen::ArrayXd start,
            bool usehess = false,
                     double tol = 1e-3,
                     bool verbose = true,
                     int trace = 0,
                     int maxiter = 10){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::VectorXd theta = start.segment(X.cols(),dat.n_cov_pars()).matrix();
  Eigen::VectorXd beta = start.segment(0,X.cols()).matrix();
  double var_par = 1.0;//family=="gaussian"||family=="Gamma" ? start(start.size()-1) : 1;
  glmmr::MCMLDmatrix dmat(&dat, theta);
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(Z.cols(),1);
  Eigen::MatrixXd L = dmat.genD(0,true,false);
  glmmr::mcmlModel model(Z,&L,X,y,&u,beta,var_par,family,link);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,&model, start,trace);
  
  Eigen::ArrayXd diff(start.size());
  double maxdiff = 1;
  int iter = 1;

  Eigen::VectorXd newbeta = Eigen::VectorXd::Zero(beta.size());
  Eigen::VectorXd newtheta = Eigen::VectorXd::Zero(theta.size());
  double new_var_par = 1;
  bool converged = false;
  if(trace > 0 ) Rcpp::Rcout << "\n STARTING LA \n " ;
  
  //get good starting values
  //mc.mcnr();
  
  while(maxdiff > tol && iter <= maxiter){
    if(verbose)Rcpp::Rcout << "\n\nIter " << iter << "\n" << std::string(40, '-');
    mc.la_optim();
    newbeta = mc.get_beta();
    model.update_beta(newbeta);
    model.update_W();
    mc.la_optim_cov();
    newtheta = mc.get_theta();
    if(family=="gaussian"||family=="Gamma") new_var_par = mc.get_sigma();

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
      //model.update_zu();
      model.update_W();
      model.var_par_ = new_var_par;
      model.update_L();
    }

    iter++;

    if(verbose){

      Rcpp::Rcout << "\nbeta: " << beta.transpose() << "\ntheta: " << theta.transpose();
      if(family=="gaussian"||family=="Gamma"||family=="beta") Rcpp::Rcout << "\nsigma: " << var_par;
      Rcpp::Rcout << "\n Max. diff: " << maxdiff;
      if(converged)Rcpp::Rcout << " CONVERGED!";
      Rcpp::Rcout << "\n" << std::string(40, '-');
    }
  }

  mc.la_optim_bcov();
  beta = mc.get_beta();
  theta = mc.get_theta();
  if(family=="gaussian"||family=="Gamma") var_par = mc.get_sigma();

  if(verbose){

    Rcpp::Rcout << "\n" << std::string(40, '-') << "\nCompleted: \nbeta: " << beta.transpose() << "\ntheta: " << theta.transpose();
    if(family=="gaussian"||family=="Gamma"||family=="beta") Rcpp::Rcout << "\nsigma: " << var_par;
    Rcpp::Rcout << "\n" << std::string(40, '-');
  }
  
  //estimate standard errors
  Eigen::VectorXd se = Eigen::VectorXd::Zero(start.size());
  
  if(usehess){
    Eigen::MatrixXd hess = mc.hess_la();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(hess.rows(),hess.cols());

    hess = hess.llt().solve(I);
    for(int i = 0; i < hess.cols();i++)se(i) = sqrt(hess(i,i));
  } 
  
  // Eigen::VectorXd w = glmmr::maths::dhdmu(model.xb_,model.family_,model.link_);
  // Eigen::MatrixXd W = Eigen::MatrixXd::Zero(w.size(),w.size());
  // for(int i = 0; i < w.size(); i++){
  //   W(i,i) = w(i);
  // }
  // Eigen::MatrixXd D = dmat.genD(0,false,false);
  // Eigen::MatrixXd sigma = W + model.Z_ * D * model.Z_.transpose();
  // Eigen::MatrixXd I = Eigen::MatrixXd::Identity(sigma.rows(),sigma.cols());
  // sigma = sigma.llt().solve(I);
  // Eigen::MatrixXd infomat = model.X_.transpose()*sigma*model.X_;
  // infomat = infomat.inverse();
  // for(int i = 0; i < infomat.cols();i++)se(i) = sqrt(infomat(i,i));
  // Rcpp::Rcout << "\nSE: \n" << se.transpose();
  
  //return 0;
  // //if(verbose && !converged) Rcpp::Rcout << " \n Warning: algorithm not converged and reached maximum iterations" << std::endl;
  
  u = L*u;
  Rcpp::List res = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,
                                      _["sigma"] = var_par, _["se"] = se,
                                      _["u"] = u);
  return res;
  
}

// [[Rcpp::depends(RcppEigen)]]

//' Maximum Likelihood with Laplace Approximation and Newton-Raphson 
//' 
//' Maximum Likelihood with Laplace Approximation and Newton-Raphson
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
//' @param tol Value of the tolerance. The algorithm termninates if differences in values of parameters between
//' iterations are all less than this value.
//' @param verbose Logical indicating whether to provide output to the console
// [[Rcpp::export]]
Rcpp::List mcml_la_nr(const Eigen::ArrayXXi &cov,
                   const Eigen::ArrayXd &data,
                   const Eigen::ArrayXd &eff_range,
                   const Eigen::MatrixXd &Z, 
                   const Eigen::MatrixXd &X,
                   const Eigen::VectorXd &y, 
                   std::string family, 
                   std::string link,
                   Eigen::ArrayXd start,
                   bool usehess = false,
                   double tol = 1e-3,
                   bool verbose = true,
                   int trace = 0,
                   int maxiter = 10){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::VectorXd theta = start.segment(X.cols(),dat.n_cov_pars()).matrix();
  Eigen::VectorXd beta = start.segment(0,X.cols()).matrix();
  double var_par = 1.0;//family=="gaussian"||family=="Gamma" ? start(start.size()-1) : 1;
  glmmr::MCMLDmatrix dmat(&dat, theta);
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(Z.cols(),1);
  Eigen::MatrixXd L = dmat.genD(0,true,false);
  glmmr::mcmlModel model(Z,&L,X,y,&u,beta,var_par,family,link);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,&model, start,trace);
  model.update_W(0,true);
  
  Eigen::ArrayXd diff(start.size());
  double maxdiff = 1;
  int iter = 1;
  
  Eigen::VectorXd newbeta = Eigen::VectorXd::Zero(beta.size());
  Eigen::VectorXd newtheta = Eigen::VectorXd::Zero(theta.size());
  double new_var_par = 1;
  bool converged = false;
  if(trace > 0 ) Rcpp::Rcout << "\n STARTING LA \n " ;
  
  //get good starting values
  //mc.mcnr_b();
  
  while(maxdiff > tol && iter <= maxiter){
    if(verbose)Rcpp::Rcout << "\n\nIter " << iter << "\n" << std::string(40, '-');

    mc.mcnr_b();
    newbeta = mc.get_beta();
    model.update_beta(newbeta);
    model.update_W(0,true);
    mc.la_optim_cov();
    newtheta = mc.get_theta();
    if(family=="gaussian"||family=="Gamma"||family=="beta") new_var_par = mc.get_sigma();

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
      model.update_W(0,true);
      model.update_L();
      //model.update_D();
    }

    iter++;

    if(verbose){

      Rcpp::Rcout << "\nbeta: " << beta.transpose() << "\ntheta: " << theta.transpose();
      if(family=="gaussian"||family=="Gamma"||family=="beta") Rcpp::Rcout << "\nsigma: " << var_par;
      Rcpp::Rcout << "\n Max. diff: " << maxdiff;
      if(converged)Rcpp::Rcout << " CONVERGED!";
      Rcpp::Rcout << "\n" << std::string(40, '-');
    }
  }
  
  mc.la_optim_bcov();
  beta = mc.get_beta();
  theta = mc.get_theta();
  if(family=="gaussian"||family=="Gamma") var_par = mc.get_sigma();

  if(verbose){

    Rcpp::Rcout << "\n" << std::string(40, '-') << "\nCompleted: \nbeta: " << beta.transpose() << "\ntheta: " << theta.transpose();
    if(family=="gaussian"||family=="Gamma"||family=="beta") Rcpp::Rcout << "\nsigma: " << var_par;
    Rcpp::Rcout << "\n" << std::string(40, '-');
  }
  
  // //estimate standard errors
  Eigen::VectorXd se = Eigen::VectorXd::Zero(start.size());
  
  if(usehess){
    Eigen::MatrixXd hess = mc.hess_la();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(hess.rows(),hess.cols());
    
    hess = hess.llt().solve(I);
    for(int i = 0; i < hess.cols();i++)se(i) = sqrt(hess(i,i));
  }
  // 
  u = L*u;
  Rcpp::List res = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,
                                      _["sigma"] = var_par, _["se"] = se,
                                      _["u"] = u);
  return res;
  
}
