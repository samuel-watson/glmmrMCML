#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

#include <rbobyqa.h>
#include <RcppEigen.h>
#include "moremaths.h"


using namespace rminqa;


// [[Rcpp::depends(RcppEigen)]]


namespace glmmr{
namespace likelihood {

template<typename T>
class D_likelihood : public Functor {
  T* D_;
  Eigen::ArrayXXd u_;
  
public:
  D_likelihood(T* D,
               const Eigen::MatrixXd &u) : 
  D_(D), u_(u) {}
  double operator()(const std::vector<double> &par) override{
    int nrow = u_.cols();
    D_->update_parameters(par);
    double logl = D_->loglik(u_);
    return -1*logl;
  }
};

class L_likelihood : public Functor {
  Eigen::MatrixXd Z_;
  Eigen::MatrixXd X_;
  Eigen::VectorXd y_;
  Eigen::MatrixXd u_;
  std::string family_;
  std::string link_;
  bool fix_var_;
  double fix_var_par_;
public:
  L_likelihood(const Eigen::MatrixXd &Z, 
               const Eigen::MatrixXd &X,
               const Eigen::VectorXd &y, 
               const Eigen::MatrixXd &u, 
               std::string family, 
               std::string link,
               bool fix_var = false,
               double fix_var_par = 0) : 
  Z_(Z), X_(X), y_(y), u_(u),family_(family) , link_(link), 
  fix_var_(fix_var), fix_var_par_(fix_var_par){}
  double operator()(const std::vector<double> &par) override{
    int niter = u_.cols();
    int P = X_.cols();
    std::vector<double> par2 = par;
    Eigen::Map<Eigen::VectorXd> beta(par2.data(),P); 
    Eigen::VectorXd xb = X_*beta;
    double var_par = family_=="gaussian"&&!fix_var_ ? par[P] : fix_var_par_;
    Eigen::ArrayXd ll = Eigen::ArrayXd::Zero(niter);
    Eigen::MatrixXd zd = Z_ * u_;
#pragma omp parallel for
    for(int j=0; j<niter ; j++){
      ll(j) += glmmr::maths::log_likelihood(y_,xb + zd.col(j),var_par,family_,link_);
    }
    //Rcpp::Rcout << "\n ll: " << ll;
    return -1 * ll.mean();
  }
};

template<typename T>
class F_likelihood : public Functor {
  T* D_;
  Eigen::MatrixXd Z_;
  Eigen::MatrixXd X_;
  Eigen::VectorXd y_;
  Eigen::MatrixXd u_;
  Eigen::ArrayXd cov_par_fix_;
  std::string family_;
  std::string link_;
  bool importance_;
  bool fix_var_;
  double fix_var_par_;
  
public:
  F_likelihood(T* D,
               const Eigen::MatrixXd &Z, 
               const Eigen::MatrixXd &X,
               const Eigen::VectorXd &y, 
               const Eigen::MatrixXd &u, 
               const Eigen::ArrayXd &cov_par_fix,
               std::string family, 
               std::string link,
               bool importance = false,
               bool fix_var = false,
               double fix_var_par = 0) : 
  D_(D) ,Z_(Z), X_(X), y_(y), 
  u_(u),cov_par_fix_(cov_par_fix), family_(family) , link_(link),
  importance_(importance),fix_var_(fix_var), fix_var_par_(fix_var_par) {}
  
  
  double operator()(const std::vector<double> &par) override{
    int niter = u_.cols();
    int P = X_.cols();
    int Q = cov_par_fix_.size();
    std::vector<double> par2 = par;
    Eigen::Map<Eigen::VectorXd> beta(par2.data(),P); 
    std::vector<double> theta;
    for(int i=0; i<Q; i++)theta.push_back(par2[P+i]);
    Eigen::VectorXd xb = X_*beta;
    double var_par = family_=="gaussian"&&!fix_var_ ? par[P] : fix_var_par_;
    Eigen::ArrayXd ll = Eigen::ArrayXd::Zero(niter);
    Eigen::MatrixXd zd = Z_ * u_;
#pragma omp parallel for
    for(int j=0; j<niter ; j++){
      ll(j) += glmmr::maths::log_likelihood(y_,xb + zd.col(j),var_par,family_,link_);
    }
    D_->update_parameters(theta);
    double logl = D_->loglik(u_);
    
    if(importance_){
      D_->update_parameters(cov_par_fix_);
      double denomD = D_->loglik(u_);
      double du = exp(ll.mean() + logl)/ exp(denomD);
      return -1.0 * log(du);
    } else {
      return -1.0*(ll.mean() + logl);
    }
  }
};

}
}








#endif