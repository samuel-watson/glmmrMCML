#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

#include <rbobyqa.h>
#include <RcppEigen.h>
#include "moremaths.h"
#include "mcmlmodel.h"

using namespace rminqa;


// [[Rcpp::depends(RcppEigen)]]


namespace glmmr{
namespace likelihood {

template<typename T>
class D_likelihood : public Functor {
  T* D_;
  Eigen::MatrixXd* u_;
  
public:
  D_likelihood(T* D,
               Eigen::MatrixXd* u) : 
  D_(D), u_(u) {}
  double operator()(const std::vector<double> &par) override{
    int nrow = u_->cols();
    D_->update_parameters(par);
    double logl = D_->loglik((*u_));
    return -1*logl;
  }
};

class L_likelihood : public Functor {
  glmmr::mcmlModel* M_;
  bool fix_var_;
  double fix_var_par_;
public:
  L_likelihood(glmmr::mcmlModel* M,
               bool fix_var = false,
               double fix_var_par = 0) :  
  M_(M), fix_var_(fix_var), fix_var_par_(fix_var_par){}
  double operator()(const std::vector<double> &par) override{
    std::vector<double> par2 = par;
    Eigen::Map<Eigen::VectorXd> beta(par2.data(),M_->P_); 
    M_->update_beta(beta);
    M_->var_par_ = M_->family_=="gaussian"&&!fix_var_ ? par[M_->P_] : fix_var_par_;
    double ll = M_->log_likelihood();
    return -1*ll;
  }
};

template<typename T>
class F_likelihood : public Functor {
  T* D_;
  glmmr::mcmlModel* M_;
  bool importance_;
  bool fix_var_;
  double fix_var_par_;
  Eigen::ArrayXd cov_par_fix_;
  
public:
  F_likelihood(T* D,
               glmmr::mcmlModel* M,
               const Eigen::ArrayXd &cov_par_fix,
               bool importance = false,
               bool fix_var = false,
               double fix_var_par = 0) : 
  D_(D), M_(M), importance_(importance), 
  fix_var_(fix_var), fix_var_par_(fix_var_par),
  cov_par_fix_(cov_par_fix) {}
  
  
  double operator()(const std::vector<double> &par) override{
    int Q = cov_par_fix_.size();
    std::vector<double> par2 = par;
    Eigen::Map<Eigen::VectorXd> beta(par2.data(),M_->P_); 
    std::vector<double> theta;
    for(int i=0; i<Q; i++)theta.push_back(par2[M_->P_+i]);
    M_->update_beta(beta);
    M_->var_par_ = M_->family_=="gaussian"&&!fix_var_ ? par[M_->P_] : fix_var_par_;
    double ll = M_->log_likelihood();
    
    D_->update_parameters(theta);
    double logl = D_->loglik((*(M_->u_)));
    
    if(importance_){
      D_->update_parameters(cov_par_fix_);
      double denomD = D_->loglik((*(M_->u_)));
      double du = exp(ll + logl)/ exp(denomD);
      return -1.0 * log(du);
    } else {
      return -1.0*(ll + logl);
    }
  }
};

}
}


#endif