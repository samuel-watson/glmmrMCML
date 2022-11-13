#ifndef MCMLMODEL_H
#define MCMLMODEL_H

#include <RcppEigen.h>
#include <glmmr.h>
#include "moremaths.h"

namespace glmmr {

class mcmlModel {
public:
  const Eigen::MatrixXd &X_;
  const Eigen::MatrixXd &Z_;
  Eigen::VectorXd xb_;
  Eigen::MatrixXd ZL_;
  Eigen::MatrixXd* L_;
  const Eigen::VectorXd &y_;
  Eigen::MatrixXd* u_;
  double var_par_;
  std::string family_; 
  std::string link_;
  int n_;
  int Q_;
  int P_;
  int niter_;
  
  mcmlModel(
    const Eigen::MatrixXd &Z, 
    Eigen::MatrixXd* L,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    Eigen::MatrixXd *u,
    Eigen::VectorXd beta,
    double var_par,
    std::string family, 
    std::string link
  ) : X_(X), xb_(X.rows()), Z_(Z), ZL_(X.rows(),X.rows()), L_(L),  
      y_(y), u_(u), var_par_(var_par), family_(family), link_(link) {
    xb_ = X* beta;
    if(L != nullptr)ZL_ = Z * (*L);  
    n_ = X.rows();
    Q_ = Z.cols();
    P_ = X.cols();
    niter_ = u->cols();
  }
  
  void update_beta(const Eigen::VectorXd &beta){
    xb_ = X_* beta;
  }
  
  void update_L(){
    //L_ = L;
    ZL_ = Z_ * (*L_);
  }
  
  Eigen::MatrixXd get_zu(){
    return Z_ * (*u_);
  }
  
  // log probability for MCMC
  // u is transformed here
  double log_prob(const Eigen::VectorXd &v){
    Eigen::VectorXd mu = xb_ + ZL_*v;
    Eigen::ArrayXd ll(n_);
    Eigen::ArrayXd lp(Q_);
#pragma omp parallel for
    for(int i = 0; i < n_; i++){
      ll(i) = glmmr::maths::log_likelihood(y_(i),mu(i),var_par_,family_,link_);
    }
    
#pragma omp parallel for
    for(int i = 0; i < Q_; i++){
      lp(i) = glmmr::maths::log_likelihood(v(i),0,1,family_,link_);
    }
    
    return ll.sum() + lp.sum();
  }
  
  // update this for all the gradient in glmmr maths
  Eigen::VectorXd log_grad(const Eigen::VectorXd &v){
    
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(v.size());
    Eigen::VectorXd mu = xb_ + ZL_*v;
    
    
    const static std::unordered_map<std::string,int> string_to_case{
      {"poissonlog",1},
      {"poissonidentity",2},
      {"binomiallogit",3},
      {"binomiallog",4},
      {"binomialidentity",5},
      {"binomialprobit",6},
      {"gaussianidentity",7},
      {"gaussianlog",8}
    };
    
    switch (string_to_case.at(family_+link_)){
    case 1:
    {
      for(int i = 0; i < mu.size(); i++)mu(i) = exp(mu(i));
      grad = ZL_.transpose()*(y_-mu) - v;
      break;
    }
    case 2:
    {
      for(int i = 0; i < mu.size(); i++)mu(i) = y_(i)/mu(i) - 1;
      grad =  ZL_.transpose()*mu - v;
      break;
    }
    case 3:
      for(int i = 0; i < mu.size(); i++){
        if(y_(i)==1){
          mu(i) = 1/(exp(mu(i))+1);
        } else if(y_(i)==0){
          mu(i) = exp(mu(i))/(1+exp(mu(i)));
        }
      }
      grad =  ZL_.transpose()*mu - v;
      break;
    case 4:
      for(int i = 0; i < mu.size(); i++){
        if(y_(i)==1){
          mu(i) = 1;
        } else if(y_(i)==0){
          mu(i) = exp(mu(i))/(1-exp(mu(i)));
        }
      }
      grad =  ZL_.transpose()*mu - v;
      break;
    case 5: // need to update
      break;
    case 6: //need to update
      break;
    case 7:
      {
        Eigen::VectorXd resid = y_ - mu;
        grad = (-1/(var_par_*var_par_))*ZL_.transpose()*resid - v;
        break;
      }
    case 8: //need to update
      break;
    }
   
    return grad;
  }
    
  
  // log likelihood 
  
  double log_likelihood() {
    Eigen::ArrayXd ll = Eigen::ArrayXd::Zero(niter_);
    Eigen::MatrixXd zd = Z_ * (*u_);
#pragma omp parallel for
    for(int j=0; j<niter_ ; j++){
      for(int i = 0; i<n_; i++){
        ll(j) += glmmr::maths::log_likelihood(y_(i),xb_(i) + zd.col(j)(i),var_par_,family_,link_);
      }
    } 
    return ll.mean();
  }
  
  
};


}

#endif