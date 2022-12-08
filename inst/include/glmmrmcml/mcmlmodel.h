#ifndef MCMLMODEL_H
#define MCMLMODEL_H

#include <RcppEigen.h>
#include <glmmr.h>
#include <boost/math/special_functions/digamma.hpp>
#include "moremaths.h"

#ifdef _OPENMP
#include <omp.h>     
#else
// for machines with compilers void of openmp support
#define omp_get_num_threads()  1
#define omp_get_thread_num()   0
#define omp_get_max_threads()  1
#define omp_get_thread_limit() 1
#define omp_get_num_procs()    1
#define omp_set_nested(a)   // empty statement to remove the call
#define omp_get_wtime()        0
#endif

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

namespace glmmr {

class mcmlModel {
public:
  const Eigen::MatrixXd &X_;
  const Eigen::MatrixXd &Z_;
  Eigen::VectorXd xb_;
  Eigen::MatrixXd zu_;
  Eigen::MatrixXd ZL_;
  Eigen::MatrixXd* L_;
  Eigen::MatrixXd W_;
  //Eigen::MatrixXd ZtZ_;
  Eigen::VectorXd y_;
  //Eigen::VectorXd Zyxb_;
  Eigen::MatrixXd* u_;
  double var_par_;
  std::string family_; 
  std::string link_;
  Eigen::MatrixXd D_;
  int n_;
  int Q_;
  int P_;
  int niter_;
  int flink;
  
  mcmlModel(
    const Eigen::MatrixXd &Z, 
    Eigen::MatrixXd* L,
    const Eigen::MatrixXd &X,
    Eigen::VectorXd y,
    Eigen::MatrixXd *u,
    Eigen::VectorXd beta,
    double var_par,
    std::string family, 
    std::string link
  ) : X_(X),  Z_(Z), xb_(X.rows()),zu_(X.rows(),u->cols()), ZL_(X.rows(),Z.cols()), L_(L),
       W_(Eigen::MatrixXd::Identity(X.rows(),X.rows())), y_(y),  u_(u), var_par_(var_par), 
       family_(family), link_(link),
       D_(Z.cols(),Z.cols()){ //ZtZ_(Z.cols(),Z.cols()),Zyxb_(Z.cols()),
    xb_ = X* beta;
    if(L != nullptr){
      ZL_ = Z * (*L); 
      D_ = (*L_) * (*L_).transpose();
    } 
    n_ = X.rows();
    Q_ = Z.cols();
    P_ = X.cols();
    niter_ = u->cols();
    const static std::unordered_map<std::string,int> string_to_case{
      {"poissonlog",1},
      {"poissonidentity",2},
      {"binomiallogit",3},
      {"binomiallog",4},
      {"binomialidentity",5},
      {"binomialprobit",6},
      {"gaussianidentity",7},
      {"gaussianlog",8},
      {"gammalog",9},
      {"gammainverse",10},
      {"gammaidentity",11},
      {"betalogit",12}
    };
    
    flink = string_to_case.at(family_+link_);
    if(flink == 8){
      y_ = y_.log();
    }
    
    update_W();
    //ZtZ_ = ZL_.transpose()*ZL_;
    //ZtZ_.noalias() += Eigen::MatrixXd::Identity(ZtZ_.rows(),ZtZ_.cols());
    //Zyxb_ = ZL_.transpose()*(y_ - xb_);
  }
  
  void update_beta(const Eigen::VectorXd &beta){
    xb_ = X_* beta;
  }
  
  void update_L(){
    ZL_ = Z_ * (*L_);
  }
  
  void update_D(){
    D_ = (*L_) * (*L_).transpose();
  }
  
  Eigen::MatrixXd get_zu(){
    return Z_ * (*u_);
  }
  
  void update_zu(){
    zu_ = Z_ * (*u_);
  }
  
  void update_W(int i = 0){
    zu_ = Z_ * (*u_);
    Eigen::VectorXd w = glmmr::maths::dhdmu(xb_ + zu_.col(i),family_,link_);
    double nvar_par = 1.0;
    if(family_=="gaussian"){
      nvar_par *= var_par_*var_par_;
    } else if(family_=="Gamma"){
      nvar_par *= var_par_;
    } else if(family_=="beta"){
      nvar_par *= (1+var_par_);
    }
    for(int i = 0; i < n_; i++){
      W_(i,i) = 1/(w(i)*nvar_par);
    }
  }
  
  // log probability for MCMC
  // u is transformed here
  double log_prob(const Eigen::VectorXd &v){
    Eigen::VectorXd mu = xb_ + ZL_*v;
    Eigen::ArrayXd ll(n_);
    Eigen::ArrayXd lp(v.size());

#pragma omp parallel for
    for(int i = 0; i < n_; i++){
      ll(i) = glmmr::maths::log_likelihood(y_(i),mu(i),var_par_,flink);
    }
#pragma omp parallel for
    for(int i = 0; i < v.size(); i++){
      lp(i) = glmmr::maths::log_likelihood(v(i),0,1,7);
    }
    return ll.sum() + lp.sum();

  }
  
  // LOG GRADIENT
  Eigen::VectorXd log_grad(const Eigen::VectorXd &v,
                           bool usezl = true){
    
    Eigen::VectorXd grad(v.size()); //Eigen::VectorXd::Zero(v.size());
    Eigen::VectorXd mu = xb_;
    if(usezl){
      mu += ZL_*v;
      grad = -1.0*v;
    } else {
      mu += Z_*v;
      grad = -1.0*D_*v;
    }
    
    switch (flink){
    case 1:
    {
      mu = (mu.array().exp()).matrix();
      grad.noalias() += ZL_.transpose()*(y_-mu);
      break;
    }
    case 2:
    {
      mu = (mu.array().inverse()).matrix();
      mu = (y_.array()*mu.array()).matrix();
      mu -= Eigen::VectorXd::Ones(mu.size());
      grad.noalias() +=  ZL_.transpose()*mu;
      break;
    }
    case 3:
      {
        mu = (mu.array().exp()).matrix();
        mu += Eigen::VectorXd::Ones(mu.size());
        mu = (mu.array().inverse()).matrix();
        mu += y_;
        mu -= Eigen::VectorXd::Ones(mu.size());
        grad.noalias() +=  ZL_.transpose()*mu;
        break;
      }
    case 4:
      {
#pragma omp parallel for
        for(int i = 0; i < mu.size(); i++){
          if(y_(i)==1){
            mu(i) = 1;
          } else if(y_(i)==0){
            mu(i) = exp(mu(i))/(1-exp(mu(i)));
          }
        }
        grad.noalias() +=  ZL_.transpose()*mu;
        break;
      }
    case 5: 
      {
#pragma omp parallel for
        for(int i = 0; i < mu.size(); i++){
          if(y_(i)==1){
            mu(i) = 1/mu(i);
          } else if(y_(i)==0){
            mu(i) = -1/(1-mu(i));
          }
        }
        grad.noalias() +=  ZL_.transpose()*mu;
        break;
      }
    case 6:
      {
#pragma omp parallel for
        for(int i = 0; i < mu.size(); i++){
          if(y_(i)==1){
            mu(i) = (double)R::dnorm(mu(i),0,1,false)/((double)R::pnorm(mu(i),0,1,true,false));
          } else if(y_(i)==0){
            mu(i) = -1.0*(double)R::dnorm(mu(i),0,1,false)/(1-(double)R::pnorm(mu(i),0,1,true,false));
          }
        }
        grad.noalias() +=  ZL_.transpose()*mu;
        break;
      }
    case 7:
      {
        grad.noalias() += (1.0/(var_par_*var_par_))*(ZL_.transpose()*(y_ - mu));
        //grad.noalias() += (1.0/(var_par_*var_par_))*(Zyxb_ - ZtZ_*v);
        break;
      }
    case 8: //need to update
      {
        grad.noalias() += (1.0/(var_par_*var_par_))*(ZL_.transpose()*(y_ - mu));
        //grad.noalias() += (1.0/(var_par_*var_par_))*(Zyxb_ - ZtZ_*v);
        break;
      }
    case 9:
      {
        mu *= -1.0;
        mu = (mu.array().exp()).matrix();
// #pragma omp parallel for
//         for(int i = 0; i < mu.size(); i++)mu(i) = exp(-1.0*mu(i));
        grad.noalias() += ZL_.transpose()*(y_.array()*mu.array()-1).matrix()*var_par_;
        break;
      }
    case 10:
      {
        mu = (mu.array().inverse()).matrix();
        grad.noalias() += ZL_.transpose()*(mu-y_)*var_par_;
        break;
      }
    case 11:
      {
        mu = (mu.array().inverse()).matrix();
        grad.noalias() += ZL_.transpose()*((y_.array()*mu.array()*mu.array()).matrix() - mu)*var_par_;
        break;
      }
    case 12:
      {
#pragma omp parallel for
        for(int i = 0; i < mu.size(); i++){
          mu(i) = exp(mu(i))/(exp(mu(i))+1);
          mu(i) = (mu(i)/(1+exp(mu(i)))) * var_par_ * (log(y_(i)) - log(1- y_(i)) - boost::math::digamma(mu(i)*var_par_) + boost::math::digamma((1-mu(i))*var_par_));
        }
        grad.noalias() += ZL_.transpose()*mu;
        break;
      }
    }
   
    return grad;
  }
    
  
  
  
  double log_likelihood() {
    Eigen::ArrayXd ll = Eigen::ArrayXd::Zero(niter_);
    Eigen::MatrixXd zd = Z_ * (*u_);

#pragma omp parallel for
    for(int j=0; j<niter_ ; j++){
      for(int i = 0; i<n_; i++){
        ll(j) += glmmr::maths::log_likelihood(y_(i),xb_(i) + zd.col(j)(i),var_par_,flink);
      }
    }

    
    return ll.mean();
  }
  
  
};


}

#endif