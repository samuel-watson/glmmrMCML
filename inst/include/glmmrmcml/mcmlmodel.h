#ifndef MCMLMODEL_H
#define MCMLMODEL_H

#include <RcppEigen.h>
#include <glmmr.h>
#include <boost/math/special_functions/digamma.hpp>
#include "moremaths.h"

#ifdef _OPENMP
#include <omp.h>     
#else
#define omp_get_thread_num() 0
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
  Eigen::MatrixXd ZL_;
  Eigen::MatrixXd* L_;
  //Eigen::MatrixXd ZtZ_;
  Eigen::VectorXd y_;
  //Eigen::VectorXd Zyxb_;
  Eigen::MatrixXd* u_;
  double var_par_;
  std::string family_; 
  std::string link_;
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
  ) : X_(X),  Z_(Z), xb_(X.rows()), ZL_(X.rows(),Z.cols()), L_(L),  
       y_(y),  u_(u), var_par_(var_par), family_(family), link_(link) { //ZtZ_(Z.cols(),Z.cols()),Zyxb_(Z.cols()),
    xb_ = X* beta;
    if(L != nullptr)ZL_ = Z * (*L);  
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
    //ZtZ_ = ZL_.transpose()*ZL_;
    //ZtZ_.noalias() += Eigen::MatrixXd::Identity(ZtZ_.rows(),ZtZ_.cols());
    //Zyxb_ = ZL_.transpose()*(y_ - xb_);
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
  
  // update this for all the gradient in glmmr maths
  Eigen::VectorXd log_grad(const Eigen::VectorXd &v){
    
    Eigen::VectorXd grad = -1.0*v; //Eigen::VectorXd::Zero(v.size());
    
    switch (flink){
    case 1:
    {
      Eigen::VectorXd mu = xb_ + ZL_*v;
      mu = (mu.array().exp()).matrix();
// #pragma omp parallel for
//       for(int i = 0; i < mu.size(); i++)mu(i) = exp(mu(i));
      grad.noalias() += ZL_.transpose()*(y_-mu);
      break;
    }
    case 2:
    {
      Eigen::VectorXd mu = xb_ + ZL_*v;
      mu = (mu.array().inverse()).matrix();
      mu = (y_.array()*mu.array()).matrix();
      mu -= Eigen::VectorXd::Ones(mu.size());
// #pragma omp parallel for
//       for(int i = 0; i < mu.size(); i++)mu(i) = y_(i)/mu(i) - 1;
      grad.noalias() +=  ZL_.transpose()*mu;
      break;
    }
    case 3:
      {
        Eigen::VectorXd mu = xb_ + ZL_*v;
        mu = (mu.array().exp()).matrix();
        mu += Eigen::VectorXd::Ones(mu.size());
        mu = (mu.array().inverse()).matrix();
        mu += y_;
        mu -= Eigen::VectorXd::Ones(mu.size());
// #pragma omp parallel for
//         for(int i = 0; i < mu.size(); i++){
//           if(y_(i)==1){
//             mu(i) = 1/(exp(mu(i))+1);
//           } else if(y_(i)==0){
//             mu(i) = 1/(exp(mu(i))+1) - 1;
//           }
//         }
        grad.noalias() +=  ZL_.transpose()*mu;
        break;
      }
    case 4:
      {
        Eigen::VectorXd mu = xb_ + ZL_*v;
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
        Eigen::VectorXd mu = xb_ + ZL_*v;
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
        Eigen::VectorXd mu = xb_ + ZL_*v;
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
        Eigen::VectorXd mu = xb_ + ZL_*v;
        grad.noalias() += (1.0/(var_par_*var_par_))*(ZL_.transpose()*(y_ - mu));
        //grad.noalias() += (1.0/(var_par_*var_par_))*(Zyxb_ - ZtZ_*v);
        break;
      }
    case 8: //need to update
      {
        Eigen::VectorXd mu = xb_ + ZL_*v;
        grad.noalias() += (1.0/(var_par_*var_par_))*(ZL_.transpose()*(y_ - mu));
        //grad.noalias() += (1.0/(var_par_*var_par_))*(Zyxb_ - ZtZ_*v);
        break;
      }
    case 9:
      {
        Eigen::VectorXd mu = xb_ + ZL_*v;
        mu *= -1.0;
        mu = (mu.array().exp()).matrix();
// #pragma omp parallel for
//         for(int i = 0; i < mu.size(); i++)mu(i) = exp(-1.0*mu(i));
        grad.noalias() += ZL_.transpose()*(y_.array()*mu.array()-1).matrix()*var_par_;
        break;
      }
    case 10:
      {
        Eigen::VectorXd mu = ((xb_ + ZL_*v).array().inverse()).matrix();
        grad.noalias() += ZL_.transpose()*(mu-y_)*var_par_;
        break;
      }
    case 11:
      {
        Eigen::VectorXd mu = ((xb_ + ZL_*v).array().inverse()).matrix();
        grad.noalias() += ZL_.transpose()*((y_.array()*mu.array()*mu.array()).matrix() - mu)*var_par_;
        break;
      }
    case 12:
      {
        Eigen::VectorXd mu = xb_ + ZL_*v;
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