#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

#include <rbobyqa.h>
#include <RcppEigen.h>
#include "moremaths.h"
#include "mcmlmodel.h"

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

using namespace rminqa;


// [[Rcpp::depends(RcppEigen)]]


namespace glmmr{
namespace likelihood {

template<typename T>
class D_likelihood : public Functor<std::vector<double> > {
  T* D_;
  Eigen::MatrixXd* u_;
  
public:
  D_likelihood(T* D,
               Eigen::MatrixXd* u) : 
  D_(D), u_(u) {}
  double operator()(const std::vector<double> &par) {
    int nrow = u_->cols();
    D_->update_parameters(par);
    double logl = D_->loglik((*u_));
    return -1*logl;
  }
};

class L_likelihood : public Functor<std::vector<double> > {
  glmmr::mcmlModel* M_;
  bool fix_var_;
  double fix_var_par_;
public:
  L_likelihood(glmmr::mcmlModel* M,
               bool fix_var = false,
               double fix_var_par = 0) :  
  M_(M), fix_var_(fix_var), fix_var_par_(fix_var_par){}
  double operator()(const std::vector<double> &par) {
    std::vector<double> par2 = par;
    Eigen::Map<Eigen::VectorXd> beta(par2.data(),M_->P_); 
    M_->update_beta(beta);
    M_->var_par_ = (M_->family_=="gaussian" || M_->family_=="Gamma" || M_->family_=="beta")&&!fix_var_ ? par[M_->P_] : fix_var_par_;
    double ll = M_->log_likelihood();
    return -1*ll;
  }
};

template<typename T>
class F_likelihood : public Functor<std::vector<double> > {
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
  
  
  double operator()(const std::vector<double> &par) {
    int Q = cov_par_fix_.size();
    std::vector<double> par2 = par;
    Eigen::Map<Eigen::VectorXd> beta(par2.data(),M_->P_); 
    std::vector<double> theta;
    for(int i=0; i<Q; i++)theta.push_back(par2[M_->P_+i]);
    M_->update_beta(beta);
    M_->var_par_ = (M_->family_=="gaussian" || M_->family_=="Gamma" || M_->family_=="beta")&&!fix_var_ ? par[M_->P_] : fix_var_par_;
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

template<typename T>
class LA_likelihood : public Functor<std::vector<double> > {
  glmmr::mcmlModel* M_;
  T* D_;
public:
  LA_likelihood(glmmr::mcmlModel* M,
                T* D) :  
  M_(M), D_(D){}
  double operator()(const std::vector<double> &par) {
    std::vector<double> par2 = par;
    
    int Q = par.size() - M_->P_;
    Eigen::VectorXd v(Q);
    for(int i=0; i<Q; i++)v(i) = par2[M_->P_+i];
    //double logl = D_->loglik(v);
    double logl = v.transpose()*v;
    Eigen::Map<Eigen::VectorXd> beta(par2.data(),M_->P_);
    M_->update_beta(beta);
    M_->u_->col(0) = v;
    //double ll = M_->log_likelihood(true);
    Eigen::ArrayXd ll(M_->n_);
    Eigen::VectorXd zd =  M_->ZL_ * v;
#pragma omp parallel for
    for(int i = 0; i<M_->n_; i++){
      ll(i) = glmmr::maths::log_likelihood(M_->y_(i),M_->xb_(i) + zd(i),M_->var_par_,M_->flink);
    }
    
    return -1.0*(ll.sum() - 0.5*logl);
  }
};

template<typename T>
class LA_likelihood_cov : public Functor<std::vector<double> > {
  glmmr::mcmlModel* M_;
  T* D_;
public:
  LA_likelihood_cov(glmmr::mcmlModel* M,
                    T* D) :  
  M_(M), D_(D) {} //, W_(W)const Eigen::MatrixXd &WEigen::MatrixXd* L,L_(L),
  
  double operator()(const std::vector<double> &par) {
    int Q = (M_->family_=="gaussian" || M_->family_=="Gamma" || M_->family_=="beta") ? par.size()-1 : par.size();
    std::vector<double> theta;
    for(int i = 0; i < Q; i++)theta.push_back(par[i]);
    if(M_->family_=="gaussian" || M_->family_=="Gamma" || M_->family_=="beta")M_->var_par_ = par[Q];
    D_->gamma_ = Eigen::Map<Eigen::VectorXd>(theta.data(),theta.size()); 
    //double logl = D_->loglik((*(M_->u_)));
    double logl = M_->u_->col(0).transpose() * M_->u_->col(0);
    Eigen::MatrixXd L = D_->genD(0,true,false);
    Eigen::MatrixXd ZL = M_->Z_ * L;
    //M_->update_L();
    
    Eigen::ArrayXd ll(M_->n_);
    Eigen::VectorXd zd =  ZL * M_->u_->col(0);
#pragma omp parallel for
    for(int i = 0; i<M_->n_; i++){
      ll(i) = glmmr::maths::log_likelihood(M_->y_(i),M_->xb_(i) + zd(i),M_->var_par_,M_->flink);
    }
    
    double nvar_par = 1.0;
    if(M_->family_=="gaussian"){
      nvar_par *= M_->var_par_*M_->var_par_;
    } else if(M_->family_=="Gamma"){
      nvar_par *= M_->var_par_;
    } else if(M_->family_=="beta"){
      nvar_par *= (1+M_->var_par_);
    }
    Eigen::VectorXd w = glmmr::maths::dhdmu(M_->xb_ + zd,M_->family_,M_->link_);
    w = (w.array().inverse()).matrix();
    w *= 1/nvar_par;
    
    //double ll = M_->log_likelihood(true);
    //Eigen::MatrixXd D = D_->genD(0,false,false);
    //Eigen::MatrixXd LZWZL = ((M_->Z_).transpose()) * M_->W_ * (M_->Z_);
    Eigen::MatrixXd LZWZL = ZL.transpose() * w.asDiagonal() * ZL;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(M_->Q_,M_->Q_);
    LZWZL.noalias() += I;//D.llt().solve(I);
    double LZWdet = glmmr::maths::logdet(LZWZL);
    
    return -1*(ll.sum() - 0.5*logl - 0.5*LZWdet);
  }
};

template<typename T>
class LA_likelihood_btheta : public Functor<std::vector<double> > {
  glmmr::mcmlModel* M_;
  T* D_;
public:
  LA_likelihood_btheta(glmmr::mcmlModel* M,
                    T* D) :  
  M_(M), D_(D){}
  
  double operator()(const std::vector<double> &par) {
    
    int Q = par.size() - M_->P_;
    if(M_->family_=="gaussian")Q--;
    std::vector<double> par2 = par;
    Eigen::Map<Eigen::VectorXd> beta(par2.data(),M_->P_); 
    std::vector<double> theta;
    for(int i=0; i<Q; i++)theta.push_back(par2[M_->P_+i]);
    if(M_->family_=="gaussian")M_->var_par_ = par2[par2.size()-1];
    
    M_->update_beta(beta);
    //M_->update_W();
    D_->update_parameters(theta);
    // *(M_->L_) = D_->genD(0,true,false);
    // M_->update_L();
    //double logl = D_->loglik((*(M_->u_)));
    Eigen::MatrixXd L = D_->genD(0,true,false);
    Eigen::MatrixXd ZL = M_->Z_ * L;
    double logl = M_->u_->col(0).transpose() * M_->u_->col(0);
    //double ll = M_->log_likelihood(true);
    
    Eigen::ArrayXd ll(M_->n_);
    Eigen::VectorXd zd =  ZL * M_->u_->col(0);
    
    double nvar_par = 1.0;
    if(M_->family_=="gaussian"){
      nvar_par *= M_->var_par_*M_->var_par_;
    } else if(M_->family_=="Gamma"){
      nvar_par *= M_->var_par_;
    } else if(M_->family_=="beta"){
      nvar_par *= (1+M_->var_par_);
    }
    Eigen::VectorXd w = glmmr::maths::dhdmu(M_->xb_ + zd,M_->family_,M_->link_);
    w = (w.array().inverse()).matrix();
    w *= 1/nvar_par;
    
#pragma omp parallel for
    for(int i = 0; i<M_->n_; i++){
      ll(i) = glmmr::maths::log_likelihood(M_->y_(i),M_->xb_(i) + zd(i),M_->var_par_,M_->flink);
    }
    
    
    //Eigen::MatrixXd D = D_->genD(0,false,false);
    //Eigen::MatrixXd LZWZL = ((M_->Z_).transpose()) * M_->W_ * (M_->Z_);
    Eigen::MatrixXd LZWZL = ZL.transpose() * w.asDiagonal() * ZL;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(M_->Q_,M_->Q_);
    LZWZL.noalias() += I;//D.llt().solve(I);
    double LZWdet = glmmr::maths::logdet(LZWZL);
    
    return -1*(ll.sum()-0.5*logl-0.5*LZWdet);
  }
};

}
}


#endif