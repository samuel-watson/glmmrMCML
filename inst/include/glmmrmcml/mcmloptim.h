#ifndef MCMLCLASS_H
#define MCMLCLASS_H

#include <rbobyqa.h>
#include "likelihood.h"
#include <RcppEigen.h>
#include <glmmr/maths.h>
#include "mcmlmodel.h"

using namespace rminqa;

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {

template<typename T>
class mcmloptim{
public:
  mcmloptim(
    T* D,
    glmmr::mcmlModel* M,
    const Eigen::ArrayXd &start,
    int trace
  ):
  D_(D), M_(M), start_(start), trace_(trace) {
    Q_ = D->data_->n_cov_pars();
    cov_par_fix_ = start.segment(M_->P_,Q_);
    beta_ = start.segment(0,M_->P_); 
    theta_ = start.segment(M_->P_,Q_);
    sigma_ = M_->family_=="gaussian" ? start(M_->P_+Q_) : 0;
    for(int i = 0; i< M_->P_; i++){
      lower_b_.push_back(R_NegInf);
      upper_b_.push_back(R_PosInf);
    }
    for(int i = 0; i< Q_; i++){
      lower_t_.push_back(1e-6);
      upper_t_.push_back(R_PosInf);
    }
  }
  
  T* D_;
  glmmr::mcmlModel* M_;
  int Q_;
  Eigen::ArrayXd cov_par_fix_;
  Eigen::ArrayXd start_;
  int trace_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd theta_;
  double sigma_;
  std::vector<double> lower_b_;
  std::vector<double> upper_b_;
  std::vector<double> lower_t_;
  std::vector<double> upper_t_;
  
  // optimise multivariate normal distribution of random effects
  void d_optim(){
    glmmr::likelihood::D_likelihood<T> ddl(D_,M_->u_);
    Rbobyqa<glmmr::likelihood::D_likelihood<T>> opt;
    opt.set_upper(upper_t_);
    opt.set_lower(lower_t_);
    opt.control.iprint = trace_;
    std::vector<double> start_t;
    start_t.resize(Q_);
    Eigen::VectorXd::Map(&start_t[0], theta_.size()) = theta_;
    opt.minimize(ddl, start_t);
    std::vector<double> theta = opt.par();
    theta_ = Eigen::Map<Eigen::ArrayXd>(theta.data(),Q_);
  }
  
  // optimise likelihood conditional on random effects
  void l_optim(){
    glmmr::likelihood::L_likelihood ldl(M_);
    Rbobyqa<glmmr::likelihood::L_likelihood> opt;
    opt.control.iprint = trace_;
    std::vector<double> start_b;
    start_b.resize(M_->P_);
    Eigen::VectorXd::Map(&start_b[0], beta_.size()) = beta_;
    std::vector<double> lower_b = lower_b_;
    if(M_->family_=="gaussian"){
      start_b.push_back(sigma_);
      lower_b.push_back(0.0);
    }
    opt.set_lower(lower_b);
    opt.minimize(ldl, start_b);
    std::vector<double> beta = opt.par();
    beta_ = Eigen::Map<Eigen::ArrayXd>(beta.data(),M_->P_);//opt.par().subvec(0,P_-1);
    if(M_->family_=="gaussian") sigma_ = beta[M_->P_];
  }
  
  // jointly optimise marginal y and random effects models
  void f_optim(){
    glmmr::likelihood::F_likelihood<T> dl(D_,M_,cov_par_fix_,true,true,sigma_);
    Rbobyqa<glmmr::likelihood::F_likelihood<T>> opt;
    std::vector<double> lower = lower_b_;
    std::vector<double> allpars;
    allpars.resize(M_->P_);
    Eigen::VectorXd::Map(&allpars[0], beta_.size()) = beta_;
    for(int i=0; i< theta_.size(); i++) allpars.push_back(theta_(i));
    if(M_->family_=="gaussian"){
      lower.push_back(0.0);
      allpars.push_back(sigma_);
    }
    for(int i=0; i< theta_.size(); i++) lower.push_back(lower_t_[i]);
    opt.set_lower(lower);
    opt.control.iprint = trace_;
    opt.minimize(dl, allpars);
    std::vector<double> beta = opt.par();
    beta_ = Eigen::Map<Eigen::ArrayXd>(beta.data(),M_->P_);//beta_ = pars(0,P_-1);
    theta_ = Eigen::Map<Eigen::ArrayXd>(beta.data()+M_->P_,Q_);//theta_ = pars(P_,P_+Q_-1);
    if(M_->family_=="gaussian") sigma_ = beta[M_->P_+Q_];
  }
  
  // newton raphson step
  void mcnr(){
    Eigen::ArrayXd sigmas(M_->niter_);
    int P = M_->P_;
    int niter = M_->niter_;
    int n = M_->n_;
    Eigen::MatrixXd XtXW = Eigen::MatrixXd::Zero(P*niter,P);
    Eigen::MatrixXd Wu = Eigen::MatrixXd::Zero(n,niter);
    //check if parallelisation is faster or not! if not then declare vectors in advance
    Eigen::MatrixXd zd = M_->get_zu();//M_->Z_ * (*(M_->u_));
    Eigen::VectorXd xb = M_->xb_;
//#pragma omp parallel for
    for(int i = 0; i < niter; ++i){
      //Eigen::VectorXd zd = Z_ * u_.col(i);
      Eigen::VectorXd zdu = glmmr::maths::mod_inv_func(xb + zd.col(i), M_->link_);
      Eigen::ArrayXd resid = (M_->y_ - zdu).array();
      sigmas(i) = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
      Eigen::VectorXd wdiag = glmmr::maths::dhdmu(xb + zd.col(i),M_->family_,M_->link_);
      for(int j=0; j<n; j++){
        double var = (M_->family_=="gaussian" || M_->family_=="gamma") ? sigmas(i) : 1.0;
        XtXW.block(P*i, 0, P, P) +=  (var/(wdiag(j)*wdiag(j)))*((M_->X_.row(j).transpose()) * (M_->X_.row(j)));
        Wu(j,i) += (var/wdiag(j))*resid(j);
      }
    }
    XtXW *= (double)1/niter;
    Eigen::MatrixXd XtWXm = XtXW.block(0,0,P,P);
    for(int i = 1; i<niter; i++) XtWXm += XtXW.block(P*i,0,P,P);
    XtWXm = XtWXm.inverse();
    Eigen::VectorXd Wum = Wu.rowwise().mean();
    Eigen::VectorXd bincr = XtWXm * (M_->X_.transpose()) * Wum;
    beta_ += bincr;
    sigma_ = sigmas.mean();
  }
  
  //hessian of model parametere
  Eigen::MatrixXd b_hessian(double tol = 1e-4){
    glmmr::likelihood::L_likelihood hdl(M_,true,sigma_);
    std::vector<double> ndep;
    for(int i = 0; i < M_->P_; i++) ndep.push_back(tol);
    hdl.os.ndeps_ = ndep;
    std::vector<double> hessian(M_->P_ * M_->P_,0.0);
    std::vector<double> start_b;
    for(int i = 0; i < M_->P_; i++) start_b.push_back(beta_(i));
    hdl.Hessian(start_b,hessian);
    Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hessian.data(),M_->P_,M_->P_);
    return hess;
  }
  

  Eigen::VectorXd f_grad(double tol = 1e-4){
    glmmr::likelihood::F_likelihood<T> fdl(D_,M_,cov_par_fix_,false,true,sigma_);
    fdl.os.usebounds_ = 1;
    std::vector<double> lower = lower_b_;
    std::vector<double> upper = upper_b_;
    std::vector<double> start_b;
    for(int i = 0; i < M_->P_; i++) start_b.push_back(beta_(i));
    for(int i=0; i< theta_.size(); i++){
      lower.push_back(lower_t_[i]);
      upper.push_back(upper_t_[i]);
      start_b.push_back(theta_(i));
    } 
    fdl.os.lower_ = lower;
    fdl.os.upper_ = upper;
    std::vector<double> ndep;
    for(int i = 0; i < (M_->P_+Q_); i++) ndep.push_back(tol);
    fdl.os.ndeps_ = ndep;
    std::vector<double> gradient(M_->P_+Q_,0.0);
    fdl.Gradient(start_b,gradient);
    Eigen::VectorXd grad = Eigen::Map<Eigen::VectorXd>(gradient.data(),M_->P_+Q_);
    return grad;
  }
  
  Eigen::MatrixXd f_hess(double tol = 1e-4){
    glmmr::likelihood::F_likelihood<T> fhdl(D_,M_,cov_par_fix_,false,true,sigma_);
    fhdl.os.usebounds_ = 1;
    std::vector<double> lower = lower_b_;
    std::vector<double> upper = upper_b_;
    std::vector<double> start_b;
    for(int i = 0; i < M_->P_; i++) start_b.push_back(beta_(i));
    for(int i=0; i< theta_.size(); i++){
      lower.push_back(lower_t_[i]);
      upper.push_back(upper_t_[i]);
      start_b.push_back(theta_(i));
    } 
    fhdl.os.lower_ = lower;
    fhdl.os.upper_ = upper;
    std::vector<double> ndep;
    for(int i = 0; i < (M_->P_+Q_); i++) ndep.push_back(tol);
    fhdl.os.ndeps_ = ndep;
    std::vector<double> hessian((M_->P_+Q_) * (M_->P_+Q_),0.0);

    fhdl.Hessian(start_b,hessian);
    Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hessian.data(),M_->P_+Q_,M_->P_+Q_);
    return hess;
  }
  
  Eigen::VectorXd get_beta(){
    return beta_;
  }
  
  Eigen::VectorXd get_theta(){
    return theta_;
  }
  
  double get_sigma(){
    return sigma_;
  }
  
};

}



#endif