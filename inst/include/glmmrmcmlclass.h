#ifndef GLMMRMCMLCLASS_H
#define GLMMRMCMLCLASS_H

#include <rbobyqa.h>
#include "glmmrmcml.h"

using namespace Rcpp;
using namespace arma;
using namespace rminqa;

//We have to add a second header as rcpp will include the other header 
// in the rcpp exports file, which loads rminqa twice, causing a
// multiple definitions error on windows machines

// [[Rcpp::depends(RcppArmadillo)]]

class D_likelihood : public Functor {
  DMatrix* D_;
  arma::mat u_;
  
public:
  D_likelihood(DMatrix* D,
               const arma::mat &u) : 
  D_(D), u_(u) {}
  double operator()(const vec &par) override{
    arma::uword nrow = u_.n_cols;
    D_->gamma_ = par;
    D_->gen_blocks_byfunc();
    arma::vec logl(nrow);
#pragma omp parallel for
    for(arma::uword j=0;j<nrow;j++){
      logl(j) = D_->loglik(u_.col(j));
    }
    return -1*arma::mean(logl);
  }
};


class L_likelihood : public Functor {
  arma::mat Z_;
  arma::mat X_;
  arma::vec y_;
  arma::mat u_;
  std::string family_;
  std::string link_;
  bool fix_var_;
  double fix_var_par_;
public:
  L_likelihood(const arma::mat &Z, 
               const arma::mat &X,
               const arma::vec &y, 
               const arma::mat &u, 
               std::string family, 
               std::string link,
               bool fix_var = false,
               double fix_var_par = 0) : 
  Z_(Z), X_(X), y_(y), u_(u),family_(family) , link_(link), 
  fix_var_(fix_var), fix_var_par_(fix_var_par){}
  double operator()(const vec &par) override{
    arma::uword niter = u_.n_cols;
    arma::uword P = X_.n_cols;
    arma::vec xb = X_*par.subvec(0,P-1);
    double var_par = family_=="gaussian"&&!fix_var_ ? par(P) : fix_var_par_;
    arma::vec ll(niter,fill::zeros);
    arma::mat zd = Z_ * u_;
#pragma omp parallel for
    for(arma::uword j=0; j<niter ; j++){
      ll(j) += log_likelihood(y_,xb + zd.col(j),var_par,family_,link_);
    }
    
    return -1 * mean(ll);
  }
};

class F_likelihood : public Functor {
  DMatrix* D_;
  arma::mat Z_;
  arma::mat X_;
  arma::vec y_;
  arma::mat u_;
  arma::vec cov_par_fix_;
  std::string family_;
  std::string link_;
  bool importance_;
  bool fix_var_;
  double fix_var_par_;
  
public:
  F_likelihood(DMatrix* D,
               arma::mat Z, 
               arma::mat X,
               arma::vec y, 
               arma::mat u,
               arma::vec cov_par_fix,
               std::string family, 
               std::string link,
               bool importance,
               bool fix_var = false,
               double fix_var_par = 0) : 
  D_(D) ,Z_(Z), X_(X), y_(y), 
  u_(u),cov_par_fix_(cov_par_fix), family_(family) , link_(link),
  importance_(importance),fix_var_(fix_var), fix_var_par_(fix_var_par) {}
  double operator()(const vec &par) override{
    
    arma::uword niter = u_.n_cols;
    arma::uword n = y_.n_elem;
    arma::uword P = X_.n_cols;
    arma::uword Q = cov_par_fix_.n_elem;
    double du;
    arma::vec numerD(niter,fill::zeros);
    arma::uword nrow = u_.n_cols;
    D_->gamma_ = par.subvec(P,P+Q-1);
    D_->gen_blocks_byfunc();
#pragma omp parallel for
    for(arma::uword j=0;j<nrow;j++){
      numerD(j) += D_->loglik(u_.col(j));
    }
    
    // log likelihood for observations
    arma::vec xb(n);
    double var_par = family_=="gaussian"&&!fix_var_ ? par(P+Q) : fix_var_par_;
    xb = X_*par.subvec(0,P-1);
    arma::vec lfa(niter,fill::zeros);
    arma::mat zd = Z_ * u_;
#pragma omp parallel for
    for(arma::uword j=0; j<niter ; j++){
      lfa(j) += log_likelihood(y_,
          xb + zd.col(j),
          var_par,
          family_,
          link_);
    }
    
    if(importance_){
      // denominator density for importance sampling
      arma::vec denomD(niter,fill::zeros);
      D_->gamma_ = cov_par_fix_;
      D_->gen_blocks_byfunc();
#pragma omp parallel for
      for(arma::uword j=0;j<nrow;j++){
        denomD(j) += D_->loglik(u_.col(j));
      }
      du = 0;
      for(arma::uword j=0;j<niter;j++){
        du  += exp(lfa(j)+numerD(j))/exp(denomD(j));
      }
      
      du = -1 * log(du/niter);
    } else {
      du = -1* (mean(numerD) + mean(lfa));
    }
    
    return du;
  }
};

class mcmloptim{
public:
  mcmloptim(
    DMatrix* D,
    const arma::mat &Z, 
    const arma::mat &X,
    const arma::vec &y, 
    const arma::mat &u,
    const arma::vec &cov_par_fix,
    std::string family, 
    std::string link,
    const arma::vec &start,
    int trace
  ):
  D_(D),Z_(Z), X_(X), y_(y), 
  u_(u),cov_par_fix_(cov_par_fix), family_(family) , link_(link),
  start_(start), trace_(trace) {
    P_ = X_.n_cols;
    Q_ = cov_par_fix_.size();
    beta_ = start.subvec(0,P_-1);
    theta_ = start.subvec(P_,P_+Q_-1);
    sigma_ = family_=="gaussian" ? start(P_+Q_) : 0;
    n_ = y_.n_elem;
    niter_ = u_.n_cols;
    lower_b_ = arma::zeros<arma::vec>(P_);
    lower_b_.for_each([](arma::mat::elem_type &val) { val = R_NegInf; });
    upper_b_ = arma::zeros<arma::vec>(P_);
    upper_b_.for_each([](arma::mat::elem_type &val) { val = R_PosInf; });
    lower_t_ = arma::zeros<arma::vec>(Q_);
    lower_t_.for_each([](arma::mat::elem_type &val) { val = 1e-6; });
    upper_t_ = arma::zeros<arma::vec>(Q_);
    upper_t_.for_each([](arma::mat::elem_type &val) { val = R_PosInf; });
  }
  
  DMatrix* D_;
  const arma::mat Z_;
  const arma::mat X_;
  const arma::vec y_;
  arma::mat u_;
  arma::vec cov_par_fix_;
  std::string family_;
  std::string link_;
  arma::vec start_;
  int trace_;
  arma::uword P_;
  arma::uword Q_;
  arma::vec beta_;
  arma::vec theta_;
  double sigma_;
  arma::uword n_;
  arma::uword niter_;
  arma::vec lower_b_;
  arma::vec upper_b_;
  arma::vec lower_t_;
  arma::vec upper_t_;
  
  void d_optim(){
    D_likelihood ddl(D_,u_);
    
    Rbobyqa<D_likelihood> opt;
    opt.set_upper(upper_t_);
    opt.set_lower(lower_t_);
    opt.control.iprint = trace_;
    arma::vec start_t = start_.subvec(P_,P_+Q_-1);
    opt.minimize(ddl, start_t);
    theta_ = opt.par();
  }
  
  void l_optim(){
    L_likelihood ldl(Z_,X_,y_,u_,family_,link_);
    Rbobyqa<L_likelihood> opt;
    opt.control.iprint = trace_;
    arma::vec start_b = start_.subvec(0,P_-1);
    if(family_=="gaussian")start_b = arma::join_cols(start_b,arma::vec({sigma_}));
    opt.minimize(ldl, start_b);
    beta_ = opt.par().subvec(0,P_-1);
    if(family_=="gaussian") sigma_ = opt.par()(P_);
  }
  
  void f_optim(){
    F_likelihood dl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,true,true,sigma_);
    Rbobyqa<F_likelihood> opt;
    arma::vec lower = lower_b_;
    arma::vec allpars = arma::join_cols(beta_,theta_);
    if(family_=="gaussian"){
      lower = arma::join_cols(lower,arma::vec({0}));
      allpars = arma::join_cols(allpars,arma::vec({sigma_}));
    }
    opt.set_lower(arma::join_cols(lower,lower_t_));
    opt.control.iprint = trace_;
    opt.minimize(dl, allpars);
    arma::vec pars = opt.par();
    beta_ = pars(0,P_-1);
    theta_ = pars(P_,P_+Q_-1);
    if(family_=="gaussian") sigma_ = pars(P_+Q_);
  }
  
  void mcnr(){
    //generate residuals
    arma::vec xb = X_*start_.subvec(0,P_-1);
    arma::vec sigmas(niter_);
    arma::cube XtWX(P_,P_,niter_,fill::zeros);
    arma::mat Wu(n_,niter_,fill::zeros);
    arma::mat W(n_,n_,fill::zeros);
    arma::mat W2(n_,n_,fill::zeros);
#pragma omp parallel for
    for(arma::uword i = 0; i < niter_; ++i){
      arma::vec zd = Z_ * u_.col(i);
      zd = mod_inv_func(xb + zd, link_);
      arma::vec resid = y_ - zd;
      sigmas(i) = arma::stddev(resid);
      arma::vec wdiag = dhdmu(xb + zd,family_,link_);
      arma::vec wdiag2 = 1/arma::pow(wdiag, 2.0);
      if(family_=="gaussian" || family_=="gamma") wdiag2 *= sigmas(i);
      for(arma::uword j = 0; j<n_; j++){
        XtWX.slice(i) += wdiag2(j)*(X_.row(j).t()*X_.row(j));
        Wu(j,i) += wdiag(j)*wdiag2(j)*resid(j); 
      }
    }
    
    for(arma::uword i = 1; i< niter_; ++i){
      XtWX.slice(0) += XtWX.slice(i)*(1.0/niter_);
    }
    arma::mat XtWXm = arma::inv_sympd(XtWX.slice(0));//arma::inv_sympd(XtWX/niter_);
    arma::vec Wum = mean(Wu,1);//Wu/niter_;
    
    arma::vec bincr = XtWXm*X_.t()*Wum;
    beta_ = start_.subvec(0,P_-1) + bincr;
    sigma_ = mean(sigmas);
  }
  
  arma::mat b_hessian(double tol = 1e-4){
    L_likelihood hdl(Z_,X_,y_,u_,family_,link_,true,sigma_);
    hdl.os.ndeps_ = arma::ones<arma::vec>(P_) * tol;
    arma::mat hessian(P_,P_,fill::zeros);
    arma::vec start_b = start_.subvec(0,P_-1);
    hdl.Hessian(start_b,hessian);
    return hessian;
  }
  
  arma::vec f_grad(double tol = 1e-4){
    F_likelihood fdl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,false,true,sigma_);
    fdl.os.usebounds_ = 1;
    fdl.os.lower_ = arma::join_cols(lower_b_,lower_t_);
    fdl.os.upper_ = arma::join_cols(upper_b_,upper_t_);
    fdl.os.ndeps_ = arma::ones<arma::vec>(start_.size()) * tol;
    arma::vec gradient(start_.n_elem,fill::zeros);
    fdl.Gradient(start_,gradient);
    return gradient;
  }
  
  arma::mat f_hess(double tol = 1e-4){
    F_likelihood fhdl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,false,true,sigma_);
    arma::vec lower_b = arma::zeros<arma::vec>(P_);
    lower_b.for_each([](arma::mat::elem_type &val) { val = R_NegInf; });
    arma::vec upper_b = arma::zeros<arma::vec>(P_);
    upper_b.for_each([](arma::mat::elem_type &val) { val = R_PosInf; });
    fhdl.os.usebounds_ = 1;
    fhdl.os.lower_ = arma::join_cols(lower_b_,lower_t_);
    fhdl.os.upper_ = arma::join_cols(upper_b_,upper_t_);
    fhdl.os.ndeps_ = arma::ones<arma::vec>(start_.size()) * tol;
    arma::mat hessian(start_.n_elem,start_.n_elem,fill::zeros);
    fhdl.Hessian(start_,hessian);
    return hessian;
  }
  
  arma::vec get_beta(){
    return beta_;
  }
  
  arma::vec get_theta(){
    return theta_;
  }
  
  double get_sigma(){
    return sigma_;
  }
  
};

#endif