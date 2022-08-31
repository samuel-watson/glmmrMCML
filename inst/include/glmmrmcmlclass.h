#ifndef GLMMRMCMLCLASS_H
#define GLMMRMCMLCLASS_H

#include <rbobyqa.h>
#include <sparsechol.h>
#include "glmmrmcml.h"

using namespace Rcpp;
using namespace arma;
using namespace rminqa;

//We have to add a second header as rcpp will include the other header 
// in the rcpp exports file, which loads rminqa twice, causing a
// multiple definitions error on windows machines 

// [[Rcpp::depends(RcppArmadillo)]]

template<typename T>
class D_likelihood : public Functor {
  T* D_;
  arma::mat u_;
  
public:
  D_likelihood(T* D,
               const arma::mat &u) : 
  D_(D), u_(u) {}
  double operator()(const vec &par) override{
    arma::uword nrow = u_.n_cols;
    D_->update_parameters(par);
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

template<typename T>
class F_likelihood : public Functor {
  T* D_;
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
  F_likelihood(T* D,
               arma::mat Z, 
               arma::mat X,
               arma::vec y, 
               arma::mat u,
               arma::vec cov_par_fix,
               std::string family, 
               std::string link,
               bool importance = false,
               bool fix_var = false,
               double fix_var_par = 0) : 
  D_(D) ,Z_(Z), X_(X), y_(y), 
  u_(u),cov_par_fix_(cov_par_fix), family_(family) , link_(link),
  importance_(importance),fix_var_(fix_var), fix_var_par_(fix_var_par) {}
  
  
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
    arma::uword Q = cov_par_fix_.n_elem;
    D_->update_parameters(par.subvec(P,P+Q-1));
    arma::vec logl(niter);
#pragma omp parallel for
    for(arma::uword j=0;j<niter;j++){
      logl(j) = D_->loglik(u_.col(j));
    }
    
    return -1.0*(arma::mean(ll) + arma::mean(logl));
    
//     arma::uword niter = u_.n_cols;
//     arma::uword n = y_.n_elem;
//     arma::uword P = X_.n_cols;
//     arma::uword Q = cov_par_fix_.n_elem;
//     double du;
//     arma::vec numerD(niter,fill::zeros);
//     arma::uword nrow = u_.n_cols;
//     D_->update_parameters(par.subvec(P,P+Q-1));
// //#pragma omp parallel for
//     for(arma::uword j=0;j<nrow;j++){
//       numerD(j) += D_->loglik(u_.col(j));
//     }
//     arma::vec xb(n);
//     double var_par = family_=="gaussian"&& !fix_var_ ? par(P+Q) : fix_var_par_;
//     Rcpp::Rcout << "\n FF 3" << var_par;
//     xb = X_*par.subvec(0,P-1);
//     arma::vec lfa(niter,fill::zeros);
//     arma::mat zd = Z_ * u_;
// //#pragma omp parallel for
//     for(arma::uword j=0; j<niter ; j++){
//       lfa(j) += log_likelihood(y_,xb + zd.col(j),var_par,family_,link_);
//     }
    
//     if(importance_){
//       // denominator density for importance sampling
//       arma::vec denomD(niter,fill::zeros);
//       D_->update_parameters(cov_par_fix_);
// //#pragma omp parallel for
//       for(arma::uword j=0;j<nrow;j++){
//         denomD(j) += D_->loglik(u_.col(j));
//       }
//       du = 0;
//       for(arma::uword j=0;j<niter;j++){
//         du  += exp(lfa(j)+numerD(j))/exp(denomD(j));
//       }
//       
//       du = -1 * log(du/niter);
//     } else {
//       du = -1* (mean(numerD) + mean(lfa));
//     }
//     Rcpp::Rcout << "\n FF Finish " << par.t() << " \ndu: " << du;
//     return du;
  }
};

template<typename T>
class mcmloptim{
public:
  mcmloptim(
    T* D,
    const arma::mat &Z, 
    const arma::mat &X,
    const arma::vec &y, 
    const arma::mat &u,
    int Q,
    std::string family, 
    std::string link,
    const arma::vec &start,
    int trace
  ):
  D_(D),Z_(Z), X_(X), y_(y), 
  u_(u), Q_(Q), family_(family) , link_(link),
  start_(start), trace_(trace) {
    P_ = X_.n_cols;
    cov_par_fix_ = start.subvec(P_,P_+Q_-1);
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
  
  T* D_;
  const arma::mat Z_;
  const arma::mat X_;
  const arma::vec y_;
  arma::mat u_;
  int Q_;
  arma::vec cov_par_fix_;
  std::string family_;
  std::string link_;
  arma::vec start_;
  int trace_;
  arma::uword P_;
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
    D_likelihood<T> ddl(D_,u_);
    
    Rbobyqa<D_likelihood<T>> opt;
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
    if(family_=="gaussian"){
      start_b = arma::join_cols(start_b,arma::vec({sigma_}));
      lower_b_ = arma::join_cols(lower_b_,arma::vec({0.0}));
    }
    opt.set_lower(lower_b_);
    opt.minimize(ldl, start_b);
    beta_ = opt.par().subvec(0,P_-1);
    if(family_=="gaussian") sigma_ = opt.par()(P_);
  }
  
  void f_optim(){
    F_likelihood<T> dl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,true,true,sigma_);
    Rbobyqa<F_likelihood<T>> opt;
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
    F_likelihood<T> fdl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,false,true,sigma_);
    fdl.os.usebounds_ = 1;
    fdl.os.lower_ = arma::join_cols(lower_b_,lower_t_);
    fdl.os.upper_ = arma::join_cols(upper_b_,upper_t_);
    fdl.os.ndeps_ = arma::ones<arma::vec>(P_+Q_) * tol;
    arma::vec gradient(P_+Q_,fill::zeros);
    fdl.Gradient(start_.subvec(0,P_+Q_-1),gradient);
    return gradient;
  }
  
  arma::mat f_hess(double tol = 1e-4){
    F_likelihood<T> fhdl(D_,Z_,X_,y_,u_,cov_par_fix_,family_,link_,false,true,sigma_);
    // fhdl.os.usebounds_ = 1;
    // fhdl.os.lower_ = arma::join_cols(lower_b_,lower_t_);
    // fhdl.os.upper_ = arma::join_cols(upper_b_,upper_t_);
    fhdl.os.ndeps_ = arma::ones<arma::vec>(P_+Q_) * tol;
    arma::mat hessian(P_+Q_,P_+Q_,fill::zeros);
    fhdl.Hessian(start_.subvec(0,P_+Q_-1),hessian);
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

class SparseDMatrix {
public:
  SparseDMatrix(Rcpp::List D_data,
                const arma::vec &gamma,
                arma::uvec Ap,
                arma::uvec Ai):
  gamma_(gamma){
    B_ = as<arma::uword>(D_data["B"]);
    N_dim_ = as<arma::uvec>(D_data["N_dim"]);
    N_func_ = as<arma::uvec>(D_data["N_func"]);
    func_def_ = as<arma::umat>(D_data["func_def"]);
    N_var_func_ = as<arma::umat>(D_data["N_var_func"]);
    eff_range_ = as<arma::mat>(D_data["eff_range"]);
    col_id_ = as<arma::ucube>(D_data["col_id"]);
    N_par_ = as<arma::umat>(D_data["N_par"]);
    cov_data_ = as<arma::cube>(D_data["cov_data"]);
    Ap_ = arma::conv_to<std::vector<int>>::from(Ap);
    Ai_ = arma::conv_to<std::vector<int>>::from(Ai);
    nx_ = Ai.size();
    n_ = arma::sum(N_dim_);
    Ax_ = std::vector<double>(nx_);
    mat_ = new sparse(Ap_);
    mat_->Ai = Ai_;
    chol_ = new SparseChol(mat_);
    update_parameters(gamma_);
  }
  
  arma::uword B_;
  arma::uvec N_dim_;
  arma::uvec N_func_;
  arma::umat func_def_;
  arma::umat N_var_func_;
  arma::mat eff_range_;
  arma::ucube col_id_;
  arma::umat N_par_;
  arma::cube cov_data_;
  arma::vec gamma_;
  std::vector<int> Ap_;
  std::vector<int> Ai_;
  int nx_;
  arma::uword n_;
  std::vector<double> Ax_;
  sparse* mat_;
  SparseChol* chol_;
  
  void update_parameters(const arma::vec &gamma){
    gamma_ = gamma; // assign new parameter values
    int llim = 0;
    int nj = 0;//(int)N_dim_(0);
    int ulim = Ap_[nj+(int)N_dim_(0)];
    int j = 0;
    
    for(arma::uword b=0; b<B_; b++){
      DSubMatrix *dblock;
      arma::uvec N_par_col0 = N_par_.col(0);
      arma::uword glim = (b == B_-1 || max(N_par_.row(b)) >= max(N_par_col0)) ?  gamma_.size() : min(N_par_col0(arma::find(N_par_col0 > max(N_par_.row(b)))));
      dblock = new DSubMatrix(N_dim_(b),
                              N_func_(b),
                              func_def_.row(b).t(),
                              N_var_func_.row(b).t(),
                              col_id_.slice(b),
                              N_par_.row(b).t() - min(N_par_.row(b)),
                              cov_data_.slice(b),
                              eff_range_.row(b).t(),
                              gamma_.subvec(min(N_par_.row(b)),glim-1));

      for(int i = llim; i<ulim; i++){
        if(i == Ap_[j+1])j++;
        
        Ax_[i] = dblock->get_val(Ai_[i]-nj,j-nj);
      }
      llim = ulim;
      if(b<(B_-1)){
        nj += (int)N_dim_(b);
        ulim = Ap_[nj+(int)N_dim_(b+1)];
      } 
      if(b == (B_-1)){
        ulim = nx_;
      }
      delete dblock;
    }
    //Rcpp::Rcout << "\n\n\nD:\n";
    // for(auto t: Ax_)
    //   Rcpp::Rcout << t << " ";
    mat_->Ax = Ax_;
    chol_->ldl_numeric();
    // Rcpp::Rcout << "\n\n\nChol:\n";
    // for(auto t: chol_->L->Ax)
    //   Rcpp::Rcout << t << " ";
  }
  
  double loglik(const arma::vec &u){
    double logl;
    double logdetD = 0;
    for (auto& k : chol_->D)
      logdetD += log(k);
    std::vector<double> v = arma::conv_to<std::vector<double>>::from(u);//std::vector<double> v(n_);
    chol_->ldl_lsolve(&v[0]);
    chol_->ldl_d2solve(&v[0]);
    double quadform = inner_sum(&v[0],&v[0],n_);
    logl = (-0.5*n_ * log(2*arma::datum::pi) - 0.5*logdetD - 0.5*quadform);
    return logl;
  }
  
  
  
  ~SparseDMatrix(){
    delete mat_;
    delete chol_;
  }
};

#endif