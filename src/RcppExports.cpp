// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/glmmrMCML.h"
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// mcml_full
Rcpp::List mcml_full(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, std::string family, std::string link, Eigen::ArrayXd start, bool mcnr, int m, int thin, int maxiter, int warmup, double tol, bool verbose, int trace);
RcppExport SEXP _glmmrMCML_mcml_full(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP familySEXP, SEXP linkSEXP, SEXP startSEXP, SEXP mcnrSEXP, SEXP mSEXP, SEXP thinSEXP, SEXP maxiterSEXP, SEXP warmupSEXP, SEXP tolSEXP, SEXP verboseSEXP, SEXP traceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< Eigen::ArrayXd >::type start(startSEXP);
    Rcpp::traits::input_parameter< bool >::type mcnr(mcnrSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< int >::type warmup(warmupSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    rcpp_result_gen = Rcpp::wrap(mcml_full(cov, data, eff_range, Z, X, y, family, link, start, mcnr, m, thin, maxiter, warmup, tol, verbose, trace));
    return rcpp_result_gen;
END_RCPP
}
// mcmc_sample
Eigen::MatrixXd mcmc_sample(const Eigen::MatrixXd& Z, const Eigen::MatrixXd& L, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& beta, double var_par, std::string family, std::string link, int warmup, int nsamp, int thin);
RcppExport SEXP _glmmrMCML_mcmc_sample(SEXP ZSEXP, SEXP LSEXP, SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP var_parSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP warmupSEXP, SEXP nsampSEXP, SEXP thinSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type L(LSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type var_par(var_parSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< int >::type warmup(warmupSEXP);
    Rcpp::traits::input_parameter< int >::type nsamp(nsampSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    rcpp_result_gen = Rcpp::wrap(mcmc_sample(Z, L, X, y, beta, var_par, family, link, warmup, nsamp, thin));
    return rcpp_result_gen;
END_RCPP
}
// mcml_optim
Rcpp::List mcml_optim(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& u, std::string family, std::string link, Eigen::ArrayXd start, int trace, bool mcnr);
RcppExport SEXP _glmmrMCML_mcml_optim(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP startSEXP, SEXP traceSEXP, SEXP mcnrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< Eigen::ArrayXd >::type start(startSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< bool >::type mcnr(mcnrSEXP);
    rcpp_result_gen = Rcpp::wrap(mcml_optim(cov, data, eff_range, Z, X, y, u, family, link, start, trace, mcnr));
    return rcpp_result_gen;
END_RCPP
}
// mcml_doptim
void mcml_doptim(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& u, std::string family, std::string link, Eigen::ArrayXd start, int trace, bool mcnr);
RcppExport SEXP _glmmrMCML_mcml_doptim(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP startSEXP, SEXP traceSEXP, SEXP mcnrSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< Eigen::ArrayXd >::type start(startSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< bool >::type mcnr(mcnrSEXP);
    mcml_doptim(cov, data, eff_range, Z, X, y, u, family, link, start, trace, mcnr);
    return R_NilValue;
END_RCPP
}
// mcml_simlik
Rcpp::List mcml_simlik(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& u, std::string family, std::string link, Eigen::ArrayXd start, int trace);
RcppExport SEXP _glmmrMCML_mcml_simlik(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP startSEXP, SEXP traceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< Eigen::ArrayXd >::type start(startSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    rcpp_result_gen = Rcpp::wrap(mcml_simlik(cov, data, eff_range, Z, X, y, u, family, link, start, trace));
    return rcpp_result_gen;
END_RCPP
}
// mcml_optim_sparse
Rcpp::List mcml_optim_sparse(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::ArrayXi& Ap, const Eigen::ArrayXi& Ai, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& u, std::string family, std::string link, Eigen::ArrayXd start, int trace, bool mcnr);
RcppExport SEXP _glmmrMCML_mcml_optim_sparse(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ApSEXP, SEXP AiSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP startSEXP, SEXP traceSEXP, SEXP mcnrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXi& >::type Ap(ApSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXi& >::type Ai(AiSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< Eigen::ArrayXd >::type start(startSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< bool >::type mcnr(mcnrSEXP);
    rcpp_result_gen = Rcpp::wrap(mcml_optim_sparse(cov, data, eff_range, Ap, Ai, Z, X, y, u, family, link, start, trace, mcnr));
    return rcpp_result_gen;
END_RCPP
}
// mcml_simlik_sparse
Rcpp::List mcml_simlik_sparse(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::ArrayXi& Ap, const Eigen::ArrayXi& Ai, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& u, std::string family, std::string link, Eigen::ArrayXd start, int trace);
RcppExport SEXP _glmmrMCML_mcml_simlik_sparse(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ApSEXP, SEXP AiSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP startSEXP, SEXP traceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXi& >::type Ap(ApSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXi& >::type Ai(AiSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< Eigen::ArrayXd >::type start(startSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    rcpp_result_gen = Rcpp::wrap(mcml_simlik_sparse(cov, data, eff_range, Ap, Ai, Z, X, y, u, family, link, start, trace));
    return rcpp_result_gen;
END_RCPP
}
// mcml_hess
Eigen::MatrixXd mcml_hess(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& u, std::string family, std::string link, Eigen::ArrayXd start, double tol, int trace);
RcppExport SEXP _glmmrMCML_mcml_hess(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP startSEXP, SEXP tolSEXP, SEXP traceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< Eigen::ArrayXd >::type start(startSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    rcpp_result_gen = Rcpp::wrap(mcml_hess(cov, data, eff_range, Z, X, y, u, family, link, start, tol, trace));
    return rcpp_result_gen;
END_RCPP
}
// mcml_hess_sparse
Eigen::MatrixXd mcml_hess_sparse(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::ArrayXi& Ap, const Eigen::ArrayXi& Ai, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& u, std::string family, std::string link, Eigen::ArrayXd start, double tol, int trace);
RcppExport SEXP _glmmrMCML_mcml_hess_sparse(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ApSEXP, SEXP AiSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP startSEXP, SEXP tolSEXP, SEXP traceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXi& >::type Ap(ApSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXi& >::type Ai(AiSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< Eigen::ArrayXd >::type start(startSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    rcpp_result_gen = Rcpp::wrap(mcml_hess_sparse(cov, data, eff_range, Ap, Ai, Z, X, y, u, family, link, start, tol, trace));
    return rcpp_result_gen;
END_RCPP
}
// aic_mcml
double aic_mcml(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::MatrixXd& u, std::string family, std::string link, const Eigen::VectorXd& beta_par, const Eigen::VectorXd& cov_par);
RcppExport SEXP _glmmrMCML_aic_mcml(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP ZSEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP beta_parSEXP, SEXP cov_parSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type beta_par(beta_parSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type cov_par(cov_parSEXP);
    rcpp_result_gen = Rcpp::wrap(aic_mcml(cov, data, eff_range, Z, X, y, u, family, link, beta_par, cov_par));
    return rcpp_result_gen;
END_RCPP
}
// mvn_ll
double mvn_ll(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::ArrayXd& gamma, const Eigen::MatrixXd& u);
RcppExport SEXP _glmmrMCML_mvn_ll(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP gammaSEXP, SEXP uSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type u(uSEXP);
    rcpp_result_gen = Rcpp::wrap(mvn_ll(cov, data, eff_range, gamma, u));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_glmmrMCML_mcml_full", (DL_FUNC) &_glmmrMCML_mcml_full, 17},
    {"_glmmrMCML_mcmc_sample", (DL_FUNC) &_glmmrMCML_mcmc_sample, 11},
    {"_glmmrMCML_mcml_optim", (DL_FUNC) &_glmmrMCML_mcml_optim, 12},
    {"_glmmrMCML_mcml_doptim", (DL_FUNC) &_glmmrMCML_mcml_doptim, 12},
    {"_glmmrMCML_mcml_simlik", (DL_FUNC) &_glmmrMCML_mcml_simlik, 11},
    {"_glmmrMCML_mcml_optim_sparse", (DL_FUNC) &_glmmrMCML_mcml_optim_sparse, 14},
    {"_glmmrMCML_mcml_simlik_sparse", (DL_FUNC) &_glmmrMCML_mcml_simlik_sparse, 13},
    {"_glmmrMCML_mcml_hess", (DL_FUNC) &_glmmrMCML_mcml_hess, 12},
    {"_glmmrMCML_mcml_hess_sparse", (DL_FUNC) &_glmmrMCML_mcml_hess_sparse, 14},
    {"_glmmrMCML_aic_mcml", (DL_FUNC) &_glmmrMCML_aic_mcml, 11},
    {"_glmmrMCML_mvn_ll", (DL_FUNC) &_glmmrMCML_mvn_ll, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_glmmrMCML(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
