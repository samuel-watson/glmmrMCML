# glmmrMCML
R package for Markov Chain Monte Carlo Maximum Likelihood (MCML) estimation of generalised linear mixed models (GLMM), built on the [glmmrBase](https://github.com/samuel-watson/glmmrBase) package.

## Model specification
For model specification see readme of [glmmrBase](https://github.com/samuel-watson/glmmrBase). The `glmmrMCML` package creates the `ModelMCML` class that inherits from the `Model` class and adds the member function `MCML`

## Model estimation
The member function `MCML()` fits the model. There are multiple options to control the algorithm and set its parameters, which are detailed in the function manual. The main options for the user to choose are the algorithm (Monte Carlo Expectation Maximisation (MCEM) or Monte Carlo Newton Raphson (MCNR); default is MCEM), the standard error method (Hessian or approximate; default is Hessian), the number of MCMC samples $m$ (default is $m=100$), whether to use sparse matrix methods (default is false), and the tolerance of the algorithm $\tau$ to determine termination (default is $\tau = 0.01$). We provide several examples below.

### Cluster randomised trial
We simulate data from a parallel cluster randomised trial with ten clusters, five time periods, and ten individuals per cluster period cross-sectionally sampled. Clusters are assigned in a 1:1 ratio to treatment and control. The model is, for individual $i$ in cluster $j$ at time $t$:

$$
    y_{ijt}  \sim \text{Binomial}(1,p_{ijt}) 
$$

$$
    p_{ijt} = \Lambda(\beta_0 d_{j} + \beta_1I(t=1) + ... + \beta_6 I(t=5) + \gamma_{jt})
$$

where $d_j$ is an indicator for treatment status, $I(.)$ is the indicator function, and $\gamma_{jt}$ is a random effect. For the data generating process we use $\beta_0 = 0.5$ and $\beta_1,...,\beta_6 \sim N(0,1)$. We specify the exchangeable correlation structure $\gamma_{jt} = \gamma_{1,j} + \gamma_{2,jt}$ where $\gamma_{1,j} \sim N(0,0.25^2)$ and  $\gamma_{2,jt} \sim N(0,0.10^2)$. See REF for a discussion of such models in the context of cluster randomised trials. We generate data using the member function `sim_data()` 

```
R> df <- nelder(~ (cl(10)*t(5))>i(10))
R> df$int <- 0
R> df[df$cl>5,'int'] <- 1
R> cov1 <- Covariance$new(
R>    formula = ~(1|gr(cl))+(1|gr(cl,t)),
R>    data=df,
R>    parameters = c(0.25,0.1))
R> mf <- MeanFunction$new(
R>    formula = ~ int + factor(t)-1,
R>    data=df,
R>    parameters = c(0.5,rnorm(5)),
R>    family = binomial())
R> model1 <- ModelMCML$new(
R>    covariance = cov1,
R>    mean.function = mf)
R> y <- model1$sim_data()
```

we can then fit the model specified above with the default arguments as:
```
R> fit1 <- model1$MCML(y,tol=5e-3,m=250)
R> fit1
Markov chain Monte Carlo Maximum Likelihood Estimation
Algorithm:  Markov Chain Expectation Maximisation 
Fixed effects formula : ~ int + factor(t) - 1
Covariance function formula:  ~ (1 | gr(cl)) + (1 | gr(cl, t))
Family:  binomial , Link function: logit 

Number of Monte Carlo simulations per iteration:  250  with tolerance  0.005 
P-value and confidence interval method:  hessian 

              Estimate Std. Err. z value p value 2.5% CI 97.5% CI
int               0.59      0.23    2.57    0.01    0.14     1.04
t1               -1.51      0.27   -5.50    0.00   -2.04    -0.97
t2               -1.18      0.26   -4.55    0.00   -1.69    -0.67
t3               -1.23      0.26   -4.70    0.00   -1.74    -0.72
t4                1.61      0.30    5.29    0.00    1.01     2.20
t5               -2.57      0.36   -7.21    0.00   -3.26    -1.87
1 | gr(cl)        0.62      0.14    4.47    0.00    0.35     0.89
1 | gr(cl, t)     0.10      0.01   10.00    0.00    0.08     0.12

cAIC:  422.04
Approximate R-squared: Conditional:  0.24  Marginal:  0.21
```

If alternatively, we wanted to fit a model with an autoregressive temporal structure within clusters then we could re-specify the covariance for a new model object:
```
R> cov2 <- Covariance$new(
R>    formula = ~(1|gr(cl)*ar1(t)),
R>    data=df,
R>    parameters = c(0.25,0.8))
R> model2 <- model1$clone(deep=TRUE)
R> model2$covariance <- cov2
R> fit2 <- model2$MCML(y,tol=5e-3,m=250)
R> fit2
Markov chain Monte Carlo Maximum Likelihood Estimation
Algorithm:  Markov Chain Expectation Maximisation 
Fixed effects formula : ~ int + factor(t) - 1
Covariance function formula:  ~ (1 | gr(cl) * ar1(t))
Family:  binomial , Link function: logit 

Number of Monte Carlo simulations per iteration:  250  with tolerance  0.005 
P-value and confidence interval method:  hessian 

                      Estimate Std. Err. z value p value 2.5% CI 97.5% CI
int                       0.91      0.23    3.90       0    0.45     1.37
t1                       -1.82      0.28   -6.38       0   -2.37    -1.26
t2                       -1.32      0.26   -4.99       0   -1.83    -0.80
t3                       -1.38      0.26   -5.29       0   -1.90    -0.87
t4                        1.44      0.31    4.69       0    0.84     2.04
t5                       -2.69      0.36   -7.54       0   -3.39    -1.99
1 | gr(cl) * ar1(t).1     0.79      0.11    7.22       0    0.58     1.01
1 | gr(cl) * ar1(t).2     0.62      0.12    5.31       0    0.39     0.85

cAIC:  605.1
Approximate R-squared: Conditional:  0.24  Marginal:  0.21
```

### Geospatial statistical model
We next simulate data from a linear geospatial statistical model defined on the unit square. Each observation $i$ has a location $l_i \in [0,1]\times[0,1]$ and we specify the model:

$$
    y(l_i) \sim N(\beta_0 + S(l_i), 1)
$$

where $S(l_i)$ are assumed to be multivariate normal realisations of a Gaussian process with zero mean and covariance function:

$$
    Cov(S(l),S(l')) = \tau^2 h(\vert \vert l - l' \vert \vert; \theta)
$$

We set $\beta_0 = 1$. We first consider the exponential correlation function with parameters 0.25 and 0.1. We simulate locations of $250$ observations uniformly 
on the unit square:
```
R> df <- data.frame(x=runif(250),y=runif(250))
R> model3 <- ModelMCML$new(
R>   covariance = list(
R>     formula = ~(1|fexp(x,y)),
R>     parameters = c(0.25,0.1),
R>     data=df
R>   ),mean.function = list(
R>     formula = ~1,
R>     data=df,
R>     parameters = rnorm(1),
R>     family=gaussian()
R>   ),var_par=1
R> )
R> y <- model3$sim_data()
R> fit3 <- model3$MCML(y,m=250)
R> fit3
Markov chain Monte Carlo Maximum Likelihood Estimation
Algorithm:  Markov Chain Expectation Maximisation 
Fixed effects formula : ~ 1
Covariance function formula:  ~ (1 | fexp(x, y))
Family:  gaussian , Link function: identity 

Number of Monte Carlo simulations per iteration:  250  with tolerance  0.01 
P-value and confidence interval method:  hessian 

                 Estimate Std. Err. z value p value 2.5% CI 97.5% CI
(Intercept)          0.86      0.06   13.91       0    0.74     0.99
1 | fexp(x, y).1     0.26      0.02   10.57       0    0.21     0.31
1 | fexp(x, y).2     0.02      0.00    4.81       0    0.01     0.03
sigma                0.98        NA      NA      NA      NA       NA

cAIC:  1060.84
Approximate R-squared: Conditional:  0.06  Marginal:  0
```

We now instead turn to a compactly specified, or tapered, covariance function for $h(.)$, which is the product of the powered Whittle-Matern and Wendland functions with parameters 0.25 and 0.1. We set an effective range of 0.35 and use the sparse matrix approach. Note, this exercise is purely illustrative and it is not intended to demonstrate an appropriate way to analyse data of this type!
```
R> df <- data.frame(x=runif(250),y=runif(250))
R> cov3 <- Covariance$new(
R>     formula = ~(1|prodwm(x,y)),
R>     eff_range = 0.35,
R>     parameters = c(0.25,0.1),
R>     data=df)
R> des2 <- des$clone()
R> des2$covariance <- cov3
R> fit4 <- des2$MCML(y,m=250,sparse=TRUE)
R> fit4
Markov chain Monte Carlo Maximum Likelihood Estimation
Algorithm:  Markov Chain Expectation Maximisation 
Fixed effects formula : ~ 1
Covariance function formula:  ~ (1 | prodwm(x, y))
Family:  gaussian , Link function: identity 

Number of Monte Carlo simulations per iteration:  250  with tolerance  0.01 
P-value and confidence interval method:  hessian 

                   Estimate Std. Err. z value p value 2.5% CI 97.5% CI
(Intercept)            0.86      0.06   13.87       0    0.74     0.98
1 | prodwm(x, y).1     0.25      0.03   10.15       0    0.21     0.30
1 | prodwm(x, y).2     0.06      0.02    3.07       0    0.02     0.10
sigma                  0.98        NA      NA      NA      NA       NA

cAIC:  1055.59
Approximate R-squared: Conditional:  0.06  Marginal:  0
```
