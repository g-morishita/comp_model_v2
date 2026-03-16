functions {
#include "prior_lpdf.stanfunctions"
}
data {
  int<lower=1> N;
  int<lower=1> A;
  int<lower=1> E;
  int<lower=0> D;
  array[E] int<lower=1,upper=N> step_subject;
  array[E] int<lower=0,upper=A> step_choice;
  array[E] int<lower=0,upper=A> step_update_action;
  vector[E] step_reward;
  array[E] vector<lower=0,upper=1>[A] step_avail_mask;
  array[E] int step_block;
  int<lower=0,upper=1> reset_on_block;
  real q_init;

  int alpha_prior_family;
  real alpha_prior_p1;
  real alpha_prior_p2;
  real alpha_prior_p3;
  int beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
}
parameters {
  real mu_alpha_z;
  real<lower=0> sd_alpha_z;
  vector[N] raw_alpha_z;

  real mu_beta_z;
  real<lower=0> sd_beta_z;
  vector[N] raw_beta_z;
}
transformed parameters {
  vector[N] alpha_z = mu_alpha_z + sd_alpha_z * raw_alpha_z;
  vector[N] beta_z = mu_beta_z + sd_beta_z * raw_beta_z;
  vector<lower=0,upper=1>[N] alpha = inv_logit(alpha_z);
  vector<lower=0>[N] beta = log1p_exp(beta_z);
}
model {
  array[N] vector[A] Q;

  target += prior_lpdf(mu_alpha_z | alpha_prior_family, alpha_prior_p1, alpha_prior_p2, alpha_prior_p3);
  sd_alpha_z ~ normal(0, 1);
  raw_alpha_z ~ normal(0, 1);

  target += prior_lpdf(mu_beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  sd_beta_z ~ normal(0, 1);
  raw_beta_z ~ normal(0, 1);

  for (n in 1:N) Q[n] = rep_vector(q_init, A);

  for (e in 1:E) {
    int n = step_subject[e];
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
    }
    if (step_choice[e] > 0) {
      vector[A] u = beta[n] * Q[n];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      Q[n][a] = Q[n][a] + alpha[n] * (step_reward[e] - Q[n][a]);
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    array[N] vector[A] Q;
    for (n in 1:N) Q[n] = rep_vector(q_init, A);
    int d = 0;

    for (e in 1:E) {
      int n = step_subject[e];
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        Q[n] = rep_vector(q_init, A);
      }
      if (step_choice[e] > 0) {
        d += 1;
        vector[A] u = beta[n] * Q[n];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        Q[n][a] = Q[n][a] + alpha[n] * (step_reward[e] - Q[n][a]);
      }
    }
  }
  real alpha_pop = inv_logit(mu_alpha_z);
  real beta_pop = log1p_exp(mu_beta_z);
}
