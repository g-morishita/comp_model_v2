/*
 * Model: Social RL — Demonstrator Reward Only
 * Hierarchy: Study-subject
 * Parameters: alpha_other[N], beta[N]
 */
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

  array[E] int<lower=0,upper=A> step_social_action;
  vector[E] step_social_reward;

  int alpha_other_prior_family;
  real alpha_other_prior_p1;
  real alpha_other_prior_p2;
  real alpha_other_prior_p3;
  int sd_alpha_other_prior_family;
  real sd_alpha_other_prior_p1;
  real sd_alpha_other_prior_p2;
  real sd_alpha_other_prior_p3;
  int beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
  int sd_beta_prior_family;
  real sd_beta_prior_p1;
  real sd_beta_prior_p2;
  real sd_beta_prior_p3;
}
parameters {
  real mu_alpha_other_z;
  real<lower=0> sd_alpha_other_z;
  vector[N] raw_alpha_other_z;

  real mu_beta_z;
  real<lower=0> sd_beta_z;
  vector[N] raw_beta_z;
}
transformed parameters {
  vector[N] alpha_other_z = mu_alpha_other_z + sd_alpha_other_z * raw_alpha_other_z;
  vector[N] beta_z = mu_beta_z + sd_beta_z * raw_beta_z;
  vector<lower=0,upper=1>[N] alpha_other = inv_logit(alpha_other_z);
  vector<lower=0>[N] beta = log1p_exp(beta_z);
}
model {
  array[N] vector[A] Q;

  target += prior_lpdf(mu_alpha_other_z | alpha_other_prior_family, alpha_other_prior_p1, alpha_other_prior_p2, alpha_other_prior_p3);
  target += prior_lpdf(sd_alpha_other_z | sd_alpha_other_prior_family, sd_alpha_other_prior_p1, sd_alpha_other_prior_p2, sd_alpha_other_prior_p3);
  raw_alpha_other_z ~ normal(0, 1);

  target += prior_lpdf(mu_beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(sd_beta_z | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);
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
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      Q[n][sa] = Q[n][sa] + alpha_other[n] * (step_social_reward[e] - Q[n][sa]);
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    array[N] vector[A] Q;
    int d = 0;

    for (n in 1:N) Q[n] = rep_vector(q_init, A);

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
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        Q[n][sa] = Q[n][sa] + alpha_other[n] * (step_social_reward[e] - Q[n][sa]);
      }
    }
  }
  real alpha_other_pop = inv_logit(mu_alpha_other_z);
  real beta_pop = log1p_exp(mu_beta_z);
}
