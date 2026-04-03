/*
 * Model: Social RL — Demonstrator Reward Only
 * Hierarchy: Subject-block-condition
 * Parameters: alpha_other[C], beta[C]
 */
functions {
#include "prior_lpdf.stanfunctions"
}
data {
  int<lower=1> A;
  int<lower=1> E;
  int<lower=0> D;
  array[E] int<lower=0,upper=A> step_choice;
  array[E] int<lower=0,upper=A> step_update_action;
  vector[E] step_reward;
  array[E] vector<lower=0,upper=1>[A] step_avail_mask;
  array[E] int step_block;
  int<lower=0,upper=1> reset_on_block;
  real q_init;

  array[E] int<lower=0,upper=A> step_social_action;
  vector[E] step_social_reward;

  int<lower=2> C;
  int<lower=1,upper=C> baseline_cond;
  array[E] int<lower=1,upper=C> step_condition;

  int alpha_other_prior_family;
  real alpha_other_prior_p1;
  real alpha_other_prior_p2;
  real alpha_other_prior_p3;
  int beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
}
parameters {
  real alpha_other_shared_z;
  real beta_shared_z;
  vector[C - 1] alpha_other_delta_z;
  vector[C - 1] beta_delta_z;
}
transformed parameters {
  vector<lower=0,upper=1>[C] alpha_other;
  vector<lower=0>[C] beta;
  {
    int d = 0;
    for (c in 1:C) {
      real ao_z = alpha_other_shared_z;
      real bz = beta_shared_z;
      if (c != baseline_cond) {
        d += 1;
        ao_z += alpha_other_delta_z[d];
        bz += beta_delta_z[d];
      }
      alpha_other[c] = inv_logit(ao_z);
      beta[c] = log1p_exp(bz);
    }
  }
}
model {
  vector[A] Q = rep_vector(q_init, A);

  target += prior_lpdf(alpha_other_shared_z | alpha_other_prior_family, alpha_other_prior_p1, alpha_other_prior_p2, alpha_other_prior_p3);
  target += prior_lpdf(beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  alpha_other_delta_z ~ normal(0, 1);
  beta_delta_z ~ normal(0, 1);

  for (e in 1:E) {
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
    }
    if (step_choice[e] > 0) {
      int cc = step_condition[e];
      vector[A] u = beta[cc] * Q;
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      int cc = step_condition[e];
      Q[sa] = Q[sa] + alpha_other[cc] * (step_social_reward[e] - Q[sa]);
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    vector[A] Q = rep_vector(q_init, A);
    int d = 0;

    for (e in 1:E) {
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        Q = rep_vector(q_init, A);
      }
      if (step_choice[e] > 0) {
        d += 1;
        int cc = step_condition[e];
        vector[A] u = beta[cc] * Q;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        int cc = step_condition[e];
        Q[sa] = Q[sa] + alpha_other[cc] * (step_social_reward[e] - Q[sa]);
      }
    }
  }
}
