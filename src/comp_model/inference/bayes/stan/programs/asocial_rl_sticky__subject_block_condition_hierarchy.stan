/*
 * Model: Asocial RL with Stickiness
 * Hierarchy: Subject-block-condition
 * Parameters: alpha[C], beta[C], stickiness[C]
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

  int<lower=2> C;
  int<lower=1,upper=C> baseline_cond;
  array[E] int<lower=1,upper=C> step_condition;

  int alpha_prior_family;
  real alpha_prior_p1;
  real alpha_prior_p2;
  real alpha_prior_p3;
  int beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
  int stickiness_prior_family;
  real stickiness_prior_p1;
  real stickiness_prior_p2;
  real stickiness_prior_p3;
}
parameters {
  real alpha_shared_z;
  real beta_shared_z;
  real stickiness_shared_z;
  vector[C - 1] alpha_delta_z;
  vector[C - 1] beta_delta_z;
  vector[C - 1] stickiness_delta_z;
}
transformed parameters {
  vector<lower=0,upper=1>[C] alpha;
  vector<lower=0>[C] beta;
  vector[C] stickiness;
  {
    int d = 0;
    for (c in 1:C) {
      real az = alpha_shared_z;
      real bz = beta_shared_z;
      real sz = stickiness_shared_z;
      if (c != baseline_cond) {
        d += 1;
        az += alpha_delta_z[d];
        bz += beta_delta_z[d];
        sz += stickiness_delta_z[d];
      }
      alpha[c] = inv_logit(az);
      beta[c] = log1p_exp(bz);
      stickiness[c] = sz;
    }
  }
}
model {
  vector[A] Q = rep_vector(q_init, A);
  int last_self_choice = 0;

  target += prior_lpdf(alpha_shared_z | alpha_prior_family, alpha_prior_p1, alpha_prior_p2, alpha_prior_p3);
  target += prior_lpdf(beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(stickiness_shared_z | stickiness_prior_family, stickiness_prior_p1, stickiness_prior_p2, stickiness_prior_p3);
  alpha_delta_z ~ normal(0, 1);
  beta_delta_z ~ normal(0, 1);
  stickiness_delta_z ~ normal(0, 1);

  for (e in 1:E) {
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
      last_self_choice = 0;
    }
    if (step_choice[e] > 0) {
      int cc = step_condition[e];
      vector[A] u = beta[cc] * Q;
      if (last_self_choice > 0) u[last_self_choice] += stickiness[cc];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
      // Preserve the previous choice even when the trial omits feedback.
      last_self_choice = step_choice[e];
    }
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      int cc = step_condition[e];
      Q[a] = Q[a] + alpha[cc] * (step_reward[e] - Q[a]);
      last_self_choice = a;
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    vector[A] Q = rep_vector(q_init, A);
    int last_self_choice = 0;
    int d = 0;

    for (e in 1:E) {
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        Q = rep_vector(q_init, A);
        last_self_choice = 0;
      }
      if (step_choice[e] > 0) {
        d += 1;
        int cc = step_condition[e];
        vector[A] u = beta[cc] * Q;
        if (last_self_choice > 0) u[last_self_choice] += stickiness[cc];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
        // Preserve the previous choice even when the trial omits feedback.
        last_self_choice = step_choice[e];
      }
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        int cc = step_condition[e];
        Q[a] = Q[a] + alpha[cc] * (step_reward[e] - Q[a]);
        last_self_choice = a;
      }
    }
  }
}
