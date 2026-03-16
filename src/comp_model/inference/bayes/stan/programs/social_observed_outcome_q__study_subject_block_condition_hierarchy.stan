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

  int<lower=2> C;
  int<lower=1,upper=C> baseline_cond;
  array[E] int<lower=1,upper=C> step_condition;

  int alpha_self_prior_family;
  real alpha_self_prior_p1;
  real alpha_self_prior_p2;
  real alpha_self_prior_p3;
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
  // Population-level: shared
  real mu_alpha_self_shared_z;
  real<lower=0> sd_alpha_self_shared_z;
  real mu_alpha_other_shared_z;
  real<lower=0> sd_alpha_other_shared_z;
  real mu_beta_shared_z;
  real<lower=0> sd_beta_shared_z;

  // Population-level: deltas
  vector[C - 1] mu_alpha_self_delta_z;
  vector<lower=0>[C - 1] sd_alpha_self_delta_z;
  vector[C - 1] mu_alpha_other_delta_z;
  vector<lower=0>[C - 1] sd_alpha_other_delta_z;
  vector[C - 1] mu_beta_delta_z;
  vector<lower=0>[C - 1] sd_beta_delta_z;

  // Per-subject raw (non-centered)
  vector[N] raw_alpha_self_shared_z;
  vector[N] raw_alpha_other_shared_z;
  vector[N] raw_beta_shared_z;
  array[C - 1] vector[N] raw_alpha_self_delta_z;
  array[C - 1] vector[N] raw_alpha_other_delta_z;
  array[C - 1] vector[N] raw_beta_delta_z;
}
transformed parameters {
  // Per-subject unconstrained shared
  vector[N] alpha_self_shared_z = mu_alpha_self_shared_z + sd_alpha_self_shared_z * raw_alpha_self_shared_z;
  vector[N] alpha_other_shared_z = mu_alpha_other_shared_z + sd_alpha_other_shared_z * raw_alpha_other_shared_z;
  vector[N] beta_shared_z = mu_beta_shared_z + sd_beta_shared_z * raw_beta_shared_z;

  // Per-subject unconstrained deltas
  array[C - 1] vector[N] alpha_self_delta_z;
  array[C - 1] vector[N] alpha_other_delta_z;
  array[C - 1] vector[N] beta_delta_z;
  for (d in 1:(C - 1)) {
    alpha_self_delta_z[d] = mu_alpha_self_delta_z[d] + sd_alpha_self_delta_z[d] * raw_alpha_self_delta_z[d];
    alpha_other_delta_z[d] = mu_alpha_other_delta_z[d] + sd_alpha_other_delta_z[d] * raw_alpha_other_delta_z[d];
    beta_delta_z[d] = mu_beta_delta_z[d] + sd_beta_delta_z[d] * raw_beta_delta_z[d];
  }

  // Per-subject, per-condition constrained parameters
  array[N] vector<lower=0,upper=1>[C] alpha_self;
  array[N] vector<lower=0,upper=1>[C] alpha_other;
  array[N] vector<lower=0>[C] beta;
  for (n in 1:N) {
    int d = 0;
    for (c in 1:C) {
      real as_z = alpha_self_shared_z[n];
      real ao_z = alpha_other_shared_z[n];
      real bz = beta_shared_z[n];
      if (c != baseline_cond) {
        d += 1;
        as_z += alpha_self_delta_z[d][n];
        ao_z += alpha_other_delta_z[d][n];
        bz += beta_delta_z[d][n];
      }
      alpha_self[n][c] = inv_logit(as_z);
      alpha_other[n][c] = inv_logit(ao_z);
      beta[n][c] = log1p_exp(bz);
    }
  }
}
model {
  array[N] vector[A] Q;

  // Priors: shared
  target += prior_lpdf(mu_alpha_self_shared_z | alpha_self_prior_family, alpha_self_prior_p1, alpha_self_prior_p2, alpha_self_prior_p3);
  sd_alpha_self_shared_z ~ normal(0, 1);
  raw_alpha_self_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_alpha_other_shared_z | alpha_other_prior_family, alpha_other_prior_p1, alpha_other_prior_p2, alpha_other_prior_p3);
  sd_alpha_other_shared_z ~ normal(0, 1);
  raw_alpha_other_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  sd_beta_shared_z ~ normal(0, 1);
  raw_beta_shared_z ~ normal(0, 1);

  // Priors: deltas
  mu_alpha_self_delta_z ~ normal(0, 1);
  sd_alpha_self_delta_z ~ normal(0, 1);
  mu_alpha_other_delta_z ~ normal(0, 1);
  sd_alpha_other_delta_z ~ normal(0, 1);
  mu_beta_delta_z ~ normal(0, 1);
  sd_beta_delta_z ~ normal(0, 1);
  for (d in 1:(C - 1)) {
    raw_alpha_self_delta_z[d] ~ normal(0, 1);
    raw_alpha_other_delta_z[d] ~ normal(0, 1);
    raw_beta_delta_z[d] ~ normal(0, 1);
  }

  for (n in 1:N) Q[n] = rep_vector(q_init, A);

  for (e in 1:E) {
    int n = step_subject[e];
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
    }
    if (step_choice[e] > 0) {
      int cc = step_condition[e];
      vector[A] u = beta[n][cc] * Q[n];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      int cc = step_condition[e];
      Q[n][a] = Q[n][a] + alpha_self[n][cc] * (step_reward[e] - Q[n][a]);
    }
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      int cc = step_condition[e];
      Q[n][sa] = Q[n][sa] + alpha_other[n][cc] * (step_social_reward[e] - Q[n][sa]);
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
        int cc = step_condition[e];
        vector[A] u = beta[n][cc] * Q[n];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        int cc = step_condition[e];
        Q[n][a] = Q[n][a] + alpha_self[n][cc] * (step_reward[e] - Q[n][a]);
      }
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        int cc = step_condition[e];
        Q[n][sa] = Q[n][sa] + alpha_other[n][cc] * (step_social_reward[e] - Q[n][sa]);
      }
    }
  }

  // Population-level constrained parameters (baseline condition)
  real alpha_self_shared_pop = inv_logit(mu_alpha_self_shared_z);
  real alpha_other_shared_pop = inv_logit(mu_alpha_other_shared_z);
  real beta_shared_pop = log1p_exp(mu_beta_shared_z);
}
