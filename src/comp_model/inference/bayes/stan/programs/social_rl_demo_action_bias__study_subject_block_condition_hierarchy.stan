/*
 * Model: Social RL — Demonstrator Action Bias
 * Hierarchy: Study-subject-block-condition
 * Parameters: demo_bias[N][C]
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

  array[E] int<lower=0,upper=A> step_social_action;
  vector[E] step_social_reward;

  int<lower=2> C;
  int<lower=1,upper=C> baseline_cond;
  array[E] int<lower=1,upper=C> step_condition;

  int demo_bias_prior_family;
  real demo_bias_prior_p1;
  real demo_bias_prior_p2;
  real demo_bias_prior_p3;
  int demo_bias_delta_prior_family;
  real demo_bias_delta_prior_p1;
  real demo_bias_delta_prior_p2;
  real demo_bias_delta_prior_p3;
  int sd_demo_bias_prior_family;
  real sd_demo_bias_prior_p1;
  real sd_demo_bias_prior_p2;
  real sd_demo_bias_prior_p3;
  int sd_demo_bias_delta_prior_family;
  real sd_demo_bias_delta_prior_p1;
  real sd_demo_bias_delta_prior_p2;
  real sd_demo_bias_delta_prior_p3;
}
parameters {
  real mu_demo_bias_shared_z;
  real<lower=0> sd_demo_bias_shared_z;

  vector[C - 1] mu_demo_bias_delta_z;
  vector<lower=0>[C - 1] sd_demo_bias_delta_z;

  vector[N] raw_demo_bias_shared_z;
  array[C - 1] vector[N] raw_demo_bias_delta_z;
}
transformed parameters {
  vector[N] demo_bias_shared_z =
      mu_demo_bias_shared_z + sd_demo_bias_shared_z * raw_demo_bias_shared_z;

  array[C - 1] vector[N] demo_bias_delta_z;
  for (d in 1:(C - 1)) {
    demo_bias_delta_z[d] =
        mu_demo_bias_delta_z[d] + sd_demo_bias_delta_z[d] * raw_demo_bias_delta_z[d];
  }

  array[N] vector[C] demo_bias;
  for (n in 1:N) {
    int d = 0;
    for (c in 1:C) {
      real dbz = demo_bias_shared_z[n];
      if (c != baseline_cond) {
        d += 1;
        dbz += demo_bias_delta_z[d][n];
      }
      demo_bias[n][c] = dbz;
    }
  }
}
model {
  array[N] int last_demo_action;

  target += prior_lpdf(mu_demo_bias_shared_z | demo_bias_prior_family, demo_bias_prior_p1, demo_bias_prior_p2, demo_bias_prior_p3);
  target += prior_lpdf(sd_demo_bias_shared_z | sd_demo_bias_prior_family, sd_demo_bias_prior_p1, sd_demo_bias_prior_p2, sd_demo_bias_prior_p3);
  raw_demo_bias_shared_z ~ normal(0, 1);

  for (d in 1:(C - 1))
    target += prior_lpdf(mu_demo_bias_delta_z[d] | demo_bias_delta_prior_family, demo_bias_delta_prior_p1, demo_bias_delta_prior_p2, demo_bias_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_demo_bias_delta_z[d] | sd_demo_bias_delta_prior_family, sd_demo_bias_delta_prior_p1, sd_demo_bias_delta_prior_p2, sd_demo_bias_delta_prior_p3);
  for (d in 1:(C - 1)) {
    raw_demo_bias_delta_z[d] ~ normal(0, 1);
  }

  for (n in 1:N) {
    last_demo_action[n] = 0;
  }

  for (e in 1:E) {
    int n = step_subject[e];
    int cc = step_condition[e];
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      last_demo_action[n] = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = rep_vector(0.0, A);
      if (last_demo_action[n] > 0) u[last_demo_action[n]] += demo_bias[n][cc];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    if (step_social_action[e] > 0) {
      last_demo_action[n] = step_social_action[e];
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  vector[C] demo_bias_pop;
  real demo_bias_shared_pop;
  {
    array[N] int last_demo_action;
    int d = 0;

    for (n in 1:N) {
      last_demo_action[n] = 0;
    }

    for (e in 1:E) {
      int n = step_subject[e];
      int cc = step_condition[e];
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        last_demo_action[n] = 0;
      }
      if (step_choice[e] > 0) {
        vector[A] u = rep_vector(0.0, A);
        d += 1;
        if (last_demo_action[n] > 0) u[last_demo_action[n]] += demo_bias[n][cc];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      if (step_social_action[e] > 0) {
        last_demo_action[n] = step_social_action[e];
      }
    }
  }

  {
    int d_idx = 0;
    for (c in 1:C) {
      real dbz = mu_demo_bias_shared_z;
      if (c != baseline_cond) {
        d_idx += 1;
        dbz += mu_demo_bias_delta_z[d_idx];
      }
      demo_bias_pop[c] = dbz;
    }
  }

  demo_bias_shared_pop = demo_bias_pop[baseline_cond];
}
