/*
 * Model: Social RL — Demonstrator Action Bias with Stickiness
 * Hierarchy: Study-subject
 * Parameters: demo_bias[N], stickiness[N]
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

  int demo_bias_prior_family;
  real demo_bias_prior_p1;
  real demo_bias_prior_p2;
  real demo_bias_prior_p3;
  int sd_demo_bias_prior_family;
  real sd_demo_bias_prior_p1;
  real sd_demo_bias_prior_p2;
  real sd_demo_bias_prior_p3;
  int stickiness_prior_family;
  real stickiness_prior_p1;
  real stickiness_prior_p2;
  real stickiness_prior_p3;
  int sd_stickiness_prior_family;
  real sd_stickiness_prior_p1;
  real sd_stickiness_prior_p2;
  real sd_stickiness_prior_p3;
}
parameters {
  real mu_demo_bias_z;
  real<lower=0> sd_demo_bias_z;
  vector[N] raw_demo_bias_z;

  real mu_stickiness_z;
  real<lower=0> sd_stickiness_z;
  vector[N] raw_stickiness_z;
}
transformed parameters {
  vector[N] demo_bias = mu_demo_bias_z + sd_demo_bias_z * raw_demo_bias_z;
  vector[N] stickiness = mu_stickiness_z + sd_stickiness_z * raw_stickiness_z;
}
model {
  array[N] int last_demo_action;
  array[N] int last_self_choice;

  target += prior_lpdf(mu_demo_bias_z | demo_bias_prior_family, demo_bias_prior_p1, demo_bias_prior_p2, demo_bias_prior_p3);
  target += prior_lpdf(sd_demo_bias_z | sd_demo_bias_prior_family, sd_demo_bias_prior_p1, sd_demo_bias_prior_p2, sd_demo_bias_prior_p3);
  raw_demo_bias_z ~ normal(0, 1);

  target += prior_lpdf(mu_stickiness_z | stickiness_prior_family, stickiness_prior_p1, stickiness_prior_p2, stickiness_prior_p3);
  target += prior_lpdf(sd_stickiness_z | sd_stickiness_prior_family, sd_stickiness_prior_p1, sd_stickiness_prior_p2, sd_stickiness_prior_p3);
  raw_stickiness_z ~ normal(0, 1);

  for (n in 1:N) {
    last_demo_action[n] = 0;
    last_self_choice[n] = 0;
  }

  for (e in 1:E) {
    int n = step_subject[e];
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      last_demo_action[n] = 0;
      last_self_choice[n] = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = rep_vector(0.0, A);
      if (last_demo_action[n] > 0) u[last_demo_action[n]] += demo_bias[n];
      if (last_self_choice[n] > 0) u[last_self_choice[n]] += stickiness[n];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
      last_self_choice[n] = step_choice[e];
    }
    if (step_update_action[e] > 0) {
      last_self_choice[n] = step_update_action[e];
    }
    if (step_social_action[e] > 0) {
      last_demo_action[n] = step_social_action[e];
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    array[N] int last_demo_action;
    array[N] int last_self_choice;
    int d = 0;

    for (n in 1:N) {
      last_demo_action[n] = 0;
      last_self_choice[n] = 0;
    }

    for (e in 1:E) {
      int n = step_subject[e];
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        last_demo_action[n] = 0;
        last_self_choice[n] = 0;
      }
      if (step_choice[e] > 0) {
        vector[A] u = rep_vector(0.0, A);
        d += 1;
        if (last_demo_action[n] > 0) u[last_demo_action[n]] += demo_bias[n];
        if (last_self_choice[n] > 0) u[last_self_choice[n]] += stickiness[n];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
        last_self_choice[n] = step_choice[e];
      }
      if (step_update_action[e] > 0) {
        last_self_choice[n] = step_update_action[e];
      }
      if (step_social_action[e] > 0) {
        last_demo_action[n] = step_social_action[e];
      }
    }
  }
  real demo_bias_pop = mu_demo_bias_z;
  real stickiness_pop = mu_stickiness_z;
}
