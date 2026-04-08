/*
 * Model: Social RL — Demonstrator Action Bias with Stickiness
 * Hierarchy: Subject-block-condition
 * Parameters: demo_bias[C], stickiness[C]
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

  array[E] int<lower=0,upper=A> step_social_action;
  vector[E] step_social_reward;

  int<lower=2> C;
  int<lower=1,upper=C> baseline_cond;
  array[E] int<lower=1,upper=C> step_condition;

  int demo_bias_prior_family;
  real demo_bias_prior_p1;
  real demo_bias_prior_p2;
  real demo_bias_prior_p3;
  int stickiness_prior_family;
  real stickiness_prior_p1;
  real stickiness_prior_p2;
  real stickiness_prior_p3;
}
parameters {
  real demo_bias_shared_z;
  real stickiness_shared_z;
  vector[C - 1] demo_bias_delta_z;
  vector[C - 1] stickiness_delta_z;
}
transformed parameters {
  vector[C] demo_bias;
  vector[C] stickiness;
  {
    int d = 0;
    for (c in 1:C) {
      real dbz = demo_bias_shared_z;
      real sz = stickiness_shared_z;
      if (c != baseline_cond) {
        d += 1;
        dbz += demo_bias_delta_z[d];
        sz += stickiness_delta_z[d];
      }
      demo_bias[c] = dbz;
      stickiness[c] = sz;
    }
  }
}
model {
  int last_demo_action = 0;
  int last_self_choice = 0;

  target += prior_lpdf(demo_bias_shared_z | demo_bias_prior_family, demo_bias_prior_p1, demo_bias_prior_p2, demo_bias_prior_p3);
  target += prior_lpdf(stickiness_shared_z | stickiness_prior_family, stickiness_prior_p1, stickiness_prior_p2, stickiness_prior_p3);
  demo_bias_delta_z ~ normal(0, 1);
  stickiness_delta_z ~ normal(0, 1);

  for (e in 1:E) {
    int cc = step_condition[e];
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      last_demo_action = 0;
      last_self_choice = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = rep_vector(0.0, A);
      if (last_demo_action > 0) u[last_demo_action] += demo_bias[cc];
      if (last_self_choice > 0) u[last_self_choice] += stickiness[cc];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
      last_self_choice = step_choice[e];
    }
    if (step_social_action[e] > 0) {
      last_demo_action = step_social_action[e];
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    int last_demo_action = 0;
    int last_self_choice = 0;
    int d = 0;

    for (e in 1:E) {
      int cc = step_condition[e];
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        last_demo_action = 0;
        last_self_choice = 0;
      }
      if (step_choice[e] > 0) {
        vector[A] u = rep_vector(0.0, A);
        d += 1;
        if (last_demo_action > 0) u[last_demo_action] += demo_bias[cc];
        if (last_self_choice > 0) u[last_self_choice] += stickiness[cc];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
        last_self_choice = step_choice[e];
      }
      if (step_social_action[e] > 0) {
        last_demo_action = step_social_action[e];
      }
    }
  }
}
