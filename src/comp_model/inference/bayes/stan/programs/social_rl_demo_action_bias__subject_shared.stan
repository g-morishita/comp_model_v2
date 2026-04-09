/*
 * Model: Social RL — Demonstrator Action Bias
 * Hierarchy: Subject-shared
 * Parameters: demo_bias
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

  int demo_bias_prior_family;
  real demo_bias_prior_p1;
  real demo_bias_prior_p2;
  real demo_bias_prior_p3;
}
parameters {
  real demo_bias;
}
model {
  int last_demo_action = 0;

  target += prior_lpdf(demo_bias | demo_bias_prior_family, demo_bias_prior_p1, demo_bias_prior_p2, demo_bias_prior_p3);

  for (e in 1:E) {
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      last_demo_action = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = rep_vector(0.0, A);
      if (last_demo_action > 0) u[last_demo_action] += demo_bias;
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
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
    int d = 0;

    for (e in 1:E) {
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        last_demo_action = 0;
      }
      if (step_choice[e] > 0) {
        vector[A] u = rep_vector(0.0, A);
        d += 1;
        if (last_demo_action > 0) u[last_demo_action] += demo_bias;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      if (step_social_action[e] > 0) {
        last_demo_action = step_social_action[e];
      }
    }
  }
}
