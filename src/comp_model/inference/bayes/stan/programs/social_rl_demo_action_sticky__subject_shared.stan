/*
 * Model: Sticky Social RL — Demonstrator Action Only
 * Hierarchy: Subject-shared
 * Parameters: alpha_other_action, beta, stickiness
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

  int alpha_other_action_prior_family;
  real alpha_other_action_prior_p1;
  real alpha_other_action_prior_p2;
  real alpha_other_action_prior_p3;
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
  real alpha_other_action_z;
  real beta_z;
  real stickiness_z;
}
transformed parameters {
  real<lower=0,upper=1> alpha_other_action = inv_logit(alpha_other_action_z);
  real<lower=0> beta = log1p_exp(beta_z);
  real stickiness = stickiness_z;
}
model {
  vector[A] T = rep_vector(1.0 / A, A);
  int last_self_choice = 0;

  target += prior_lpdf(alpha_other_action_z | alpha_other_action_prior_family, alpha_other_action_prior_p1, alpha_other_action_prior_p2, alpha_other_action_prior_p3);
  target += prior_lpdf(beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(stickiness_z | stickiness_prior_family, stickiness_prior_p1, stickiness_prior_p2, stickiness_prior_p3);

  for (e in 1:E) {
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      T = rep_vector(1.0 / A, A);
      last_self_choice = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = beta * T;
      if (last_self_choice > 0) u[last_self_choice] += stickiness;
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
      last_self_choice = step_choice[e];
    }
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      T = (1 - alpha_other_action) * T;
      T[sa] = T[sa] + alpha_other_action;
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    vector[A] T = rep_vector(1.0 / A, A);
    int last_self_choice = 0;
    int d = 0;

    for (e in 1:E) {
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        T = rep_vector(1.0 / A, A);
        last_self_choice = 0;
      }
      if (step_choice[e] > 0) {
        d += 1;
        vector[A] u = beta * T;
        if (last_self_choice > 0) u[last_self_choice] += stickiness;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
        last_self_choice = step_choice[e];
      }
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        T = (1 - alpha_other_action) * T;
        T[sa] = T[sa] + alpha_other_action;
      }
    }
  }
}
