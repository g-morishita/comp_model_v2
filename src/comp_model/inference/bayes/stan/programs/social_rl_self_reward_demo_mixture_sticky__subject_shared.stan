/*
 * Model: Sticky Social RL — Self Reward + Demonstrator Mixture
 * Hierarchy: Subject-shared
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

  int alpha_self_prior_family;
  real alpha_self_prior_p1;
  real alpha_self_prior_p2;
  real alpha_self_prior_p3;
  int alpha_other_outcome_prior_family;
  real alpha_other_outcome_prior_p1;
  real alpha_other_outcome_prior_p2;
  real alpha_other_outcome_prior_p3;
  int alpha_other_action_prior_family;
  real alpha_other_action_prior_p1;
  real alpha_other_action_prior_p2;
  real alpha_other_action_prior_p3;
  int w_imitation_prior_family;
  real w_imitation_prior_p1;
  real w_imitation_prior_p2;
  real w_imitation_prior_p3;
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
  real alpha_self_z;
  real alpha_other_outcome_z;
  real alpha_other_action_z;
  real w_imitation_z;
  real beta_z;
  real stickiness;
}
transformed parameters {
  real<lower=0,upper=1> alpha_self = inv_logit(alpha_self_z);
  real<lower=0,upper=1> alpha_other_outcome = inv_logit(alpha_other_outcome_z);
  real<lower=0,upper=1> alpha_other_action = inv_logit(alpha_other_action_z);
  real<lower=0,upper=1> w_imitation = inv_logit(w_imitation_z);
  real<lower=0> beta = log1p_exp(beta_z);
}
model {
  vector[A] Q = rep_vector(q_init, A);
  vector[A] T = rep_vector(1.0 / A, A);
  int last_self_choice = 0;

  target += prior_lpdf(alpha_self_z | alpha_self_prior_family, alpha_self_prior_p1, alpha_self_prior_p2, alpha_self_prior_p3);
  target += prior_lpdf(alpha_other_outcome_z | alpha_other_outcome_prior_family, alpha_other_outcome_prior_p1, alpha_other_outcome_prior_p2, alpha_other_outcome_prior_p3);
  target += prior_lpdf(alpha_other_action_z | alpha_other_action_prior_family, alpha_other_action_prior_p1, alpha_other_action_prior_p2, alpha_other_action_prior_p3);
  target += prior_lpdf(w_imitation_z | w_imitation_prior_family, w_imitation_prior_p1, w_imitation_prior_p2, w_imitation_prior_p3);
  target += prior_lpdf(beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(stickiness | stickiness_prior_family, stickiness_prior_p1, stickiness_prior_p2, stickiness_prior_p3);

  for (e in 1:E) {
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
      T = rep_vector(1.0 / A, A);
      last_self_choice = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = beta * (w_imitation * T + (1 - w_imitation) * Q);
      if (last_self_choice > 0) u[last_self_choice] += stickiness;
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      Q[a] = Q[a] + alpha_self * (step_reward[e] - Q[a]);
      last_self_choice = a;
    }
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      Q[sa] = Q[sa] + alpha_other_outcome * (step_social_reward[e] - Q[sa]);
      T = (1 - alpha_other_action) * T;                                                 // decay all action tendencies toward 0
      T[sa] = T[sa] + alpha_other_action;                                               // chosen action gets the toward-1 increment
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    vector[A] Q = rep_vector(q_init, A);
    vector[A] T = rep_vector(1.0 / A, A);
    int last_self_choice = 0;
    int d = 0;

    for (e in 1:E) {
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        Q = rep_vector(q_init, A);
        T = rep_vector(1.0 / A, A);
        last_self_choice = 0;
      }
      if (step_choice[e] > 0) {
        vector[A] u = beta * (w_imitation * T + (1 - w_imitation) * Q);
        d += 1;
        if (last_self_choice > 0) u[last_self_choice] += stickiness;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        Q[a] = Q[a] + alpha_self * (step_reward[e] - Q[a]);
        last_self_choice = a;
      }
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        Q[sa] = Q[sa] + alpha_other_outcome * (step_social_reward[e] - Q[sa]);
        T = (1 - alpha_other_action) * T;                                                 // decay all action tendencies toward 0
        T[sa] = T[sa] + alpha_other_action;                                               // chosen action gets the toward-1 increment
      }
    }
  }
}
