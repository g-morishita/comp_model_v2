/*
 * Model: Sticky Social RL — Demonstrator Mixture
 * Hierarchy: Subject-block-condition
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
  real alpha_other_outcome_shared_z;
  real alpha_other_action_shared_z;
  real w_imitation_shared_z;
  real beta_shared_z;
  real stickiness_shared_z;
  vector[C - 1] alpha_other_outcome_delta_z;
  vector[C - 1] alpha_other_action_delta_z;
  vector[C - 1] w_imitation_delta_z;
  vector[C - 1] beta_delta_z;
  vector[C - 1] stickiness_delta_z;
}
transformed parameters {
  vector<lower=0,upper=1>[C] alpha_other_outcome;
  vector<lower=0,upper=1>[C] alpha_other_action;
  vector<lower=0,upper=1>[C] w_imitation;
  vector<lower=0>[C] beta;
  vector[C] stickiness;
  {
    int d = 0;
    for (c in 1:C) {
      real aoo_z = alpha_other_outcome_shared_z;
      real aoa_z = alpha_other_action_shared_z;
      real wi_z = w_imitation_shared_z;
      real bz = beta_shared_z;
      real sz = stickiness_shared_z;
      if (c != baseline_cond) {
        d += 1;
        aoo_z += alpha_other_outcome_delta_z[d];
        aoa_z += alpha_other_action_delta_z[d];
        wi_z += w_imitation_delta_z[d];
        bz += beta_delta_z[d];
        sz += stickiness_delta_z[d];
      }
      alpha_other_outcome[c] = inv_logit(aoo_z);
      alpha_other_action[c] = inv_logit(aoa_z);
      w_imitation[c] = inv_logit(wi_z);
      beta[c] = log1p_exp(bz);
      stickiness[c] = sz;
    }
  }
}
model {
  vector[A] Q = rep_vector(q_init, A);
  vector[A] T = rep_vector(1.0 / A, A);
  int last_self_choice = 0;

  target += prior_lpdf(alpha_other_outcome_shared_z | alpha_other_outcome_prior_family, alpha_other_outcome_prior_p1, alpha_other_outcome_prior_p2, alpha_other_outcome_prior_p3);
  target += prior_lpdf(alpha_other_action_shared_z | alpha_other_action_prior_family, alpha_other_action_prior_p1, alpha_other_action_prior_p2, alpha_other_action_prior_p3);
  target += prior_lpdf(w_imitation_shared_z | w_imitation_prior_family, w_imitation_prior_p1, w_imitation_prior_p2, w_imitation_prior_p3);
  target += prior_lpdf(beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(stickiness_shared_z | stickiness_prior_family, stickiness_prior_p1, stickiness_prior_p2, stickiness_prior_p3);
  alpha_other_outcome_delta_z ~ normal(0, 1);
  alpha_other_action_delta_z ~ normal(0, 1);
  w_imitation_delta_z ~ normal(0, 1);
  beta_delta_z ~ normal(0, 1);
  stickiness_delta_z ~ normal(0, 1);

  for (e in 1:E) {
    int cc = step_condition[e];
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
      T = rep_vector(1.0 / A, A);
      last_self_choice = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = beta[cc] * (w_imitation[cc] * T + (1 - w_imitation[cc]) * Q);
      if (last_self_choice > 0) u[last_self_choice] += stickiness[cc];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
      last_self_choice = step_update_action[e];
    }
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      Q[sa] = Q[sa] + alpha_other_outcome[cc] * (step_social_reward[e] - Q[sa]);
      T = (1 - alpha_other_action[cc]) * T;
      T[sa] = T[sa] + alpha_other_action[cc];
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
      int cc = step_condition[e];
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        Q = rep_vector(q_init, A);
        T = rep_vector(1.0 / A, A);
        last_self_choice = 0;
      }
      if (step_choice[e] > 0) {
        vector[A] u = beta[cc] * (w_imitation[cc] * T + (1 - w_imitation[cc]) * Q);
        d += 1;
        if (last_self_choice > 0) u[last_self_choice] += stickiness[cc];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      if (step_update_action[e] > 0) {
        last_self_choice = step_update_action[e];
      }
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        Q[sa] = Q[sa] + alpha_other_outcome[cc] * (step_social_reward[e] - Q[sa]);
        T = (1 - alpha_other_action[cc]) * T;
        T[sa] = T[sa] + alpha_other_action[cc];
      }
    }
  }
}
