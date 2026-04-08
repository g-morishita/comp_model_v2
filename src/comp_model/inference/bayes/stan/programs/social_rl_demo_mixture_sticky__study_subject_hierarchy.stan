/*
 * Model: Sticky Social RL — Demonstrator Mixture
 * Hierarchy: Study-subject
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
  real q_init;

  array[E] int<lower=0,upper=A> step_social_action;
  vector[E] step_social_reward;

  int alpha_other_outcome_prior_family;
  real alpha_other_outcome_prior_p1;
  real alpha_other_outcome_prior_p2;
  real alpha_other_outcome_prior_p3;
  int sd_alpha_other_outcome_prior_family;
  real sd_alpha_other_outcome_prior_p1;
  real sd_alpha_other_outcome_prior_p2;
  real sd_alpha_other_outcome_prior_p3;
  int alpha_other_action_prior_family;
  real alpha_other_action_prior_p1;
  real alpha_other_action_prior_p2;
  real alpha_other_action_prior_p3;
  int sd_alpha_other_action_prior_family;
  real sd_alpha_other_action_prior_p1;
  real sd_alpha_other_action_prior_p2;
  real sd_alpha_other_action_prior_p3;
  int w_imitation_prior_family;
  real w_imitation_prior_p1;
  real w_imitation_prior_p2;
  real w_imitation_prior_p3;
  int sd_w_imitation_prior_family;
  real sd_w_imitation_prior_p1;
  real sd_w_imitation_prior_p2;
  real sd_w_imitation_prior_p3;
  int beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
  int sd_beta_prior_family;
  real sd_beta_prior_p1;
  real sd_beta_prior_p2;
  real sd_beta_prior_p3;
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
  real mu_alpha_other_outcome_z;
  real<lower=0> sd_alpha_other_outcome_z;
  vector[N] raw_alpha_other_outcome_z;

  real mu_alpha_other_action_z;
  real<lower=0> sd_alpha_other_action_z;
  vector[N] raw_alpha_other_action_z;

  real mu_w_imitation_z;
  real<lower=0> sd_w_imitation_z;
  vector[N] raw_w_imitation_z;

  real mu_beta_z;
  real<lower=0> sd_beta_z;
  vector[N] raw_beta_z;

  real mu_stickiness_z;
  real<lower=0> sd_stickiness_z;
  vector[N] raw_stickiness_z;
}
transformed parameters {
  vector[N] alpha_other_outcome_z = mu_alpha_other_outcome_z
                                    + sd_alpha_other_outcome_z * raw_alpha_other_outcome_z;
  vector[N] alpha_other_action_z = mu_alpha_other_action_z
                                   + sd_alpha_other_action_z * raw_alpha_other_action_z;
  vector[N] w_imitation_z = mu_w_imitation_z + sd_w_imitation_z * raw_w_imitation_z;
  vector[N] beta_z = mu_beta_z + sd_beta_z * raw_beta_z;
  vector[N] stickiness_z = mu_stickiness_z + sd_stickiness_z * raw_stickiness_z;

  vector<lower=0,upper=1>[N] alpha_other_outcome = inv_logit(alpha_other_outcome_z);
  vector<lower=0,upper=1>[N] alpha_other_action = inv_logit(alpha_other_action_z);
  vector<lower=0,upper=1>[N] w_imitation = inv_logit(w_imitation_z);
  vector<lower=0>[N] beta = log1p_exp(beta_z);
  vector[N] stickiness = stickiness_z;
}
model {
  array[N] vector[A] Q;
  array[N] vector[A] T;
  array[N] int last_self_choice;

  target += prior_lpdf(mu_alpha_other_outcome_z | alpha_other_outcome_prior_family, alpha_other_outcome_prior_p1, alpha_other_outcome_prior_p2, alpha_other_outcome_prior_p3);
  target += prior_lpdf(sd_alpha_other_outcome_z | sd_alpha_other_outcome_prior_family, sd_alpha_other_outcome_prior_p1, sd_alpha_other_outcome_prior_p2, sd_alpha_other_outcome_prior_p3);
  raw_alpha_other_outcome_z ~ normal(0, 1);

  target += prior_lpdf(mu_alpha_other_action_z | alpha_other_action_prior_family, alpha_other_action_prior_p1, alpha_other_action_prior_p2, alpha_other_action_prior_p3);
  target += prior_lpdf(sd_alpha_other_action_z | sd_alpha_other_action_prior_family, sd_alpha_other_action_prior_p1, sd_alpha_other_action_prior_p2, sd_alpha_other_action_prior_p3);
  raw_alpha_other_action_z ~ normal(0, 1);

  target += prior_lpdf(mu_w_imitation_z | w_imitation_prior_family, w_imitation_prior_p1, w_imitation_prior_p2, w_imitation_prior_p3);
  target += prior_lpdf(sd_w_imitation_z | sd_w_imitation_prior_family, sd_w_imitation_prior_p1, sd_w_imitation_prior_p2, sd_w_imitation_prior_p3);
  raw_w_imitation_z ~ normal(0, 1);

  target += prior_lpdf(mu_beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(sd_beta_z | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);
  raw_beta_z ~ normal(0, 1);

  target += prior_lpdf(mu_stickiness_z | stickiness_prior_family, stickiness_prior_p1, stickiness_prior_p2, stickiness_prior_p3);
  target += prior_lpdf(sd_stickiness_z | sd_stickiness_prior_family, sd_stickiness_prior_p1, sd_stickiness_prior_p2, sd_stickiness_prior_p3);
  raw_stickiness_z ~ normal(0, 1);

  for (n in 1:N) {
    Q[n] = rep_vector(q_init, A);
    T[n] = rep_vector(1.0 / A, A);
    last_self_choice[n] = 0;
  }

  for (e in 1:E) {
    int n = step_subject[e];
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
      T[n] = rep_vector(1.0 / A, A);
      last_self_choice[n] = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = beta[n] * (w_imitation[n] * T[n] + (1 - w_imitation[n]) * Q[n]);
      if (last_self_choice[n] > 0) u[last_self_choice[n]] += stickiness[n];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
      last_self_choice[n] = step_choice[e];
    }
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      Q[n][sa] = Q[n][sa] + alpha_other_outcome[n] * (step_social_reward[e] - Q[n][sa]);
      T[n] = (1 - alpha_other_action[n]) * T[n];
      T[n][sa] = T[n][sa] + alpha_other_action[n];
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    array[N] vector[A] Q;
    array[N] vector[A] T;
    array[N] int last_self_choice;
    int d = 0;

    for (n in 1:N) {
      Q[n] = rep_vector(q_init, A);
      T[n] = rep_vector(1.0 / A, A);
      last_self_choice[n] = 0;
    }

    for (e in 1:E) {
      int n = step_subject[e];
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        Q[n] = rep_vector(q_init, A);
        T[n] = rep_vector(1.0 / A, A);
        last_self_choice[n] = 0;
      }
      if (step_choice[e] > 0) {
        vector[A] u = beta[n] * (w_imitation[n] * T[n] + (1 - w_imitation[n]) * Q[n]);
        d += 1;
        if (last_self_choice[n] > 0) u[last_self_choice[n]] += stickiness[n];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
        last_self_choice[n] = step_choice[e];
      }
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        Q[n][sa] = Q[n][sa] + alpha_other_outcome[n] * (step_social_reward[e] - Q[n][sa]);
        T[n] = (1 - alpha_other_action[n]) * T[n];
        T[n][sa] = T[n][sa] + alpha_other_action[n];
      }
    }
  }
  real alpha_other_outcome_pop = inv_logit(mu_alpha_other_outcome_z);
  real alpha_other_action_pop = inv_logit(mu_alpha_other_action_z);
  real w_imitation_pop = inv_logit(mu_w_imitation_z);
  real beta_pop = log1p_exp(mu_beta_z);
  real stickiness_pop = mu_stickiness_z;
}
