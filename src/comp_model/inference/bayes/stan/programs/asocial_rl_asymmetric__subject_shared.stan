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

  int alpha_pos_prior_family;
  real alpha_pos_prior_p1;
  real alpha_pos_prior_p2;
  real alpha_pos_prior_p3;
  int alpha_neg_prior_family;
  real alpha_neg_prior_p1;
  real alpha_neg_prior_p2;
  real alpha_neg_prior_p3;
  int beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
}
parameters {
  real alpha_pos_z;
  real alpha_neg_z;
  real beta_z;
}
transformed parameters {
  real<lower=0,upper=1> alpha_pos = inv_logit(alpha_pos_z);
  real<lower=0,upper=1> alpha_neg = inv_logit(alpha_neg_z);
  real<lower=0> beta = log1p_exp(beta_z);
}
model {
  vector[A] Q = rep_vector(q_init, A);

  target += prior_lpdf(alpha_pos_z | alpha_pos_prior_family, alpha_pos_prior_p1, alpha_pos_prior_p2, alpha_pos_prior_p3);
  target += prior_lpdf(alpha_neg_z | alpha_neg_prior_family, alpha_neg_prior_p1, alpha_neg_prior_p2, alpha_neg_prior_p3);
  target += prior_lpdf(beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);

  for (e in 1:E) {
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
    }
    if (step_choice[e] > 0) {
      vector[A] u = beta * Q;
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      real delta = step_reward[e] - Q[a];
      Q[a] = Q[a] + (delta >= 0 ? alpha_pos : alpha_neg) * delta;
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  {
    vector[A] Q = rep_vector(q_init, A);
    int d = 0;

    for (e in 1:E) {
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        Q = rep_vector(q_init, A);
      }
      if (step_choice[e] > 0) {
        d += 1;
        vector[A] u = beta * Q;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        real delta = step_reward[e] - Q[a];
        Q[a] = Q[a] + (delta >= 0 ? alpha_pos : alpha_neg) * delta;
      }
    }
  }
}
