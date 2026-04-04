/*
 * Model: Sticky Social RL — Self Reward + Demonstrator Action Mixture
 * Hierarchy: Study-subject-block-condition
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

  int<lower=2> C;
  int<lower=1,upper=C> baseline_cond;
  array[E] int<lower=1,upper=C> step_condition;

  int alpha_self_prior_family;
  real alpha_self_prior_p1;
  real alpha_self_prior_p2;
  real alpha_self_prior_p3;
  int alpha_self_delta_prior_family;
  real alpha_self_delta_prior_p1;
  real alpha_self_delta_prior_p2;
  real alpha_self_delta_prior_p3;
  int alpha_other_action_prior_family;
  real alpha_other_action_prior_p1;
  real alpha_other_action_prior_p2;
  real alpha_other_action_prior_p3;
  int alpha_other_action_delta_prior_family;
  real alpha_other_action_delta_prior_p1;
  real alpha_other_action_delta_prior_p2;
  real alpha_other_action_delta_prior_p3;
  int w_imitation_prior_family;
  real w_imitation_prior_p1;
  real w_imitation_prior_p2;
  real w_imitation_prior_p3;
  int w_imitation_delta_prior_family;
  real w_imitation_delta_prior_p1;
  real w_imitation_delta_prior_p2;
  real w_imitation_delta_prior_p3;
  int beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
  int beta_delta_prior_family;
  real beta_delta_prior_p1;
  real beta_delta_prior_p2;
  real beta_delta_prior_p3;
  int stickiness_prior_family;
  real stickiness_prior_p1;
  real stickiness_prior_p2;
  real stickiness_prior_p3;
  int stickiness_delta_prior_family;
  real stickiness_delta_prior_p1;
  real stickiness_delta_prior_p2;
  real stickiness_delta_prior_p3;
  int sd_alpha_self_prior_family;
  real sd_alpha_self_prior_p1;
  real sd_alpha_self_prior_p2;
  real sd_alpha_self_prior_p3;
  int sd_alpha_other_action_prior_family;
  real sd_alpha_other_action_prior_p1;
  real sd_alpha_other_action_prior_p2;
  real sd_alpha_other_action_prior_p3;
  int sd_w_imitation_prior_family;
  real sd_w_imitation_prior_p1;
  real sd_w_imitation_prior_p2;
  real sd_w_imitation_prior_p3;
  int sd_beta_prior_family;
  real sd_beta_prior_p1;
  real sd_beta_prior_p2;
  real sd_beta_prior_p3;
  int sd_stickiness_prior_family;
  real sd_stickiness_prior_p1;
  real sd_stickiness_prior_p2;
  real sd_stickiness_prior_p3;
  int sd_alpha_self_delta_prior_family;
  real sd_alpha_self_delta_prior_p1;
  real sd_alpha_self_delta_prior_p2;
  real sd_alpha_self_delta_prior_p3;
  int sd_alpha_other_action_delta_prior_family;
  real sd_alpha_other_action_delta_prior_p1;
  real sd_alpha_other_action_delta_prior_p2;
  real sd_alpha_other_action_delta_prior_p3;
  int sd_w_imitation_delta_prior_family;
  real sd_w_imitation_delta_prior_p1;
  real sd_w_imitation_delta_prior_p2;
  real sd_w_imitation_delta_prior_p3;
  int sd_beta_delta_prior_family;
  real sd_beta_delta_prior_p1;
  real sd_beta_delta_prior_p2;
  real sd_beta_delta_prior_p3;
  int sd_stickiness_delta_prior_family;
  real sd_stickiness_delta_prior_p1;
  real sd_stickiness_delta_prior_p2;
  real sd_stickiness_delta_prior_p3;
}
parameters {
  real mu_alpha_self_shared_z;
  real<lower=0> sd_alpha_self_shared_z;
  real mu_alpha_other_action_shared_z;
  real<lower=0> sd_alpha_other_action_shared_z;
  real mu_w_imitation_shared_z;
  real<lower=0> sd_w_imitation_shared_z;
  real mu_beta_shared_z;
  real<lower=0> sd_beta_shared_z;
  real mu_stickiness_shared_z;
  real<lower=0> sd_stickiness_shared_z;

  vector[C - 1] mu_alpha_self_delta_z;
  vector<lower=0>[C - 1] sd_alpha_self_delta_z;
  vector[C - 1] mu_alpha_other_action_delta_z;
  vector<lower=0>[C - 1] sd_alpha_other_action_delta_z;
  vector[C - 1] mu_w_imitation_delta_z;
  vector<lower=0>[C - 1] sd_w_imitation_delta_z;
  vector[C - 1] mu_beta_delta_z;
  vector<lower=0>[C - 1] sd_beta_delta_z;
  vector[C - 1] mu_stickiness_delta_z;
  vector<lower=0>[C - 1] sd_stickiness_delta_z;

  vector[N] raw_alpha_self_shared_z;
  vector[N] raw_alpha_other_action_shared_z;
  vector[N] raw_w_imitation_shared_z;
  vector[N] raw_beta_shared_z;
  vector[N] raw_stickiness_shared_z;
  array[C - 1] vector[N] raw_alpha_self_delta_z;
  array[C - 1] vector[N] raw_alpha_other_action_delta_z;
  array[C - 1] vector[N] raw_w_imitation_delta_z;
  array[C - 1] vector[N] raw_beta_delta_z;
  array[C - 1] vector[N] raw_stickiness_delta_z;
}
transformed parameters {
  vector[N] alpha_self_shared_z =
      mu_alpha_self_shared_z + sd_alpha_self_shared_z * raw_alpha_self_shared_z;
  vector[N] alpha_other_action_shared_z =
      mu_alpha_other_action_shared_z + sd_alpha_other_action_shared_z * raw_alpha_other_action_shared_z;
  vector[N] w_imitation_shared_z =
      mu_w_imitation_shared_z + sd_w_imitation_shared_z * raw_w_imitation_shared_z;
  vector[N] beta_shared_z = mu_beta_shared_z + sd_beta_shared_z * raw_beta_shared_z;
  vector[N] stickiness_shared_z =
      mu_stickiness_shared_z + sd_stickiness_shared_z * raw_stickiness_shared_z;

  array[C - 1] vector[N] alpha_self_delta_z;
  array[C - 1] vector[N] alpha_other_action_delta_z;
  array[C - 1] vector[N] w_imitation_delta_z;
  array[C - 1] vector[N] beta_delta_z;
  array[C - 1] vector[N] stickiness_delta_z;
  for (d in 1:(C - 1)) {
    alpha_self_delta_z[d] =
        mu_alpha_self_delta_z[d] + sd_alpha_self_delta_z[d] * raw_alpha_self_delta_z[d];
    alpha_other_action_delta_z[d] =
        mu_alpha_other_action_delta_z[d] + sd_alpha_other_action_delta_z[d] * raw_alpha_other_action_delta_z[d];
    w_imitation_delta_z[d] =
        mu_w_imitation_delta_z[d] + sd_w_imitation_delta_z[d] * raw_w_imitation_delta_z[d];
    beta_delta_z[d] = mu_beta_delta_z[d] + sd_beta_delta_z[d] * raw_beta_delta_z[d];
    stickiness_delta_z[d] =
        mu_stickiness_delta_z[d] + sd_stickiness_delta_z[d] * raw_stickiness_delta_z[d];
  }

  array[N] vector<lower=0,upper=1>[C] alpha_self;
  array[N] vector<lower=0,upper=1>[C] alpha_other_action;
  array[N] vector<lower=0,upper=1>[C] w_imitation;
  array[N] vector<lower=0>[C] beta;
  array[N] vector[C] stickiness;
  for (n in 1:N) {
    int d = 0;
    for (c in 1:C) {
      real asz = alpha_self_shared_z[n];
      real aaz = alpha_other_action_shared_z[n];
      real wiz = w_imitation_shared_z[n];
      real bz = beta_shared_z[n];
      real sz = stickiness_shared_z[n];
      if (c != baseline_cond) {
        d += 1;
        asz += alpha_self_delta_z[d][n];
        aaz += alpha_other_action_delta_z[d][n];
        wiz += w_imitation_delta_z[d][n];
        bz += beta_delta_z[d][n];
        sz += stickiness_delta_z[d][n];
      }
      alpha_self[n][c] = inv_logit(asz);
      alpha_other_action[n][c] = inv_logit(aaz);
      w_imitation[n][c] = inv_logit(wiz);
      beta[n][c] = log1p_exp(bz);
      stickiness[n][c] = sz;
    }
  }
}
model {
  array[N] vector[A] Q;
  array[N] vector[A] T;
  array[N] int last_self_choice;

  target += prior_lpdf(mu_alpha_self_shared_z | alpha_self_prior_family, alpha_self_prior_p1, alpha_self_prior_p2, alpha_self_prior_p3);
  target += prior_lpdf(sd_alpha_self_shared_z | sd_alpha_self_prior_family, sd_alpha_self_prior_p1, sd_alpha_self_prior_p2, sd_alpha_self_prior_p3);
  raw_alpha_self_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_alpha_other_action_shared_z | alpha_other_action_prior_family, alpha_other_action_prior_p1, alpha_other_action_prior_p2, alpha_other_action_prior_p3);
  target += prior_lpdf(sd_alpha_other_action_shared_z | sd_alpha_other_action_prior_family, sd_alpha_other_action_prior_p1, sd_alpha_other_action_prior_p2, sd_alpha_other_action_prior_p3);
  raw_alpha_other_action_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_w_imitation_shared_z | w_imitation_prior_family, w_imitation_prior_p1, w_imitation_prior_p2, w_imitation_prior_p3);
  target += prior_lpdf(sd_w_imitation_shared_z | sd_w_imitation_prior_family, sd_w_imitation_prior_p1, sd_w_imitation_prior_p2, sd_w_imitation_prior_p3);
  raw_w_imitation_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(sd_beta_shared_z | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);
  raw_beta_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_stickiness_shared_z | stickiness_prior_family, stickiness_prior_p1, stickiness_prior_p2, stickiness_prior_p3);
  target += prior_lpdf(sd_stickiness_shared_z | sd_stickiness_prior_family, sd_stickiness_prior_p1, sd_stickiness_prior_p2, sd_stickiness_prior_p3);
  raw_stickiness_shared_z ~ normal(0, 1);

  for (d in 1:(C - 1))
    target += prior_lpdf(mu_alpha_self_delta_z[d] | alpha_self_delta_prior_family, alpha_self_delta_prior_p1, alpha_self_delta_prior_p2, alpha_self_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_alpha_self_delta_z[d] | sd_alpha_self_delta_prior_family, sd_alpha_self_delta_prior_p1, sd_alpha_self_delta_prior_p2, sd_alpha_self_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(mu_alpha_other_action_delta_z[d] | alpha_other_action_delta_prior_family, alpha_other_action_delta_prior_p1, alpha_other_action_delta_prior_p2, alpha_other_action_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_alpha_other_action_delta_z[d] | sd_alpha_other_action_delta_prior_family, sd_alpha_other_action_delta_prior_p1, sd_alpha_other_action_delta_prior_p2, sd_alpha_other_action_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(mu_w_imitation_delta_z[d] | w_imitation_delta_prior_family, w_imitation_delta_prior_p1, w_imitation_delta_prior_p2, w_imitation_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_w_imitation_delta_z[d] | sd_w_imitation_delta_prior_family, sd_w_imitation_delta_prior_p1, sd_w_imitation_delta_prior_p2, sd_w_imitation_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(mu_beta_delta_z[d] | beta_delta_prior_family, beta_delta_prior_p1, beta_delta_prior_p2, beta_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_beta_delta_z[d] | sd_beta_delta_prior_family, sd_beta_delta_prior_p1, sd_beta_delta_prior_p2, sd_beta_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(mu_stickiness_delta_z[d] | stickiness_delta_prior_family, stickiness_delta_prior_p1, stickiness_delta_prior_p2, stickiness_delta_prior_p3);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_stickiness_delta_z[d] | sd_stickiness_delta_prior_family, sd_stickiness_delta_prior_p1, sd_stickiness_delta_prior_p2, sd_stickiness_delta_prior_p3);

  for (d in 1:(C - 1)) {
    raw_alpha_self_delta_z[d] ~ normal(0, 1);
    raw_alpha_other_action_delta_z[d] ~ normal(0, 1);
    raw_w_imitation_delta_z[d] ~ normal(0, 1);
    raw_beta_delta_z[d] ~ normal(0, 1);
    raw_stickiness_delta_z[d] ~ normal(0, 1);
  }

  for (n in 1:N) {
    Q[n] = rep_vector(q_init, A);
    T[n] = rep_vector(1.0 / A, A);
    last_self_choice[n] = 0;
  }

  for (e in 1:E) {
    int n = step_subject[e];
    int cc = step_condition[e];
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
      T[n] = rep_vector(1.0 / A, A);
      last_self_choice[n] = 0;
    }
    if (step_choice[e] > 0) {
      vector[A] u = beta[n][cc] * (w_imitation[n][cc] * T[n] + (1 - w_imitation[n][cc]) * Q[n]);
      if (last_self_choice[n] > 0) u[last_self_choice[n]] += stickiness[n][cc];
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(step_choice[e] | u);
      // Preserve the previous choice even when the trial omits feedback.
      last_self_choice[n] = step_choice[e];
    }
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      Q[n][a] = Q[n][a] + alpha_self[n][cc] * (step_reward[e] - Q[n][a]);
      last_self_choice[n] = a;
    }
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      T[n] = (1 - alpha_other_action[n][cc]) * T[n];
      T[n][sa] = T[n][sa] + alpha_other_action[n][cc];
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D);
  vector<lower=0,upper=1>[C] alpha_self_pop;
  vector<lower=0,upper=1>[C] alpha_other_action_pop;
  vector<lower=0,upper=1>[C] w_imitation_pop;
  vector<lower=0>[C] beta_pop;
  vector[C] stickiness_pop;
  real alpha_self_shared_pop;
  real alpha_other_action_shared_pop;
  real w_imitation_shared_pop;
  real beta_shared_pop;
  real stickiness_shared_pop;
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
      int cc = step_condition[e];
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        Q[n] = rep_vector(q_init, A);
        T[n] = rep_vector(1.0 / A, A);
        last_self_choice[n] = 0;
      }
      if (step_choice[e] > 0) {
        vector[A] u = beta[n][cc] * (w_imitation[n][cc] * T[n] + (1 - w_imitation[n][cc]) * Q[n]);
        d += 1;
        if (last_self_choice[n] > 0) u[last_self_choice[n]] += stickiness[n][cc];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
        // Preserve the previous choice even when the trial omits feedback.
        last_self_choice[n] = step_choice[e];
      }
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        Q[n][a] = Q[n][a] + alpha_self[n][cc] * (step_reward[e] - Q[n][a]);
        last_self_choice[n] = a;
      }
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        T[n] = (1 - alpha_other_action[n][cc]) * T[n];
        T[n][sa] = T[n][sa] + alpha_other_action[n][cc];
      }
    }
  }

  {
    int d_idx = 0;
    for (c in 1:C) {
      real asz = mu_alpha_self_shared_z;
      real aaz = mu_alpha_other_action_shared_z;
      real wiz = mu_w_imitation_shared_z;
      real bz = mu_beta_shared_z;
      real sz = mu_stickiness_shared_z;
      if (c != baseline_cond) {
        d_idx += 1;
        asz += mu_alpha_self_delta_z[d_idx];
        aaz += mu_alpha_other_action_delta_z[d_idx];
        wiz += mu_w_imitation_delta_z[d_idx];
        bz += mu_beta_delta_z[d_idx];
        sz += mu_stickiness_delta_z[d_idx];
      }
      alpha_self_pop[c] = inv_logit(asz);
      alpha_other_action_pop[c] = inv_logit(aaz);
      w_imitation_pop[c] = inv_logit(wiz);
      beta_pop[c] = log1p_exp(bz);
      stickiness_pop[c] = sz;
    }
  }

  alpha_self_shared_pop = alpha_self_pop[baseline_cond];
  alpha_other_action_shared_pop = alpha_other_action_pop[baseline_cond];
  w_imitation_shared_pop = w_imitation_pop[baseline_cond];
  beta_shared_pop = beta_pop[baseline_cond];
  stickiness_shared_pop = stickiness_pop[baseline_cond];
}
