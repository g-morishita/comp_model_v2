/*
 * Model: Asocial RL — Asymmetric Learning Rates
 * Hierarchy: Subject-shared (single subject, one parameter set for all conditions)
 * Parameters: alpha_pos (learning rate for positive PEs), alpha_neg (learning rate for negative PEs), beta (inverse temperature)
 *
 * Rescorla-Wagner Q-learning for a single subject where the learning rate differs
 * depending on the sign of the prediction error: alpha_pos is used when reward exceeds
 * the Q-value, alpha_neg when it falls below.
 */
functions {
#include "prior_lpdf.stanfunctions"
}
data {
  int<lower=1> A;                                    // number of available actions
  int<lower=1> E;                                    // total number of steps (DECISION + UPDATE events)
  int<lower=0> D;                                    // number of DECISION steps (used to size log_lik)
  array[E] int<lower=0,upper=A> step_choice;         // chosen action at step e; 0 if no choice (UPDATE-only step)
  array[E] int<lower=0,upper=A> step_update_action;  // action whose Q-value is updated at step e; 0 if no update
  vector[E] step_reward;                             // reward received at the update step
  array[E] vector<lower=0,upper=1>[A] step_avail_mask; // binary mask of available actions (1 = available)
  array[E] int step_block;                           // block index for each step (used to detect block boundaries)
  int<lower=0,upper=1> reset_on_block;               // 1 = reset Q-values at each new block, 0 = carry over
  real q_init;                                       // initial Q-value assigned to every action

  int alpha_pos_prior_family;   // prior family code for alpha_pos
  real alpha_pos_prior_p1;      // first hyperparameter of the alpha_pos prior
  real alpha_pos_prior_p2;      // second hyperparameter of the alpha_pos prior
  real alpha_pos_prior_p3;      // third hyperparameter of the alpha_pos prior
  int alpha_neg_prior_family;   // prior family code for alpha_neg
  real alpha_neg_prior_p1;      // first hyperparameter of the alpha_neg prior
  real alpha_neg_prior_p2;      // second hyperparameter of the alpha_neg prior
  real alpha_neg_prior_p3;      // third hyperparameter of the alpha_neg prior
  int beta_prior_family;        // prior family code for beta
  real beta_prior_p1;           // first hyperparameter of the beta prior
  real beta_prior_p2;           // second hyperparameter of the beta prior
  real beta_prior_p3;           // third hyperparameter of the beta prior
}
parameters {
  real alpha_pos_z; // unconstrained learning rate for positive prediction errors
  real alpha_neg_z; // unconstrained learning rate for negative prediction errors
  real beta_z;      // unconstrained inverse temperature
}
transformed parameters {
  real<lower=0,upper=1> alpha_pos = inv_logit(alpha_pos_z); // learning rate for positive PEs, in (0,1)
  real<lower=0,upper=1> alpha_neg = inv_logit(alpha_neg_z); // learning rate for negative PEs, in (0,1)
  real<lower=0> beta = log1p_exp(beta_z);                   // inverse temperature > 0 (softplus)
}
model {
  vector[A] Q = rep_vector(q_init, A); // action values, initialised to q_init

  target += prior_lpdf(alpha_pos_z | alpha_pos_prior_family, alpha_pos_prior_p1, alpha_pos_prior_p2, alpha_pos_prior_p3);
  target += prior_lpdf(alpha_neg_z | alpha_neg_prior_family, alpha_neg_prior_p1, alpha_neg_prior_p2, alpha_neg_prior_p3);
  target += prior_lpdf(beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);

  for (e in 1:E) {
    // Reset Q-values when a new block starts (if reset_on_block is enabled)
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
    }
    // DECISION step: add softmax log-probability of the observed choice
    if (step_choice[e] > 0) {
      vector[A] u = beta * Q;                                                  // scale by inverse temperature
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // UPDATE step: asymmetric Rescorla-Wagner update
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      real delta = step_reward[e] - Q[a];                                          // prediction error
      Q[a] = Q[a] + (delta >= 0 ? alpha_pos : alpha_neg) * delta; // use alpha_pos if PE >= 0, else alpha_neg
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D); // per-decision log-likelihood for WAIC model comparison
  {
    vector[A] Q = rep_vector(q_init, A); // local copy of Q-values for this forward pass
    int d = 0;                           // decision counter (indexes into log_lik)

    for (e in 1:E) {
      // Reset Q-values when a new block starts (if reset_on_block is enabled)
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        Q = rep_vector(q_init, A);
      }
      // DECISION step: record per-trial log-likelihood
      if (step_choice[e] > 0) {
        d += 1;
        vector[A] u = beta * Q;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // UPDATE step: asymmetric Rescorla-Wagner update
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        real delta = step_reward[e] - Q[a];
        Q[a] = Q[a] + (delta >= 0 ? alpha_pos : alpha_neg) * delta;
      }
    }
  }
}
