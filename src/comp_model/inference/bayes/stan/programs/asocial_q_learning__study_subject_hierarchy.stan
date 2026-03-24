/*
 * Model: Asocial Q-Learning
 * Hierarchy: Study-subject (multiple subjects, each drawn from a group-level distribution)
 * Parameters: alpha[N], beta[N] — per-subject, non-centredly parameterised from group mean/SD
 *
 * Standard Rescorla-Wagner Q-learning with a two-level hierarchy: subject parameters are
 * drawn from group-level normal distributions in unconstrained space, then mapped to their
 * natural ranges via inv_logit (alpha) and softplus (beta).
 */
functions {
#include "prior_lpdf.stanfunctions"
}
data {
  int<lower=1> N;                                    // number of subjects
  int<lower=1> A;                                    // number of available actions
  int<lower=1> E;                                    // total number of steps (DECISION + UPDATE events)
  int<lower=0> D;                                    // number of DECISION steps (used to size log_lik)
  array[E] int<lower=1,upper=N> step_subject;        // subject index for each step
  array[E] int<lower=0,upper=A> step_choice;         // chosen action at step e; 0 if no choice (UPDATE-only step)
  array[E] int<lower=0,upper=A> step_update_action;  // action whose Q-value is updated at step e; 0 if no update
  vector[E] step_reward;                             // reward received at the update step
  array[E] vector<lower=0,upper=1>[A] step_avail_mask; // binary mask of available actions (1 = available)
  array[E] int step_block;                           // block index for each step (used to detect block boundaries)
  int<lower=0,upper=1> reset_on_block;               // 1 = reset Q-values at each new block, 0 = carry over
  real q_init;                                       // initial Q-value assigned to every action

  int alpha_prior_family;   // prior family code for the group-level alpha mean
  real alpha_prior_p1;      // first hyperparameter of the alpha prior
  real alpha_prior_p2;      // second hyperparameter of the alpha prior
  real alpha_prior_p3;      // third hyperparameter of the alpha prior
  int beta_prior_family;    // prior family code for the group-level beta mean
  real beta_prior_p1;       // first hyperparameter of the beta prior
  real beta_prior_p2;       // second hyperparameter of the beta prior
  real beta_prior_p3;       // third hyperparameter of the beta prior
}
parameters {
  real mu_alpha_z;              // group-level mean of alpha in unconstrained space
  real<lower=0> sd_alpha_z;     // group-level SD of alpha in unconstrained space
  vector[N] raw_alpha_z;        // per-subject standard normal deviates for alpha (non-centred)

  real mu_beta_z;               // group-level mean of beta in unconstrained space
  real<lower=0> sd_beta_z;      // group-level SD of beta in unconstrained space
  vector[N] raw_beta_z;         // per-subject standard normal deviates for beta (non-centred)
}
transformed parameters {
  vector[N] alpha_z = mu_alpha_z + sd_alpha_z * raw_alpha_z; // per-subject alpha in unconstrained space
  vector[N] beta_z = mu_beta_z + sd_beta_z * raw_beta_z;     // per-subject beta in unconstrained space
  vector<lower=0,upper=1>[N] alpha = inv_logit(alpha_z);      // per-subject learning rate in (0,1)
  vector<lower=0>[N] beta = log1p_exp(beta_z);                // per-subject inverse temperature > 0
}
model {
  array[N] vector[A] Q; // per-subject action-value vectors

  target += prior_lpdf(mu_alpha_z | alpha_prior_family, alpha_prior_p1, alpha_prior_p2, alpha_prior_p3);
  sd_alpha_z ~ normal(0, 1);   // half-normal prior on group SD (constrained positive)
  raw_alpha_z ~ normal(0, 1);  // standard normal prior for non-centred parameterisation

  target += prior_lpdf(mu_beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  sd_beta_z ~ normal(0, 1);
  raw_beta_z ~ normal(0, 1);

  for (n in 1:N) Q[n] = rep_vector(q_init, A); // initialise Q-values for all subjects

  for (e in 1:E) {
    int n = step_subject[e]; // subject for this step
    // Reset this subject's Q-values when their block changes (same-subject check prevents cross-subject resets)
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
    }
    // DECISION step: add softmax log-probability for subject n
    if (step_choice[e] > 0) {
      vector[A] u = beta[n] * Q[n];                                              // scale by subject's beta
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // UPDATE step: Rescorla-Wagner update for subject n
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      Q[n][a] = Q[n][a] + alpha[n] * (step_reward[e] - Q[n][a]); // delta rule with subject's alpha
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D); // per-decision log-likelihood for LOO-CV
  {
    array[N] vector[A] Q;
    for (n in 1:N) Q[n] = rep_vector(q_init, A);
    int d = 0; // decision counter (indexes into log_lik)

    for (e in 1:E) {
      int n = step_subject[e];
      // Reset this subject's Q-values when their block changes
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        Q[n] = rep_vector(q_init, A);
      }
      // DECISION step: record per-trial log-likelihood
      if (step_choice[e] > 0) {
        d += 1;
        vector[A] u = beta[n] * Q[n];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // UPDATE step: Rescorla-Wagner update for subject n
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        Q[n][a] = Q[n][a] + alpha[n] * (step_reward[e] - Q[n][a]);
      }
    }
  }
  real alpha_pop = inv_logit(mu_alpha_z); // group-level alpha (population mean on constrained scale)
  real beta_pop = log1p_exp(mu_beta_z);   // group-level beta (population mean on constrained scale)
}
