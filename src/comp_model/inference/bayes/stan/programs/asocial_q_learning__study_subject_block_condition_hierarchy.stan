/*
 * Model: Asocial Q-Learning
 * Hierarchy: Study-subject-block-condition (multiple subjects, per-subject per-condition parameters)
 * Parameters: alpha[N][C], beta[N][C] — per-subject per-condition, built from group-level shared + delta hierarchies
 *
 * Two-level hierarchical Q-learning where each subject has a condition-specific alpha and beta.
 * The shared (baseline) component and each condition's delta are each drawn from their own
 * group-level normal distributions in unconstrained space (non-centred parameterisation).
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

  int<lower=2> C;                        // number of experimental conditions
  int<lower=1,upper=C> baseline_cond;    // index of the baseline condition (receives no delta)
  array[E] int<lower=1,upper=C> step_condition; // condition index for each step

  int alpha_prior_family;   // prior family code for the group-level shared alpha mean
  real alpha_prior_p1;      // first hyperparameter of the alpha prior
  real alpha_prior_p2;      // second hyperparameter of the alpha prior
  real alpha_prior_p3;      // third hyperparameter of the alpha prior
  int beta_prior_family;    // prior family code for the group-level shared beta mean
  real beta_prior_p1;       // first hyperparameter of the beta prior
  real beta_prior_p2;       // second hyperparameter of the beta prior
  real beta_prior_p3;       // third hyperparameter of the beta prior
}
parameters {
  // Population-level: shared
  real mu_alpha_shared_z;              // group mean of the baseline alpha (unconstrained)
  real<lower=0> sd_alpha_shared_z;     // group SD of the baseline alpha (unconstrained)
  real mu_beta_shared_z;               // group mean of the baseline beta (unconstrained)
  real<lower=0> sd_beta_shared_z;      // group SD of the baseline beta (unconstrained)

  // Population-level: deltas
  vector[C - 1] mu_alpha_delta_z;          // group means of the per-condition alpha deltas (unconstrained)
  vector<lower=0>[C - 1] sd_alpha_delta_z; // group SDs of the per-condition alpha deltas
  vector[C - 1] mu_beta_delta_z;           // group means of the per-condition beta deltas (unconstrained)
  vector<lower=0>[C - 1] sd_beta_delta_z;  // group SDs of the per-condition beta deltas

  // Per-subject raw (non-centered)
  vector[N] raw_alpha_shared_z;            // per-subject standard normal deviates for baseline alpha
  vector[N] raw_beta_shared_z;             // per-subject standard normal deviates for baseline beta
  array[C - 1] vector[N] raw_alpha_delta_z; // per-subject standard normal deviates for alpha deltas
  array[C - 1] vector[N] raw_beta_delta_z;  // per-subject standard normal deviates for beta deltas
}
transformed parameters {
  // Per-subject unconstrained shared
  vector[N] alpha_shared_z = mu_alpha_shared_z + sd_alpha_shared_z * raw_alpha_shared_z; // subject baseline alpha (unconstrained)
  vector[N] beta_shared_z = mu_beta_shared_z + sd_beta_shared_z * raw_beta_shared_z;     // subject baseline beta (unconstrained)

  // Per-subject unconstrained deltas
  array[C - 1] vector[N] alpha_delta_z; // per-subject condition alpha deltas (unconstrained)
  array[C - 1] vector[N] beta_delta_z;  // per-subject condition beta deltas (unconstrained)
  for (d in 1:(C - 1)) {
    alpha_delta_z[d] = mu_alpha_delta_z[d] + sd_alpha_delta_z[d] * raw_alpha_delta_z[d];
    beta_delta_z[d] = mu_beta_delta_z[d] + sd_beta_delta_z[d] * raw_beta_delta_z[d];
  }

  // Per-subject, per-condition constrained parameters
  array[N] vector<lower=0,upper=1>[C] alpha; // per-subject per-condition learning rate in (0,1)
  array[N] vector<lower=0>[C] beta;          // per-subject per-condition inverse temperature > 0
  for (n in 1:N) {
    int d = 0; // delta index counter
    for (c in 1:C) {
      real az = alpha_shared_z[n]; // start from this subject's baseline
      real bz = beta_shared_z[n];
      // Non-baseline conditions add the subject's condition-specific delta
      if (c != baseline_cond) {
        d += 1;
        az += alpha_delta_z[d][n];
        bz += beta_delta_z[d][n];
      }
      alpha[n][c] = inv_logit(az);  // map to (0,1)
      beta[n][c] = log1p_exp(bz);   // map to (0,inf)
    }
  }
}
model {
  array[N] vector[A] Q; // per-subject action-value vectors

  // Priors: shared
  target += prior_lpdf(mu_alpha_shared_z | alpha_prior_family, alpha_prior_p1, alpha_prior_p2, alpha_prior_p3);
  sd_alpha_shared_z ~ normal(0, 1);   // half-normal prior on group SD
  raw_alpha_shared_z ~ normal(0, 1);  // non-centred parameterisation

  target += prior_lpdf(mu_beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  sd_beta_shared_z ~ normal(0, 1);
  raw_beta_shared_z ~ normal(0, 1);

  // Priors: deltas
  mu_alpha_delta_z ~ normal(0, 1);   // regularising prior on group-level delta means
  sd_alpha_delta_z ~ normal(0, 1);
  mu_beta_delta_z ~ normal(0, 1);
  sd_beta_delta_z ~ normal(0, 1);
  for (d in 1:(C - 1)) {
    raw_alpha_delta_z[d] ~ normal(0, 1); // non-centred deviates for per-subject alpha deltas
    raw_beta_delta_z[d] ~ normal(0, 1);
  }

  for (n in 1:N) Q[n] = rep_vector(q_init, A); // initialise Q-values for all subjects

  for (e in 1:E) {
    int n = step_subject[e]; // subject for this step
    // Reset this subject's Q-values when their block changes
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
    }
    // DECISION step: add softmax log-probability using subject n's condition-specific beta
    if (step_choice[e] > 0) {
      int cc = step_condition[e];                                                // condition at this step
      vector[A] u = beta[n][cc] * Q[n];                                         // scale by subject's condition beta
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // UPDATE step: Rescorla-Wagner update with subject n's condition-specific alpha
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      int cc = step_condition[e];
      Q[n][a] = Q[n][a] + alpha[n][cc] * (step_reward[e] - Q[n][a]); // delta rule
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
        int cc = step_condition[e];
        vector[A] u = beta[n][cc] * Q[n];
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // UPDATE step: Rescorla-Wagner update with subject n's condition-specific alpha
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        int cc = step_condition[e];
        Q[n][a] = Q[n][a] + alpha[n][cc] * (step_reward[e] - Q[n][a]);
      }
    }
  }

  // Population-level constrained parameters (baseline condition)
  real alpha_shared_pop = inv_logit(mu_alpha_shared_z); // group-mean baseline alpha on constrained scale
  real beta_shared_pop = log1p_exp(mu_beta_shared_z);   // group-mean baseline beta on constrained scale
}
