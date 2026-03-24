/*
 * Model: Asocial Q-Learning
 * Hierarchy: Subject-block-condition (single subject, per-condition parameters via delta offsets)
 * Parameters: alpha[C], beta[C] — one pair per condition, constructed as baseline + condition delta
 *
 * Standard Rescorla-Wagner Q-learning for a single subject with condition-specific parameters.
 * Each condition's alpha and beta are derived by adding an unconstrained delta offset to the shared
 * baseline, then applying the appropriate link function (inv_logit / softplus).
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

  int<lower=2> C;                        // number of experimental conditions
  int<lower=1,upper=C> baseline_cond;    // index of the baseline condition (receives no delta)
  array[E] int<lower=1,upper=C> step_condition; // condition index for each step

  int alpha_prior_family;   // prior family code for the baseline alpha
  real alpha_prior_p1;      // first hyperparameter of the alpha prior
  real alpha_prior_p2;      // second hyperparameter of the alpha prior
  real alpha_prior_p3;      // third hyperparameter of the alpha prior
  int beta_prior_family;    // prior family code for the baseline beta
  real beta_prior_p1;       // first hyperparameter of the beta prior
  real beta_prior_p2;       // second hyperparameter of the beta prior
  real beta_prior_p3;       // third hyperparameter of the beta prior
}
parameters {
  real alpha_shared_z;           // unconstrained baseline learning rate (applies to baseline_cond)
  real beta_shared_z;            // unconstrained baseline inverse temperature
  vector[C - 1] alpha_delta_z;   // unconstrained condition offsets for alpha (one per non-baseline condition)
  vector[C - 1] beta_delta_z;    // unconstrained condition offsets for beta
}
transformed parameters {
  vector<lower=0,upper=1>[C] alpha; // per-condition learning rate in (0,1)
  vector<lower=0>[C] beta;          // per-condition inverse temperature > 0
  {
    int d = 0; // delta index counter (incremented for each non-baseline condition)
    for (c in 1:C) {
      real az = alpha_shared_z; // start from the shared baseline
      real bz = beta_shared_z;
      // Non-baseline conditions add their condition-specific delta
      if (c != baseline_cond) {
        d += 1;
        az += alpha_delta_z[d];
        bz += beta_delta_z[d];
      }
      alpha[c] = inv_logit(az);    // map to (0,1)
      beta[c] = log1p_exp(bz);     // map to (0,inf)
    }
  }
}
model {
  vector[A] Q = rep_vector(q_init, A); // action values, initialised to q_init

  target += prior_lpdf(alpha_shared_z | alpha_prior_family, alpha_prior_p1, alpha_prior_p2, alpha_prior_p3);
  target += prior_lpdf(beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  alpha_delta_z ~ normal(0, 1); // regularising prior on condition deltas
  beta_delta_z ~ normal(0, 1);

  for (e in 1:E) {
    // Reset Q-values when a new block starts (if reset_on_block is enabled)
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
    }
    // DECISION step: add softmax log-probability using this step's condition-specific beta
    if (step_choice[e] > 0) {
      int cc = step_condition[e];                                                // condition at this step
      vector[A] u = beta[cc] * Q;                                               // scale by condition beta
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // UPDATE step: Rescorla-Wagner update using this step's condition-specific alpha
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      int cc = step_condition[e];                               // condition determines the learning rate
      Q[a] = Q[a] + alpha[cc] * (step_reward[e] - Q[a]);       // delta rule with condition alpha
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D); // per-decision log-likelihood for LOO-CV
  {
    vector[A] Q = rep_vector(q_init, A); // local copy of Q-values for this forward pass
    int d = 0;                           // decision counter (indexes into log_lik)

    for (e in 1:E) {
      // Reset Q-values when a new block starts (if reset_on_block is enabled)
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        Q = rep_vector(q_init, A);
      }
      // DECISION step: record per-trial log-likelihood using condition-specific beta
      if (step_choice[e] > 0) {
        d += 1;
        int cc = step_condition[e];
        vector[A] u = beta[cc] * Q;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // UPDATE step: Rescorla-Wagner update using condition-specific alpha
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        int cc = step_condition[e];
        Q[a] = Q[a] + alpha[cc] * (step_reward[e] - Q[a]);
      }
    }
  }
}
