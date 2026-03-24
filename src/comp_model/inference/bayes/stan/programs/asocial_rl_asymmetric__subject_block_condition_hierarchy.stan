/*
 * Model: Asocial RL — Asymmetric Learning Rates
 * Hierarchy: Subject-block-condition (single subject, per-condition parameters via delta offsets)
 * Parameters: alpha_pos[C], alpha_neg[C], beta[C] — one triple per condition, baseline + delta
 *
 * Asymmetric Rescorla-Wagner Q-learning for a single subject with condition-specific parameters.
 * Each condition's alpha_pos, alpha_neg and beta are constructed by adding an unconstrained
 * delta to the shared baseline, then applying the appropriate link function.
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

  int alpha_pos_prior_family;   // prior family code for the baseline alpha_pos
  real alpha_pos_prior_p1;      // first hyperparameter of the alpha_pos prior
  real alpha_pos_prior_p2;      // second hyperparameter of the alpha_pos prior
  real alpha_pos_prior_p3;      // third hyperparameter of the alpha_pos prior
  int alpha_neg_prior_family;   // prior family code for the baseline alpha_neg
  real alpha_neg_prior_p1;      // first hyperparameter of the alpha_neg prior
  real alpha_neg_prior_p2;      // second hyperparameter of the alpha_neg prior
  real alpha_neg_prior_p3;      // third hyperparameter of the alpha_neg prior
  int beta_prior_family;        // prior family code for the baseline beta
  real beta_prior_p1;           // first hyperparameter of the beta prior
  real beta_prior_p2;           // second hyperparameter of the beta prior
  real beta_prior_p3;           // third hyperparameter of the beta prior
}
parameters {
  real alpha_pos_shared_z;           // unconstrained baseline positive learning rate
  real alpha_neg_shared_z;           // unconstrained baseline negative learning rate
  real beta_shared_z;                // unconstrained baseline inverse temperature
  vector[C - 1] alpha_pos_delta_z;   // unconstrained condition offsets for alpha_pos (one per non-baseline condition)
  vector[C - 1] alpha_neg_delta_z;   // unconstrained condition offsets for alpha_neg
  vector[C - 1] beta_delta_z;        // unconstrained condition offsets for beta
}
transformed parameters {
  vector<lower=0,upper=1>[C] alpha_pos; // per-condition positive learning rate in (0,1)
  vector<lower=0,upper=1>[C] alpha_neg; // per-condition negative learning rate in (0,1)
  vector<lower=0>[C] beta;              // per-condition inverse temperature > 0
  {
    int d = 0; // delta index counter (incremented for each non-baseline condition)
    for (c in 1:C) {
      real apz = alpha_pos_shared_z; // start from the shared baseline
      real anz = alpha_neg_shared_z;
      real bz = beta_shared_z;
      // Non-baseline conditions add their condition-specific delta
      if (c != baseline_cond) {
        d += 1;
        apz += alpha_pos_delta_z[d];
        anz += alpha_neg_delta_z[d];
        bz += beta_delta_z[d];
      }
      alpha_pos[c] = inv_logit(apz); // map to (0,1)
      alpha_neg[c] = inv_logit(anz);
      beta[c] = log1p_exp(bz);       // map to (0,inf)
    }
  }
}
model {
  vector[A] Q = rep_vector(q_init, A); // action values, initialised to q_init

  target += prior_lpdf(alpha_pos_shared_z | alpha_pos_prior_family, alpha_pos_prior_p1, alpha_pos_prior_p2, alpha_pos_prior_p3);
  target += prior_lpdf(alpha_neg_shared_z | alpha_neg_prior_family, alpha_neg_prior_p1, alpha_neg_prior_p2, alpha_neg_prior_p3);
  target += prior_lpdf(beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  alpha_pos_delta_z ~ normal(0, 1); // regularising prior on condition deltas
  alpha_neg_delta_z ~ normal(0, 1);
  beta_delta_z ~ normal(0, 1);

  for (e in 1:E) {
    // Reset Q-values when a new block starts (if reset_on_block is enabled)
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
    }
    // DECISION step: add softmax log-probability using the condition-specific beta
    if (step_choice[e] > 0) {
      int cc = step_condition[e];                                                // condition at this step
      vector[A] u = beta[cc] * Q;                                               // scale by condition beta
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // UPDATE step: asymmetric update using the condition-specific alpha_pos / alpha_neg
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      int cc = step_condition[e];
      real delta = step_reward[e] - Q[a];                                               // prediction error
      Q[a] = Q[a] + (delta >= 0 ? alpha_pos[cc] : alpha_neg[cc]) * delta; // branch on sign of PE
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
      // DECISION step: record per-trial log-likelihood with condition-specific beta
      if (step_choice[e] > 0) {
        d += 1;
        int cc = step_condition[e];
        vector[A] u = beta[cc] * Q;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // UPDATE step: asymmetric update with condition-specific alphas
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        int cc = step_condition[e];
        real delta = step_reward[e] - Q[a];
        Q[a] = Q[a] + (delta >= 0 ? alpha_pos[cc] : alpha_neg[cc]) * delta;
      }
    }
  }
}
