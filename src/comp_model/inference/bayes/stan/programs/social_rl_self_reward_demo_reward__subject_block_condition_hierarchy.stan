/*
 * Model: Social RL — Self Reward + Demonstrator Reward
 * Hierarchy: Subject-block-condition (single subject, per-condition parameters via delta offsets)
 * Parameters: alpha_self[C], alpha_other[C], beta[C] — one triple per condition, baseline + delta
 *
 * Social Q-learning (own + demonstrator outcomes) for a single subject with condition-specific
 * parameters. Each condition's alpha_self, alpha_other, and beta are constructed by adding an
 * unconstrained delta to the shared baseline, then applying the appropriate link function.
 */
functions {
#include "prior_lpdf.stanfunctions"
}
data {
  int<lower=1> A;                                    // number of available actions
  int<lower=1> E;                                    // total number of steps (DECISION + UPDATE events)
  int<lower=0> D;                                    // number of DECISION steps (used to size log_lik)
  array[E] int<lower=0,upper=A> step_choice;         // chosen action at step e; 0 if no choice (UPDATE-only step)
  array[E] int<lower=0,upper=A> step_update_action;  // action for own-outcome update; 0 if no self-update at this step
  vector[E] step_reward;                             // own reward received at the update step
  array[E] vector<lower=0,upper=1>[A] step_avail_mask; // binary mask of available actions (1 = available)
  array[E] int step_block;                           // block index for each step (used to detect block boundaries)
  int<lower=0,upper=1> reset_on_block;               // 1 = reset Q-values at each new block, 0 = carry over
  real q_init;                                       // initial Q-value assigned to every action

  array[E] int<lower=0,upper=A> step_social_action;  // demonstrator's action at this step; 0 if no social info
  vector[E] step_social_reward;                      // demonstrator's reward at this step (0 if no social info)

  int<lower=2> C;                        // number of experimental conditions
  int<lower=1,upper=C> baseline_cond;    // index of the baseline condition (receives no delta)
  array[E] int<lower=1,upper=C> step_condition; // condition index for each step

  int alpha_self_prior_family;   // prior family code for the baseline alpha_self
  real alpha_self_prior_p1;      // first hyperparameter of the alpha_self prior
  real alpha_self_prior_p2;      // second hyperparameter of the alpha_self prior
  real alpha_self_prior_p3;      // third hyperparameter of the alpha_self prior
  int alpha_other_prior_family;  // prior family code for the baseline alpha_other
  real alpha_other_prior_p1;     // first hyperparameter of the alpha_other prior
  real alpha_other_prior_p2;     // second hyperparameter of the alpha_other prior
  real alpha_other_prior_p3;     // third hyperparameter of the alpha_other prior
  int beta_prior_family;         // prior family code for the baseline beta
  real beta_prior_p1;            // first hyperparameter of the beta prior
  real beta_prior_p2;            // second hyperparameter of the beta prior
  real beta_prior_p3;            // third hyperparameter of the beta prior
}
parameters {
  real alpha_self_shared_z;           // unconstrained baseline own-outcome learning rate
  real alpha_other_shared_z;          // unconstrained baseline demonstrator-outcome learning rate
  real beta_shared_z;                 // unconstrained baseline inverse temperature
  vector[C - 1] alpha_self_delta_z;   // unconstrained condition offsets for alpha_self (one per non-baseline condition)
  vector[C - 1] alpha_other_delta_z;  // unconstrained condition offsets for alpha_other
  vector[C - 1] beta_delta_z;         // unconstrained condition offsets for beta
}
transformed parameters {
  vector<lower=0,upper=1>[C] alpha_self;  // per-condition own-outcome learning rate in (0,1)
  vector<lower=0,upper=1>[C] alpha_other; // per-condition demonstrator-outcome learning rate in (0,1)
  vector<lower=0>[C] beta;               // per-condition inverse temperature > 0
  {
    int d = 0; // delta index counter (incremented for each non-baseline condition)
    for (c in 1:C) {
      real as_z = alpha_self_shared_z;  // start from the shared baseline
      real ao_z = alpha_other_shared_z;
      real bz = beta_shared_z;
      // Non-baseline conditions add their condition-specific delta
      if (c != baseline_cond) {
        d += 1;
        as_z += alpha_self_delta_z[d];
        ao_z += alpha_other_delta_z[d];
        bz += beta_delta_z[d];
      }
      alpha_self[c] = inv_logit(as_z);  // map to (0,1)
      alpha_other[c] = inv_logit(ao_z);
      beta[c] = log1p_exp(bz);          // map to (0,inf)
    }
  }
}
model {
  vector[A] Q = rep_vector(q_init, A); // action values, initialised to q_init

  target += prior_lpdf(alpha_self_shared_z | alpha_self_prior_family, alpha_self_prior_p1, alpha_self_prior_p2, alpha_self_prior_p3);
  target += prior_lpdf(alpha_other_shared_z | alpha_other_prior_family, alpha_other_prior_p1, alpha_other_prior_p2, alpha_other_prior_p3);
  target += prior_lpdf(beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  alpha_self_delta_z ~ normal(0, 1);  // regularising prior on condition deltas
  alpha_other_delta_z ~ normal(0, 1);
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
    // OWN-OUTCOME UPDATE: Rescorla-Wagner update from own reward, using condition-specific alpha_self
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      int cc = step_condition[e];
      Q[a] = Q[a] + alpha_self[cc] * (step_reward[e] - Q[a]); // condition-specific own learning rate
    }
    // SOCIAL UPDATE: Rescorla-Wagner update from demonstrator, using condition-specific alpha_other
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      int cc = step_condition[e];
      Q[sa] = Q[sa] + alpha_other[cc] * (step_social_reward[e] - Q[sa]); // condition-specific social learning rate
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
      // DECISION step: record per-trial log-likelihood with condition-specific beta
      if (step_choice[e] > 0) {
        d += 1;
        int cc = step_condition[e];
        vector[A] u = beta[cc] * Q;
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // OWN-OUTCOME UPDATE: Rescorla-Wagner update with condition-specific alpha_self
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        int cc = step_condition[e];
        Q[a] = Q[a] + alpha_self[cc] * (step_reward[e] - Q[a]);
      }
      // SOCIAL UPDATE: Rescorla-Wagner update with condition-specific alpha_other
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        int cc = step_condition[e];
        Q[sa] = Q[sa] + alpha_other[cc] * (step_social_reward[e] - Q[sa]);
      }
    }
  }
}
