/*
 * Model: Social RL — Demonstrator Mixture (Outcome + Action Tendency)
 * Hierarchy: Study-subject (multiple subjects, each drawn from a group-level distribution)
 * Parameters: alpha_other_outcome[N], alpha_other_action[N], w_imitation[N], beta[N]
 *             — per-subject, non-centredly parameterised
 *
 * Mixture social Q-learning with a two-level hierarchy: each subject maintains two independent
 * value systems (Q for outcome, T for action tendency) combined at decision time via w_imitation.
 * All four parameters are drawn from group-level normal distributions in unconstrained space
 * (non-centred parameterisation), then mapped to their natural ranges.
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
  array[E] int<lower=0,upper=A> step_update_action;  // action for own-outcome update; 0 if no self-update at this step
  vector[E] step_reward;                             // own reward received at the update step
  array[E] vector<lower=0,upper=1>[A] step_avail_mask; // binary mask of available actions (1 = available)
  array[E] int step_block;                           // block index for each step (used to detect block boundaries)
  int<lower=0,upper=1> reset_on_block;               // 1 = reset Q-values at each new block, 0 = carry over
  real q_init;                                       // initial Q-value assigned to every action

  array[E] int<lower=0,upper=A> step_social_action;  // demonstrator's action at this step; 0 if no social info
  vector[E] step_social_reward;                      // demonstrator's reward at this step (0 if no social info)

  int alpha_other_outcome_prior_family;   // prior family code for the group-level alpha_other_outcome mean
  real alpha_other_outcome_prior_p1;      // first hyperparameter of the alpha_other_outcome prior
  real alpha_other_outcome_prior_p2;      // second hyperparameter of the alpha_other_outcome prior
  real alpha_other_outcome_prior_p3;      // third hyperparameter of the alpha_other_outcome prior
  int sd_alpha_other_outcome_prior_family;   // prior family code for the group-level alpha_other_outcome SD
  real sd_alpha_other_outcome_prior_p1;      // first hyperparameter of the alpha_other_outcome SD prior
  real sd_alpha_other_outcome_prior_p2;      // second hyperparameter of the alpha_other_outcome SD prior
  real sd_alpha_other_outcome_prior_p3;      // third hyperparameter of the alpha_other_outcome SD prior
  int alpha_other_action_prior_family;    // prior family code for the group-level alpha_other_action mean
  real alpha_other_action_prior_p1;       // first hyperparameter of the alpha_other_action prior
  real alpha_other_action_prior_p2;       // second hyperparameter of the alpha_other_action prior
  real alpha_other_action_prior_p3;       // third hyperparameter of the alpha_other_action prior
  int sd_alpha_other_action_prior_family;    // prior family code for the group-level alpha_other_action SD
  real sd_alpha_other_action_prior_p1;       // first hyperparameter of the alpha_other_action SD prior
  real sd_alpha_other_action_prior_p2;       // second hyperparameter of the alpha_other_action SD prior
  real sd_alpha_other_action_prior_p3;       // third hyperparameter of the alpha_other_action SD prior
  int w_imitation_prior_family;           // prior family code for the group-level w_imitation mean
  real w_imitation_prior_p1;              // first hyperparameter of the w_imitation prior
  real w_imitation_prior_p2;              // second hyperparameter of the w_imitation prior
  real w_imitation_prior_p3;              // third hyperparameter of the w_imitation prior
  int sd_w_imitation_prior_family;           // prior family code for the group-level w_imitation SD
  real sd_w_imitation_prior_p1;              // first hyperparameter of the w_imitation SD prior
  real sd_w_imitation_prior_p2;              // second hyperparameter of the w_imitation SD prior
  real sd_w_imitation_prior_p3;              // third hyperparameter of the w_imitation SD prior
  int beta_prior_family;                  // prior family code for the group-level beta mean
  real beta_prior_p1;                     // first hyperparameter of the beta prior
  real beta_prior_p2;                     // second hyperparameter of the beta prior
  real beta_prior_p3;                     // third hyperparameter of the beta prior
  int sd_beta_prior_family;                  // prior family code for the group-level beta SD
  real sd_beta_prior_p1;                     // first hyperparameter of the beta SD prior
  real sd_beta_prior_p2;                     // second hyperparameter of the beta SD prior
  real sd_beta_prior_p3;                     // third hyperparameter of the beta SD prior
}
parameters {
  real mu_alpha_other_outcome_z;          // group-level mean of alpha_other_outcome in unconstrained space
  real<lower=0> sd_alpha_other_outcome_z; // group-level SD of alpha_other_outcome in unconstrained space
  vector[N] raw_alpha_other_outcome_z;    // per-subject standard normal deviates for alpha_other_outcome (non-centred)

  real mu_alpha_other_action_z;           // group-level mean of alpha_other_action in unconstrained space
  real<lower=0> sd_alpha_other_action_z;  // group-level SD of alpha_other_action in unconstrained space
  vector[N] raw_alpha_other_action_z;     // per-subject standard normal deviates for alpha_other_action (non-centred)

  real mu_w_imitation_z;                  // group-level mean of w_imitation in unconstrained space
  real<lower=0> sd_w_imitation_z;         // group-level SD of w_imitation in unconstrained space
  vector[N] raw_w_imitation_z;            // per-subject standard normal deviates for w_imitation (non-centred)

  real mu_beta_z;                         // group-level mean of beta in unconstrained space
  real<lower=0> sd_beta_z;               // group-level SD of beta in unconstrained space
  vector[N] raw_beta_z;                   // per-subject standard normal deviates for beta (non-centred)
}
transformed parameters {
  vector[N] alpha_other_outcome_z   = mu_alpha_other_outcome_z   + sd_alpha_other_outcome_z   * raw_alpha_other_outcome_z;
  vector[N] alpha_other_action_z    = mu_alpha_other_action_z    + sd_alpha_other_action_z    * raw_alpha_other_action_z;
  vector[N] w_imitation_z           = mu_w_imitation_z           + sd_w_imitation_z           * raw_w_imitation_z;
  vector[N] beta_z                  = mu_beta_z                  + sd_beta_z                  * raw_beta_z;

  vector<lower=0,upper=1>[N] alpha_other_outcome = inv_logit(alpha_other_outcome_z); // per-subject demonstrator-outcome learning rate in (0,1)
  vector<lower=0,upper=1>[N] alpha_other_action  = inv_logit(alpha_other_action_z);  // per-subject demonstrator-action learning rate in (0,1)
  vector<lower=0,upper=1>[N] w_imitation         = inv_logit(w_imitation_z);         // per-subject imitation weight in (0,1)
  vector<lower=0>[N]         beta                = log1p_exp(beta_z);                // per-subject inverse temperature > 0
}
model {
  array[N] vector[A] Q; // per-subject outcome-value vectors
  array[N] vector[A] T; // per-subject action-tendency vectors

  target += prior_lpdf(mu_alpha_other_outcome_z | alpha_other_outcome_prior_family, alpha_other_outcome_prior_p1, alpha_other_outcome_prior_p2, alpha_other_outcome_prior_p3);
  target += prior_lpdf(sd_alpha_other_outcome_z | sd_alpha_other_outcome_prior_family, sd_alpha_other_outcome_prior_p1, sd_alpha_other_outcome_prior_p2, sd_alpha_other_outcome_prior_p3);   // configurable prior on group SD (constrained positive)
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

  for (n in 1:N) Q[n] = rep_vector(q_init, A);      // initialise outcome-values for all subjects
  for (n in 1:N) T[n] = rep_vector(1.0 / A, A);     // initialise action tendencies to uniform

  for (e in 1:E) {
    int n = step_subject[e]; // subject for this step
    // Reset this subject's values when their block changes
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
      T[n] = rep_vector(1.0 / A, A);
    }
    // DECISION step: combine Q and T via w_imitation, then softmax
    if (step_choice[e] > 0) {
      vector[A] u = beta[n] * (w_imitation[n] * T[n] + (1 - w_imitation[n]) * Q[n]);
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // SOCIAL UPDATE: update Q from demonstrator outcome and T from demonstrator action
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      Q[n][sa] = Q[n][sa] + alpha_other_outcome[n] * (step_social_reward[e] - Q[n][sa]); // outcome update
      T[n][sa] = T[n][sa] + alpha_other_action[n] * (1 - T[n][sa]);                      // chosen action tendency toward 1
      for (a in 1:A) if (sa != a) T[n][a] = T[n][a] + alpha_other_action[n] * (0 - T[n][a]); // unchosen toward 0
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D); // per-decision log-likelihood for LOO-CV
  {
    array[N] vector[A] Q;
    array[N] vector[A] T;
    for (n in 1:N) Q[n] = rep_vector(q_init, A);
    for (n in 1:N) T[n] = rep_vector(1.0 / A, A);
    int d = 0; // decision counter (indexes into log_lik)

    for (e in 1:E) {
      int n = step_subject[e];
      // Reset this subject's values when their block changes
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        Q[n] = rep_vector(q_init, A);
        T[n] = rep_vector(1.0 / A, A);
      }
      // DECISION step: record per-trial log-likelihood
      if (step_choice[e] > 0) {
        d += 1;
        vector[A] u = beta[n] * (w_imitation[n] * T[n] + (1 - w_imitation[n]) * Q[n]);
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // SOCIAL UPDATE
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        Q[n][sa] = Q[n][sa] + alpha_other_outcome[n] * (step_social_reward[e] - Q[n][sa]);
        T[n][sa] = T[n][sa] + alpha_other_action[n] * (1 - T[n][sa]);
        for (a in 1:A) if (sa != a) T[n][a] = T[n][a] + alpha_other_action[n] * (0 - T[n][a]);
      }
    }
  }
  real alpha_other_outcome_pop = inv_logit(mu_alpha_other_outcome_z); // group-level alpha_other_outcome (population mean, constrained)
  real alpha_other_action_pop  = inv_logit(mu_alpha_other_action_z);  // group-level alpha_other_action (population mean, constrained)
  real w_imitation_pop         = inv_logit(mu_w_imitation_z);         // group-level w_imitation (population mean, constrained)
  real beta_pop                = log1p_exp(mu_beta_z);                // group-level beta (population mean, constrained)
}
