/*
 * Model: Social RL — Self Reward + Demonstrator Mixture (Outcome + Action Tendency)
 * Hierarchy: Study-subject-block-condition (multiple subjects, per-subject per-condition parameters)
 * Parameters: alpha_self[N][C], alpha_other_outcome[N][C], alpha_other_action[N][C],
 *             w_imitation[N][C], beta[N][C] — per-subject per-condition, via shared + delta hierarchies
 *
 * Two-level hierarchical mixture social Q-learning. Each subject maintains two independent value
 * systems (Q for outcome, T for action tendency) combined at decision time via w_imitation.
 * All five parameters are built from a subject-level baseline plus a condition delta, both drawn
 * from their own group distributions (non-centred parameterisation).
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

  int<lower=2> C;                        // number of experimental conditions
  int<lower=1,upper=C> baseline_cond;    // index of the baseline condition (receives no delta)
  array[E] int<lower=1,upper=C> step_condition; // condition index for each step

  int alpha_self_prior_family;            // prior family code for the group-level shared alpha_self mean
  real alpha_self_prior_p1;               // first hyperparameter of the alpha_self prior
  real alpha_self_prior_p2;               // second hyperparameter of the alpha_self prior
  real alpha_self_prior_p3;               // third hyperparameter of the alpha_self prior
  int alpha_other_outcome_prior_family;   // prior family code for the group-level shared alpha_other_outcome mean
  real alpha_other_outcome_prior_p1;      // first hyperparameter of the alpha_other_outcome prior
  real alpha_other_outcome_prior_p2;      // second hyperparameter of the alpha_other_outcome prior
  real alpha_other_outcome_prior_p3;      // third hyperparameter of the alpha_other_outcome prior
  int alpha_other_action_prior_family;    // prior family code for the group-level shared alpha_other_action mean
  real alpha_other_action_prior_p1;       // first hyperparameter of the alpha_other_action prior
  real alpha_other_action_prior_p2;       // second hyperparameter of the alpha_other_action prior
  real alpha_other_action_prior_p3;       // third hyperparameter of the alpha_other_action prior
  int w_imitation_prior_family;           // prior family code for the group-level shared w_imitation mean
  real w_imitation_prior_p1;              // first hyperparameter of the w_imitation prior
  real w_imitation_prior_p2;              // second hyperparameter of the w_imitation prior
  real w_imitation_prior_p3;              // third hyperparameter of the w_imitation prior
  int beta_prior_family;                  // prior family code for the group-level shared beta mean
  real beta_prior_p1;                     // first hyperparameter of the beta prior
  real beta_prior_p2;                     // second hyperparameter of the beta prior
  real beta_prior_p3;                     // third hyperparameter of the beta prior
  int sd_alpha_self_prior_family;            // prior family code for the group-level shared alpha_self SD
  real sd_alpha_self_prior_p1;               // first hyperparameter of the shared alpha_self SD prior
  real sd_alpha_self_prior_p2;               // second hyperparameter of the shared alpha_self SD prior
  real sd_alpha_self_prior_p3;               // third hyperparameter of the shared alpha_self SD prior
  int sd_alpha_other_outcome_prior_family;   // prior family code for the group-level shared alpha_other_outcome SD
  real sd_alpha_other_outcome_prior_p1;      // first hyperparameter of the shared alpha_other_outcome SD prior
  real sd_alpha_other_outcome_prior_p2;      // second hyperparameter of the shared alpha_other_outcome SD prior
  real sd_alpha_other_outcome_prior_p3;      // third hyperparameter of the shared alpha_other_outcome SD prior
  int sd_alpha_other_action_prior_family;    // prior family code for the group-level shared alpha_other_action SD
  real sd_alpha_other_action_prior_p1;       // first hyperparameter of the shared alpha_other_action SD prior
  real sd_alpha_other_action_prior_p2;       // second hyperparameter of the shared alpha_other_action SD prior
  real sd_alpha_other_action_prior_p3;       // third hyperparameter of the shared alpha_other_action SD prior
  int sd_w_imitation_prior_family;           // prior family code for the group-level shared w_imitation SD
  real sd_w_imitation_prior_p1;              // first hyperparameter of the shared w_imitation SD prior
  real sd_w_imitation_prior_p2;              // second hyperparameter of the shared w_imitation SD prior
  real sd_w_imitation_prior_p3;              // third hyperparameter of the shared w_imitation SD prior
  int sd_beta_prior_family;                  // prior family code for the group-level shared beta SD
  real sd_beta_prior_p1;                     // first hyperparameter of the shared beta SD prior
  real sd_beta_prior_p2;                     // second hyperparameter of the shared beta SD prior
  real sd_beta_prior_p3;                     // third hyperparameter of the shared beta SD prior
  int sd_alpha_self_delta_prior_family;            // prior family code for the group-level alpha_self delta SD
  real sd_alpha_self_delta_prior_p1;               // first hyperparameter of the alpha_self delta SD prior
  real sd_alpha_self_delta_prior_p2;               // second hyperparameter of the alpha_self delta SD prior
  real sd_alpha_self_delta_prior_p3;               // third hyperparameter of the alpha_self delta SD prior
  int sd_alpha_other_outcome_delta_prior_family;   // prior family code for the group-level alpha_other_outcome delta SD
  real sd_alpha_other_outcome_delta_prior_p1;      // first hyperparameter of the alpha_other_outcome delta SD prior
  real sd_alpha_other_outcome_delta_prior_p2;      // second hyperparameter of the alpha_other_outcome delta SD prior
  real sd_alpha_other_outcome_delta_prior_p3;      // third hyperparameter of the alpha_other_outcome delta SD prior
  int sd_alpha_other_action_delta_prior_family;    // prior family code for the group-level alpha_other_action delta SD
  real sd_alpha_other_action_delta_prior_p1;       // first hyperparameter of the alpha_other_action delta SD prior
  real sd_alpha_other_action_delta_prior_p2;       // second hyperparameter of the alpha_other_action delta SD prior
  real sd_alpha_other_action_delta_prior_p3;       // third hyperparameter of the alpha_other_action delta SD prior
  int sd_w_imitation_delta_prior_family;           // prior family code for the group-level w_imitation delta SD
  real sd_w_imitation_delta_prior_p1;              // first hyperparameter of the w_imitation delta SD prior
  real sd_w_imitation_delta_prior_p2;              // second hyperparameter of the w_imitation delta SD prior
  real sd_w_imitation_delta_prior_p3;              // third hyperparameter of the w_imitation delta SD prior
  int sd_beta_delta_prior_family;                  // prior family code for the group-level beta delta SD
  real sd_beta_delta_prior_p1;                     // first hyperparameter of the beta delta SD prior
  real sd_beta_delta_prior_p2;                     // second hyperparameter of the beta delta SD prior
  real sd_beta_delta_prior_p3;                     // third hyperparameter of the beta delta SD prior
}
parameters {
  // Population-level: shared (baseline)
  real mu_alpha_self_shared_z;                    // group mean of baseline alpha_self (unconstrained)
  real<lower=0> sd_alpha_self_shared_z;           // group SD of baseline alpha_self (unconstrained)
  real mu_alpha_other_outcome_shared_z;           // group mean of baseline alpha_other_outcome (unconstrained)
  real<lower=0> sd_alpha_other_outcome_shared_z;  // group SD of baseline alpha_other_outcome (unconstrained)
  real mu_alpha_other_action_shared_z;            // group mean of baseline alpha_other_action (unconstrained)
  real<lower=0> sd_alpha_other_action_shared_z;   // group SD of baseline alpha_other_action (unconstrained)
  real mu_w_imitation_shared_z;                   // group mean of baseline w_imitation (unconstrained)
  real<lower=0> sd_w_imitation_shared_z;          // group SD of baseline w_imitation (unconstrained)
  real mu_beta_shared_z;                          // group mean of baseline beta (unconstrained)
  real<lower=0> sd_beta_shared_z;                 // group SD of baseline beta (unconstrained)

  // Population-level: condition deltas
  vector[C - 1] mu_alpha_self_delta_z;                   // group means of per-condition alpha_self deltas (unconstrained)
  vector<lower=0>[C - 1] sd_alpha_self_delta_z;          // group SDs of per-condition alpha_self deltas
  vector[C - 1] mu_alpha_other_outcome_delta_z;          // group means of per-condition alpha_other_outcome deltas (unconstrained)
  vector<lower=0>[C - 1] sd_alpha_other_outcome_delta_z; // group SDs of per-condition alpha_other_outcome deltas
  vector[C - 1] mu_alpha_other_action_delta_z;           // group means of per-condition alpha_other_action deltas (unconstrained)
  vector<lower=0>[C - 1] sd_alpha_other_action_delta_z;  // group SDs of per-condition alpha_other_action deltas
  vector[C - 1] mu_w_imitation_delta_z;                  // group means of per-condition w_imitation deltas (unconstrained)
  vector<lower=0>[C - 1] sd_w_imitation_delta_z;         // group SDs of per-condition w_imitation deltas
  vector[C - 1] mu_beta_delta_z;                         // group means of per-condition beta deltas (unconstrained)
  vector<lower=0>[C - 1] sd_beta_delta_z;                // group SDs of per-condition beta deltas

  // Per-subject raw deviates (non-centred)
  vector[N] raw_alpha_self_shared_z;                   // per-subject deviates for baseline alpha_self
  vector[N] raw_alpha_other_outcome_shared_z;          // per-subject deviates for baseline alpha_other_outcome
  vector[N] raw_alpha_other_action_shared_z;           // per-subject deviates for baseline alpha_other_action
  vector[N] raw_w_imitation_shared_z;                  // per-subject deviates for baseline w_imitation
  vector[N] raw_beta_shared_z;                         // per-subject deviates for baseline beta
  array[C - 1] vector[N] raw_alpha_self_delta_z;          // per-subject deviates for alpha_self condition deltas
  array[C - 1] vector[N] raw_alpha_other_outcome_delta_z; // per-subject deviates for alpha_other_outcome condition deltas
  array[C - 1] vector[N] raw_alpha_other_action_delta_z;  // per-subject deviates for alpha_other_action condition deltas
  array[C - 1] vector[N] raw_w_imitation_delta_z;         // per-subject deviates for w_imitation condition deltas
  array[C - 1] vector[N] raw_beta_delta_z;                // per-subject deviates for beta condition deltas
}
transformed parameters {
  // Per-subject unconstrained shared (baseline)
  vector[N] alpha_self_shared_z           = mu_alpha_self_shared_z           + sd_alpha_self_shared_z           * raw_alpha_self_shared_z;
  vector[N] alpha_other_outcome_shared_z  = mu_alpha_other_outcome_shared_z  + sd_alpha_other_outcome_shared_z  * raw_alpha_other_outcome_shared_z;
  vector[N] alpha_other_action_shared_z   = mu_alpha_other_action_shared_z   + sd_alpha_other_action_shared_z   * raw_alpha_other_action_shared_z;
  vector[N] w_imitation_shared_z          = mu_w_imitation_shared_z          + sd_w_imitation_shared_z          * raw_w_imitation_shared_z;
  vector[N] beta_shared_z                 = mu_beta_shared_z                 + sd_beta_shared_z                 * raw_beta_shared_z;

  // Per-subject unconstrained condition deltas
  array[C - 1] vector[N] alpha_self_delta_z;
  array[C - 1] vector[N] alpha_other_outcome_delta_z;
  array[C - 1] vector[N] alpha_other_action_delta_z;
  array[C - 1] vector[N] w_imitation_delta_z;
  array[C - 1] vector[N] beta_delta_z;
  for (d in 1:(C - 1)) {
    alpha_self_delta_z[d]          = mu_alpha_self_delta_z[d]          + sd_alpha_self_delta_z[d]          * raw_alpha_self_delta_z[d];
    alpha_other_outcome_delta_z[d] = mu_alpha_other_outcome_delta_z[d] + sd_alpha_other_outcome_delta_z[d] * raw_alpha_other_outcome_delta_z[d];
    alpha_other_action_delta_z[d]  = mu_alpha_other_action_delta_z[d]  + sd_alpha_other_action_delta_z[d]  * raw_alpha_other_action_delta_z[d];
    w_imitation_delta_z[d]         = mu_w_imitation_delta_z[d]         + sd_w_imitation_delta_z[d]         * raw_w_imitation_delta_z[d];
    beta_delta_z[d]                = mu_beta_delta_z[d]                + sd_beta_delta_z[d]                * raw_beta_delta_z[d];
  }

  // Per-subject, per-condition constrained parameters
  array[N] vector<lower=0,upper=1>[C] alpha_self;          // per-subject per-condition own-outcome learning rate in (0,1)
  array[N] vector<lower=0,upper=1>[C] alpha_other_outcome; // per-subject per-condition demonstrator-outcome learning rate in (0,1)
  array[N] vector<lower=0,upper=1>[C] alpha_other_action;  // per-subject per-condition demonstrator-action learning rate in (0,1)
  array[N] vector<lower=0,upper=1>[C] w_imitation;         // per-subject per-condition imitation weight in (0,1)
  array[N] vector<lower=0>[C]         beta;                // per-subject per-condition inverse temperature > 0
  for (n in 1:N) {
    int d = 0; // delta index counter
    for (c in 1:C) {
      real as_z  = alpha_self_shared_z[n];
      real aoo_z = alpha_other_outcome_shared_z[n];
      real aoa_z = alpha_other_action_shared_z[n];
      real wi_z  = w_imitation_shared_z[n];
      real bz    = beta_shared_z[n];
      // Non-baseline conditions add the subject's condition-specific delta
      if (c != baseline_cond) {
        d += 1;
        as_z  += alpha_self_delta_z[d][n];
        aoo_z += alpha_other_outcome_delta_z[d][n];
        aoa_z += alpha_other_action_delta_z[d][n];
        wi_z  += w_imitation_delta_z[d][n];
        bz    += beta_delta_z[d][n];
      }
      alpha_self[n][c]          = inv_logit(as_z);   // map to (0,1)
      alpha_other_outcome[n][c] = inv_logit(aoo_z);
      alpha_other_action[n][c]  = inv_logit(aoa_z);
      w_imitation[n][c]         = inv_logit(wi_z);
      beta[n][c]                = log1p_exp(bz);     // map to (0,inf)
    }
  }
}
model {
  array[N] vector[A] Q; // per-subject outcome-value vectors
  array[N] vector[A] T; // per-subject action-tendency vectors

  // Priors: shared (baseline)
  target += prior_lpdf(mu_alpha_self_shared_z | alpha_self_prior_family, alpha_self_prior_p1, alpha_self_prior_p2, alpha_self_prior_p3);
  target += prior_lpdf(sd_alpha_self_shared_z | sd_alpha_self_prior_family, sd_alpha_self_prior_p1, sd_alpha_self_prior_p2, sd_alpha_self_prior_p3);
  raw_alpha_self_shared_z ~ normal(0, 1);  // non-centred parameterisation

  target += prior_lpdf(mu_alpha_other_outcome_shared_z | alpha_other_outcome_prior_family, alpha_other_outcome_prior_p1, alpha_other_outcome_prior_p2, alpha_other_outcome_prior_p3);
  target += prior_lpdf(sd_alpha_other_outcome_shared_z | sd_alpha_other_outcome_prior_family, sd_alpha_other_outcome_prior_p1, sd_alpha_other_outcome_prior_p2, sd_alpha_other_outcome_prior_p3);
  raw_alpha_other_outcome_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_alpha_other_action_shared_z | alpha_other_action_prior_family, alpha_other_action_prior_p1, alpha_other_action_prior_p2, alpha_other_action_prior_p3);
  target += prior_lpdf(sd_alpha_other_action_shared_z | sd_alpha_other_action_prior_family, sd_alpha_other_action_prior_p1, sd_alpha_other_action_prior_p2, sd_alpha_other_action_prior_p3);
  raw_alpha_other_action_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_w_imitation_shared_z | w_imitation_prior_family, w_imitation_prior_p1, w_imitation_prior_p2, w_imitation_prior_p3);
  target += prior_lpdf(sd_w_imitation_shared_z | sd_w_imitation_prior_family, sd_w_imitation_prior_p1, sd_w_imitation_prior_p2, sd_w_imitation_prior_p3);
  raw_w_imitation_shared_z ~ normal(0, 1);

  target += prior_lpdf(mu_beta_shared_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(sd_beta_shared_z | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);
  raw_beta_shared_z ~ normal(0, 1);

  // Priors: condition deltas
  mu_alpha_self_delta_z ~ normal(0, 1);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_alpha_self_delta_z[d] | sd_alpha_self_delta_prior_family, sd_alpha_self_delta_prior_p1, sd_alpha_self_delta_prior_p2, sd_alpha_self_delta_prior_p3);
  mu_alpha_other_outcome_delta_z ~ normal(0, 1);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_alpha_other_outcome_delta_z[d] | sd_alpha_other_outcome_delta_prior_family, sd_alpha_other_outcome_delta_prior_p1, sd_alpha_other_outcome_delta_prior_p2, sd_alpha_other_outcome_delta_prior_p3);
  mu_alpha_other_action_delta_z ~ normal(0, 1);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_alpha_other_action_delta_z[d] | sd_alpha_other_action_delta_prior_family, sd_alpha_other_action_delta_prior_p1, sd_alpha_other_action_delta_prior_p2, sd_alpha_other_action_delta_prior_p3);
  mu_w_imitation_delta_z ~ normal(0, 1);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_w_imitation_delta_z[d] | sd_w_imitation_delta_prior_family, sd_w_imitation_delta_prior_p1, sd_w_imitation_delta_prior_p2, sd_w_imitation_delta_prior_p3);
  mu_beta_delta_z ~ normal(0, 1);
  for (d in 1:(C - 1))
    target += prior_lpdf(sd_beta_delta_z[d] | sd_beta_delta_prior_family, sd_beta_delta_prior_p1, sd_beta_delta_prior_p2, sd_beta_delta_prior_p3);
  for (d in 1:(C - 1)) {
    raw_alpha_self_delta_z[d] ~ normal(0, 1);          // non-centred deviates for per-subject alpha_self deltas
    raw_alpha_other_outcome_delta_z[d] ~ normal(0, 1);
    raw_alpha_other_action_delta_z[d] ~ normal(0, 1);
    raw_w_imitation_delta_z[d] ~ normal(0, 1);
    raw_beta_delta_z[d] ~ normal(0, 1);
  }

  for (n in 1:N) Q[n] = rep_vector(q_init, A);     // initialise outcome-values for all subjects
  for (n in 1:N) T[n] = rep_vector(1.0 / A, A);    // initialise action tendencies to uniform

  for (e in 1:E) {
    int n = step_subject[e];   // subject for this step
    int cc = step_condition[e]; // condition for this step
    // Reset this subject's values when their block changes
    if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
        step_block[e] != step_block[e - 1]) {
      Q[n] = rep_vector(q_init, A);
      T[n] = rep_vector(1.0 / A, A);
    }
    // DECISION step: combine Q and T via subject's condition-specific w_imitation, then softmax
    if (step_choice[e] > 0) {
      vector[A] u = beta[n][cc] * (w_imitation[n][cc] * T[n] + (1 - w_imitation[n][cc]) * Q[n]);
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // OWN-OUTCOME UPDATE: Rescorla-Wagner update on Q with subject n's condition-specific alpha_self
    if (step_update_action[e] > 0) {
      int a = step_update_action[e];
      Q[n][a] = Q[n][a] + alpha_self[n][cc] * (step_reward[e] - Q[n][a]);
    }
    // SOCIAL UPDATE: update Q from demonstrator outcome and T from demonstrator action
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      Q[n][sa] = Q[n][sa] + alpha_other_outcome[n][cc] * (step_social_reward[e] - Q[n][sa]); // outcome update
      T[n][sa] = T[n][sa] + alpha_other_action[n][cc] * (1 - T[n][sa]);                       // chosen action tendency toward 1
      for (a in 1:A) if (sa != a) T[n][a] = T[n][a] + alpha_other_action[n][cc] * (0 - T[n][a]); // unchosen toward 0
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D); // per-decision log-likelihood for LOO-CV
  vector<lower=0,upper=1>[C] alpha_self_pop;          // group-mean alpha_self for every condition
  vector<lower=0,upper=1>[C] alpha_other_outcome_pop; // group-mean alpha_other_outcome for every condition
  vector<lower=0,upper=1>[C] alpha_other_action_pop;  // group-mean alpha_other_action for every condition
  vector<lower=0,upper=1>[C] w_imitation_pop;         // group-mean w_imitation for every condition
  vector<lower=0>[C] beta_pop;                        // group-mean beta for every condition
  real alpha_self_shared_pop;                         // group-mean baseline alpha_self
  real alpha_other_outcome_shared_pop;                // group-mean baseline alpha_other_outcome
  real alpha_other_action_shared_pop;                 // group-mean baseline alpha_other_action
  real w_imitation_shared_pop;                        // group-mean baseline w_imitation
  real beta_shared_pop;                               // group-mean baseline beta
  {
    array[N] vector[A] Q;
    array[N] vector[A] T;
    for (n in 1:N) Q[n] = rep_vector(q_init, A);
    for (n in 1:N) T[n] = rep_vector(1.0 / A, A);
    int d = 0; // decision counter (indexes into log_lik)

    for (e in 1:E) {
      int n = step_subject[e];
      int cc = step_condition[e];
      // Reset this subject's values when their block changes
      if (reset_on_block == 1 && e > 1 && step_subject[e] == step_subject[e - 1] &&
          step_block[e] != step_block[e - 1]) {
        Q[n] = rep_vector(q_init, A);
        T[n] = rep_vector(1.0 / A, A);
      }
      // DECISION step: record per-trial log-likelihood
      if (step_choice[e] > 0) {
        d += 1;
        vector[A] u = beta[n][cc] * (w_imitation[n][cc] * T[n] + (1 - w_imitation[n][cc]) * Q[n]);
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // OWN-OUTCOME UPDATE
      if (step_update_action[e] > 0) {
        int a = step_update_action[e];
        Q[n][a] = Q[n][a] + alpha_self[n][cc] * (step_reward[e] - Q[n][a]);
      }
      // SOCIAL UPDATE
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        Q[n][sa] = Q[n][sa] + alpha_other_outcome[n][cc] * (step_social_reward[e] - Q[n][sa]);
        T[n][sa] = T[n][sa] + alpha_other_action[n][cc] * (1 - T[n][sa]);
        for (a in 1:A) if (sa != a) T[n][a] = T[n][a] + alpha_other_action[n][cc] * (0 - T[n][a]);
      }
    }
  }

  {
    int d_idx = 0;
    for (c in 1:C) {
      real asz = mu_alpha_self_shared_z;
      real aoz = mu_alpha_other_outcome_shared_z;
      real aaz = mu_alpha_other_action_shared_z;
      real wiz = mu_w_imitation_shared_z;
      real bz = mu_beta_shared_z;
      if (c != baseline_cond) {
        d_idx += 1;
        asz += mu_alpha_self_delta_z[d_idx];
        aoz += mu_alpha_other_outcome_delta_z[d_idx];
        aaz += mu_alpha_other_action_delta_z[d_idx];
        wiz += mu_w_imitation_delta_z[d_idx];
        bz += mu_beta_delta_z[d_idx];
      }
      alpha_self_pop[c] = inv_logit(asz);
      alpha_other_outcome_pop[c] = inv_logit(aoz);
      alpha_other_action_pop[c] = inv_logit(aaz);
      w_imitation_pop[c] = inv_logit(wiz);
      beta_pop[c] = log1p_exp(bz);
    }
  }

  // Population-level constrained parameters (baseline condition)
  alpha_self_shared_pop          = alpha_self_pop[baseline_cond];
  alpha_other_outcome_shared_pop = alpha_other_outcome_pop[baseline_cond];
  alpha_other_action_shared_pop  = alpha_other_action_pop[baseline_cond];
  w_imitation_shared_pop         = w_imitation_pop[baseline_cond];
  beta_shared_pop                = beta_pop[baseline_cond];
}
