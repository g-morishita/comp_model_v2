/*
 * Model: Social RL — Demonstrator Mixture (Outcome + Action Tendency)
 * Hierarchy: Subject-shared (single subject, one parameter set for all conditions)
 * Parameters: alpha_other_outcome, alpha_other_action, w_imitation, beta
 *
 * Mixture social Q-learning for a single subject that learns from a demonstrator's
 * outcomes and actions. Two independent value systems are maintained:
 * Q (outcome values) updated by demonstrator rewards, and T (action tendencies)
 * updated by demonstrator action frequency. At decision time Q and T are combined via
 * w_imitation before the softmax.
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
  int<lower=0,upper=1> reset_on_block;               // 1 = reset Q/T values at each new block, 0 = carry over
  real q_init;                                       // initial Q-value assigned to every action

  array[E] int<lower=0,upper=A> step_social_action;  // demonstrator's action at this step; 0 if no social info
  vector[E] step_social_reward;                      // demonstrator's reward at this step (0 if no social info)

  int alpha_other_outcome_prior_family;   // prior family code for alpha_other_outcome
  real alpha_other_outcome_prior_p1;      // first hyperparameter of the alpha_other_outcome prior
  real alpha_other_outcome_prior_p2;      // second hyperparameter of the alpha_other_outcome prior
  real alpha_other_outcome_prior_p3;      // third hyperparameter of the alpha_other_outcome prior
  int alpha_other_action_prior_family;    // prior family code for alpha_other_action
  real alpha_other_action_prior_p1;       // first hyperparameter of the alpha_other_action prior
  real alpha_other_action_prior_p2;       // second hyperparameter of the alpha_other_action prior
  real alpha_other_action_prior_p3;       // third hyperparameter of the alpha_other_action prior
  int w_imitation_prior_family;           // prior family code for w_imitation
  real w_imitation_prior_p1;              // first hyperparameter of the w_imitation prior
  real w_imitation_prior_p2;              // second hyperparameter of the w_imitation prior
  real w_imitation_prior_p3;              // third hyperparameter of the w_imitation prior
  int beta_prior_family;                  // prior family code for beta
  real beta_prior_p1;                     // first hyperparameter of the beta prior
  real beta_prior_p2;                     // second hyperparameter of the beta prior
  real beta_prior_p3;                     // third hyperparameter of the beta prior
}
parameters {
  real alpha_other_outcome_z; // unconstrained demonstrator-outcome learning rate
  real alpha_other_action_z;  // unconstrained demonstrator-action learning rate
  real w_imitation_z;         // unconstrained imitation mixing weight
  real beta_z;                // unconstrained inverse temperature
}
transformed parameters {
  real<lower=0,upper=1> alpha_other_outcome = inv_logit(alpha_other_outcome_z); // demonstrator-outcome learning rate in (0,1)
  real<lower=0,upper=1> alpha_other_action  = inv_logit(alpha_other_action_z);  // demonstrator-action learning rate in (0,1)
  real<lower=0,upper=1> w_imitation         = inv_logit(w_imitation_z);         // imitation weight in (0,1)
  real<lower=0>         beta                = log1p_exp(beta_z);                // inverse temperature > 0 (softplus)
}
model {
  vector[A] Q = rep_vector(q_init, A);      // outcome values, initialised to q_init
  vector[A] T = rep_vector(1.0 / A, A);     // action tendencies, initialised to uniform

  target += prior_lpdf(alpha_other_outcome_z | alpha_other_outcome_prior_family, alpha_other_outcome_prior_p1, alpha_other_outcome_prior_p2, alpha_other_outcome_prior_p3);
  target += prior_lpdf(alpha_other_action_z | alpha_other_action_prior_family, alpha_other_action_prior_p1, alpha_other_action_prior_p2, alpha_other_action_prior_p3);
  target += prior_lpdf(w_imitation_z | w_imitation_prior_family, w_imitation_prior_p1, w_imitation_prior_p2, w_imitation_prior_p3);
  target += prior_lpdf(beta_z | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);

  for (e in 1:E) {
    // Reset Q and T when a new block starts (if reset_on_block is enabled)
    if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
      Q = rep_vector(q_init, A);
      T = rep_vector(1.0 / A, A);
    }
    // DECISION step: combine Q and T via w_imitation, then softmax
    if (step_choice[e] > 0) {
      vector[A] u = beta * (w_imitation * T + (1 - w_imitation) * Q); // weighted mixture of value systems
      for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity(); // mask unavailable actions
      target += categorical_logit_lpmf(step_choice[e] | u);
    }
    // SOCIAL UPDATE: update Q from demonstrator outcome and T from demonstrator action
    if (step_social_action[e] > 0) {
      int sa = step_social_action[e];
      Q[sa] = Q[sa] + alpha_other_outcome * (step_social_reward[e] - Q[sa]);          // outcome update
      T = (1 - alpha_other_action) * T;
      T[sa] = T[sa] + alpha_other_action;
    }
  }
}
generated quantities {
  vector[D] log_lik = rep_vector(0.0, D); // per-decision log-likelihood for LOO-CV
  {
    vector[A] Q = rep_vector(q_init, A); // local copy of Q-values for this forward pass
    vector[A] T = rep_vector(1.0 / A, A); // local copy of action tendencies
    int d = 0;                            // decision counter (indexes into log_lik)

    for (e in 1:E) {
      // Reset Q and T when a new block starts (if reset_on_block is enabled)
      if (reset_on_block == 1 && e > 1 && step_block[e] != step_block[e - 1]) {
        Q = rep_vector(q_init, A);
        T = rep_vector(1.0 / A, A);
      }
      // DECISION step: record per-trial log-likelihood
      if (step_choice[e] > 0) {
        d += 1;
        vector[A] u = beta * (w_imitation * T + (1 - w_imitation) * Q);
        for (a in 1:A) if (step_avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik[d] = categorical_logit_lpmf(step_choice[e] | u);
      }
      // SOCIAL UPDATE
      if (step_social_action[e] > 0) {
        int sa = step_social_action[e];
        Q[sa] = Q[sa] + alpha_other_outcome * (step_social_reward[e] - Q[sa]);
        T = (1 - alpha_other_action) * T;
        T[sa] = T[sa] + alpha_other_action;
      }
    }
  }
}
