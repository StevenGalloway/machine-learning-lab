# Problem Statement

## Objective
Estimate the probability that one NFL team would beat another in a neutral-site Super Bowl-style matchup using historical play-by-play-derived performance signals.

## Business framing
For a recruiter-facing public portfolio, the goal is to demonstrate an end-to-end predictive workflow that turns raw event data into calibrated matchup probabilities and enterprise-style documentation.

## Prediction target
Binary outcome: whether the modeled home orientation wins. Neutral-site probability is then estimated by averaging both home/away orientations.

## Why this is not a Geometric Brownian Motion use case
Geometric Brownian Motion models continuous stochastic paths such as financial asset prices. NFL games are discrete competitions with engineered matchup features, so a supervised classifier is the appropriate choice here.
