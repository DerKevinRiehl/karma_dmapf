# Karma Mechanisms for Decentralised, Cooperative Multi Agent Path Finding

## Introduction
This is the online repository of "Karma Mechanisms for Decentralised, Cooperative Multi Agent Path Finding". This repository contains a Python-implementation of a warehouse-like MAPD simulation, in which robots with kinematic constraints (orientation-awareness) navigate on a two-dimensional grid, and pickup and dropoff randomly generated tasks. 

<table>
<tr>
<td><img src="src/animations/animation_CENTRALIZED.gif" /></td>
<td><img src="src/animations/animation_DECENTRALIZED_RESPECT.gif" /></td>
</tr>
<tr>
<td><b><center>CBS (Centralised)</center></b></td>
<td><b><center>Token-Passing</center></b></td>
</tr>
<tr>
<td><img src="src/animations/animation_DECENTRALIZED_NEGOTIATE_EGOISTIC.gif" /></td>
<td><img src="src/animations/animation_DECENTRALIZED_NEGOTIATE_ALTRUISTIC.gif" /></td>
</tr>
<tr>
<td><b><center>Negotiation (Egoistic)</center></b></td>
<td><b><center>Negotiation (Altruistic)</center></b></td>
</tr>
</table>

## Abstract
Multi-Agent Path Finding (MAPF) is a fundamental coordination problem in large-scale robotic and cyber-physical systems, where multiple agents must compute conflict-free trajectories with limited computational and communication resources. 
While centralised optimal solvers provide guarantees on solution optimality, their exponential computational complexity limits scalability to large-scale systems and real-time applicability. 
Existing decentralised heuristics are faster, but lead to suboptimal outcomes and result in high disparities in costs.
This paper proposes a decentralised coordination framework for cooperative MAPF based on Karma mechanisms -- artificial, non-transferable credits that encode agents’ histories of cooperative behaviour and regulate future conflict resolution decisions. 
The approach formulates conflict resolution as a bilateral negotiation process that enables agents to resolve conflicts through pairwise replanning while promoting long-term fairness under limited communication and without global priority structures.
The mechanism is evaluated in a lifelong robotic warehouse multi-agent pickup-and-delivery scenario with kinematic orientation constraints. 
Unlike established, decentralised heuristics, the proposed Karma mechanism regulates the temporal distribution of replanning effort across agents, improving fairness while maintaining efficiency. 

## What you will find in this repository

## Installation & Run Instructions
```
pip install -r requirements.txt
```

## Log Files

## Citation
If you found this repository helpful, please cite our work:
```
Kevin Riehl, Julius Schlapbach, Anastasios Kouvelas, Michail A. Makridis
"Karma Mechanisms for Decentralised, Cooperative Multi Agent Path Finding", 2026.
Submitted to CDC2026: 65th IEEE Conference on Decision and Control, Honolulu, Hawaii.
```