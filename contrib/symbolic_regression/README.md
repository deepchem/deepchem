to understand the code 
start from the EquationSearch.py file and go on for each function

# File Descriptions

## Candidate.py
Defines the Candidate class, representing a single symbolic expression (tree) in the population, including its loss, cost, complexity, and birth time.

## CheckConstrains.py
Handles validation and constraint checking for generated trees (e.g., maximum depth, allowed operators, and structural validity).

## Complexity.py
Computes expression complexity metrics used for parsimony pressure to penalize overly complex equations.

## Dataset.py
Provides dataset description.

## EquationSearch.py
Main search controller. Orchestrates the symbolic regression process including initialization, evolutionary cycles, evaluation, and selection.

## Generation.py
Responsible for generating initial random expression trees and producing new individuals during evolution.

## Mutation.py
Implements mutation operators applied to trees (structural mutations, constant perturbations, operator substitutions, etc.).

## Options.py
Central configuration module containing hyperparameters and settings such as population size, operator sets, mutation rates, and annealing parameters.

## Population.py
Manages the population of candidates: storage, ranking, selection, replacement, and population-level operations.

## RegEvolCycle.py
Implements the regularized evolutionary cycle (evolve–simplify–optimize loop), coordinating mutation, crossover, evaluation, and survivor selection.

## RepositoryOfOperations.py
Defines the available unary and binary operators (e.g., +, *, sin, cos) along with their PyTorch implementations.

## Tree.py
Defines the expression tree data structure, including nodes, traversal utilities, evaluation logic, and conversion to symbolic equations.

## Utils.py
Collection of helper utilities (tree helpers, simplification routines, logging, formatting, etc.).

# example.py
Minimal runnable example demonstrating how to use the framework on a toy dataset.
