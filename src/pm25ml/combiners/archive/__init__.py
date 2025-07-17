"""
Combines data from multiple raw archive storage into a single file.

It has:
- a planner that defines what data needs to be combined
- a manager that orchestrates the combining process, merging only those months that need it
and skipping those that have already been combined
- a combiner that performs the actual combining of data
"""
