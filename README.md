# 2D Constraint Solver

Eigen-based damped Gauss-Newton geometry solver with point coordinates, distance/horizontal/parallel/perpendicular constraints, analytic Jacobians, rank diagnostics, and SVG output.

## Run

Requires Eigen 3. `cmake -S . -B build && cmake --build build && ctest --test-dir build`. Run build/solver for an SVG rectangle and rank/residual diagnostics.
