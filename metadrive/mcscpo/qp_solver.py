import numpy as np
from pyomo.environ import (AbstractModel, Var, Objective, Constraint, SolverFactory,Param,
                           minimize, maximize, NonNegativeReals, RangeSet)
import unittest

class QuadraticOptimizer:
    def __init__(self, m):
        self.m = m
        self.model = AbstractModel()
        
        # Index set for nu
        self.model.I = RangeSet(self.m)
        
        # Define variables
        self.model.lambda_var = Var(domain=NonNegativeReals,initialize=0.5)
        self.model.nu = Var(self.model.I, domain=NonNegativeReals,initialize=0.001)

        # Define parameters
        self.model.C = Param(self.model.I, mutable=True)
        self.model.q = Param(mutable=True)
        self.model.r = Param(self.model.I, mutable=True)
        self.model.S = Param(self.model.I, self.model.I, mutable=True)
        self.model.delta = Param(mutable=True)

        
        # Store solutions
        self.optimal_lambda = None
        self.optimal_nu = None

    def solve(self, C, q, r, S, delta):
        # Set parameter values
        instance = self.model.create_instance()
        for i in instance.I:
            instance.C[i] = C[i-1]
            instance.r[i] = r[i-1]
            for j in instance.I:
                instance.S[i,j] = S[i-1][j-1]

        instance.q = q
        instance.delta = delta

        # Print out all the inputs and their shapes
        print("C:", C, "Shape:", np.shape(C))
        print("q:", q)
        print("r:", r, "Shape:", np.shape(r))
        print("S:", S, "Shape:", np.shape(S))
        print("delta:", delta)

        # Define the objective
        def objective_rule(mod):
            term1 = -1 / (2 * mod.lambda_var + 1e-8)
            sum_r_nu = sum(mod.r[i] * mod.nu[i] for i in mod.I)
            sum_S_nu_nu = sum(mod.S[i,j] * mod.nu[i] * mod.nu[j] for i in mod.I for j in mod.I)
            term2 = mod.q + 2 * sum_r_nu + sum_S_nu_nu
            term3 = sum(mod.C[i] * mod.nu[i] for i in mod.I)
            term4 = -mod.delta * mod.lambda_var / 2
            return term1 * term2 + term3 + term4
        instance.obj = Objective(rule=objective_rule, sense=maximize)

        # Solve the model
        solver = SolverFactory('ipopt')
        # solver.options["tol"] = 1e-4  # Set a tighter tolerance
        solver.options["max_iter"] = 10000  # Increase the maximum number of iterations
        results = solver.solve(instance, tee=True)

        # Check for infeasibility
        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            self.status = "Optimal"
        elif results.solver.termination_condition == 'infeasible':
            self.status = "Infeasible"
        else:
            self.status = "Solver Status: {}".format(results.solver.status)

        # Store solutions
        self.optimal_lambda = instance.lambda_var.value
        self.optimal_nu = np.array([instance.nu[i].value for i in instance.I])


    def get_solution(self):
        if self.status == "Infeasible":
            print("The optimization problem is infeasible!")
            return None, None, self.status
        return self.optimal_lambda, self.optimal_nu, self.status

class TestQuadraticOptimizer(unittest.TestCase):

    def test_solve(self):
        # 1. Initialize the QuadraticOptimizer with m=2
        m = 2
        optimizer = QuadraticOptimizer(m)

        # 2. Provide the provided input values
        C = np.array([0.33333334, 0.47140449])
        q = 0.10000436
        r = np.array([5.1337132, 7.596346])
        S = np.array([[1245.691, 715.49426], [715.4936, 2107.7969]])

        delta = 0.02

        # 3. Call the solve method
        optimizer.solve(C, q, r, S, delta)

        # 4. Get the solution
        optimal_lambda, optimal_nu, status = optimizer.get_solution()

        # 5. Print the results (for this example, we're just printing them, but in a real test, you'd have expected values to compare against)
        print("Optimal Lambda:", optimal_lambda)
        print("Optimal Nu:", optimal_nu)
        print("Status:", status)

        # Example: Assert that the status is "Optimal" (you can add more assertions based on expected results)
        self.assertEqual(status, "Optimal")

    def test_solve_with_new_values(self):
        # Given
        C = np.array([0.5, 0.5])
        q = 0.15585482
        r = np.array([1.3896, 0.2497])
        S = np.array([[919.6349, -69.7536], [-69.7538, 502.9334]])
        delta = 0.02
        m = len(C)

        # Initialize the QuadraticOptimizer
        optimizer = QuadraticOptimizer(m)

        # Call the solve method
        optimizer.solve(C, q, r, S, delta)

        # Get the solution
        optimal_lambda, optimal_nu, status = optimizer.get_solution()
        np.set_printoptions(precision=4, suppress=True)
        print("lambda and nu value from solver = [{},{}]".format(optimal_lambda,optimal_nu))
        # Assert that the status is "Optimal"
        self.assertEqual(status, "Optimal")

if __name__ == "__main__":
    unittest.main()