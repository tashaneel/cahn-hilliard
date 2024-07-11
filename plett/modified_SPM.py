import pybamm

class modified_SPM(pybamm.lithium_ion.BaseModel):
    """
    Modified Single Particle Model (SPM) of a lithium-ion battery.

    """
    def __init__(self, name="Modified Single Particle Model"):
        super().__init__({}, name)
        self.summary_variables = []
        pybamm.citations.register("Plett2015")

        ######################
        # Variables
        ######################

        z = pybamm.Variable("State of charge")
        V = pybamm.Variable("Terminal voltage [V]")
        h = pybamm.Variable("Dynamic hysteresis voltage [V]")
        i_R1 = pybamm.Variable("Current through resistor R_1 [A]")
        c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")

        ######################
        # Parameters
        ######################

        Q = pybamm.Parameter("Total charge capacity [C]")
        R_0 = pybamm.Parameter("Equivalent series resistance in R_0 [Ohm]")
        R_1 = pybamm.Parameter("Equivalent series resistance in R_1 [Ohm]")
        C_1 = pybamm.Parameter("Capactior [F]")
        num_cells = pybamm.Parameter("Number of cells connected in series to make a battery")
        gamma = pybamm.Parameter("Rate constant")
        R = pybamm.Parameter("Particle radius [m]")
        D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
        j = pybamm.Parameter("Interfacial current density [A.m-2]")
        F = pybamm.Parameter("Faraday constant [C.mol-1]")
        c0 = pybamm.Parameter("Initial concentration [mol.m-3]")

        ######################
        # Other set-up
        ######################

        i = self.param.current_with_time # Current [A]
        M = pybamm.FunctionParameter(
            "Maximum polarisation due to hysteresis", {"Current [A]": i}
        )
        eta = pybamm.FunctionParameter(
            "Charge efficiency", {"Current [A]": i}
        )

        ocv_data = pybamm.parameters.process_1D_data("ecm_example_ocv.csv")

        def ocv(soc):
            name, (x, y) = ocv_data
            return pybamm.Interpolant(x, y, soc, name)
        
        ######################
        # Governing equations
        ######################

        N = - D * pybamm.grad(c)  # flux
        dcdt = -pybamm.div(N)
        dzdt = - eta * i / Q
        di_R1dt = - i_R1 / (R_1 * C_1) - i / (R_1 * C_1)
        V_gov = ocv(pybamm.surf(c)) - i * R_0 - i_R1 * R_1 - V
        dhdt = abs(eta * i * gamma / Q) * (M - h)

        self.rhs = {z: dzdt, h: dhdt, i_R1: di_R1dt, c: dcdt}
        self.algebraic = {V: V_gov}

        self.initial_conditions = {z: pybamm.Scalar(1), h: pybamm.Scalar(-0.075), V: pybamm.Scalar(0), i_R1: pybamm.Scalar(0), c: c0}
        lbc = pybamm.Scalar(0)
        rbc = - j / F / D
        self.boundary_conditions = {c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}


        ######################
        # (Some) variables
        ######################

        self.variables = {
            "Battery voltage [V]": V * num_cells,
            "Current [A]": i,
            "Current variable [A]": i,  # for compatibility with pybamm.Experiment
            "Current through resistor R_1 [A]": i_R1,
            "Dynamic hysteresis voltage [V]": h, 
            "State of charge": z, 
            "Terminal voltage [V]": V,
            "Time [s]": pybamm.t,
            "Concentration [mol.m-3]": c,
            "Surface concentration [mol.m-3]": pybamm.surf(c),
            "Flux [mol.m-2.s-1]": N
        }

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum state of charge", z + 0.0001),
            pybamm.Event("Maximum state of charge", 1.0001 - z),
        ]