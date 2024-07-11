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

        V = pybamm.Variable("Terminal voltage [V]")
        i_R1 = pybamm.Variable("Current through resistor R_1 [A]")
        c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")
        #surf_c = pybamm.Variable("Surface concentration [mol.m-3]")

        ######################
        # Parameters
        ######################

        R_0 = pybamm.Parameter("Equivalent series resistance in R_0 [Ohm]")
        R_1 = pybamm.Parameter("Equivalent series resistance in R_1 [Ohm]")
        C_1 = pybamm.Parameter("Capactior [F]")
        num_cells = pybamm.Parameter("Number of cells connected in series to make a battery")
        R = pybamm.Parameter("Particle radius [m]")
        D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
        j = pybamm.Parameter("Interfacial current density [A.m-2]")
        F = pybamm.Parameter("Faraday constant [C.mol-1]")
        c0 = pybamm.Parameter("Initial concentration [mol.m-3]")

        ######################
        # Other set-up
        ######################

        i = self.param.current_with_time # Current [A]

        ocv_data = pybamm.parameters.process_1D_data("ecm_example_ocv.csv")

        def ocv(soc):
            name, (x, y) = ocv_data
            return pybamm.Interpolant(x, y, soc, name)
        
        ######################
        # Governing equations
        ######################

        N = - D * pybamm.grad(c)  # flux
        dcdt = -pybamm.div(N)
        di_R1dt = - i_R1 / (R_1 * C_1) - i / (R_1 * C_1)
        V_gov = ocv(pybamm.surf(c)/25044.35001370933) - i * R_0 - i_R1 * R_1 - V
        #surf_c_gov = pybamm.surf(c) - surf_c

        self.rhs = {i_R1: di_R1dt, c: dcdt}
        self.algebraic = {V: V_gov}

        self.initial_conditions = {V: pybamm.Scalar(0), i_R1: pybamm.Scalar(0), c: c0}
        lbc = pybamm.Scalar(0)
        rbc = - i
        self.boundary_conditions = {c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}


        ######################
        # (Some) variables
        ######################

        self.variables = {
            "Battery voltage [V]": V * num_cells,
            "Current [A]": i,
            "Current variable [A]": i,  # for compatibility with pybamm.Experiment
            "Current through resistor R_1 [A]": i_R1,
            "Terminal voltage [V]": V,
            "Time [s]": pybamm.t,
            "Concentration [mol.m-3]": c,
            #"Surface concentration [mol.m-3]": surf_c,
            "Flux [mol.m-2.s-1]": N
        }

        # Events specify points at which a solution should terminate
        '''self.events += [
            pybamm.Event("Minimum state of charge", z + 0.0001),
            pybamm.Event("Maximum state of charge", 1.0001 - z),
        ]'''