import pybamm

class ECM(pybamm.lithium_ion.BaseModel):
    """
    Equivalent Circuit Model (ECM) of a lithium-ion battery.

    """
    def __init__(self, name="Equivalent circuit model"):
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

        ######################
        # Parameters
        ######################

        Q = pybamm.Parameter("Total charge capacity [C]")
        R_0 = pybamm.Parameter("Equivalent series resistance in R_0 [Ohm]")
        R_1 = pybamm.Parameter("Equivalent series resistance in R_1 [Ohm]")
        C_1 = pybamm.Parameter("Capactior [F]")
        num_cells = pybamm.Parameter("Number of cells connected in series to make a battery")
        gamma = pybamm.Parameter("Rate constant")

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

        dzdt = - eta * i / Q
        di_R1dt = - i_R1 / (R_1 * C_1) - i / (R_1 * C_1)
        V_gov = ocv(z) - i * R_0 - i_R1 * R_1 - V
        #V_gov = - i * R_0 - i_R1 * R_1 - V
        dhdt = abs(eta * i * gamma / Q) * (M - h)

        self.rhs = {z: dzdt, h: dhdt, i_R1: di_R1dt}
        self.algebraic = {V: V_gov}

        self.initial_conditions = {z: pybamm.Scalar(1), h: pybamm.Scalar(-0.075), V: pybamm.Scalar(0), i_R1: pybamm.Scalar(0)}

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
            "Time [s]": pybamm.t
        }

        # Events specify points at which a solution should terminate
        '''self.events += [
            pybamm.Event("Minimum state of charge", z),
            pybamm.Event("Maximum state of charge", 1 - z),
        ]'''
        '''self.events += [
            pybamm.Event("Minimum voltage [V]", V - 2.56),
            pybamm.Event("Maximum voltage [V]", 4.26 - V),
        ]'''