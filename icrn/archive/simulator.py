from abc import ABC, abstractmethod
from bulk.simulate import _build_bulk_forward_step, _simulate_by_segments
from icrn.utils.dict_utils import sjdict_builder


class SimulatorError(Exception):
    pass


def _check_reaction_parameters():
    pass


def _check_reaction_dynamics():
    pass


def _check_concs():
    pass


class AbstractExperiment(ABC):
    def __init__(self, icrn, exp_params) -> None:
        self._icrn = icrn
        # self._exp_params = exp_params
        self._forward_step_f = self._build_forward_step(icrn, exp_params)

    @property
    def icrn(self):
        return self._icrn

    # @property
    # def exp_params(self):
    #     return self._exp_params

    @property
    def forward_step_f(self):
        return self._forward_step_f

    # def save_exp_params(self):
    #     pass

    @abstractmethod
    def simulate_time(self):
        pass

    @abstractmethod
    def simulate_segment(self):
        pass

    @abstractmethod
    def dict_builder(self):
        pass

    @abstractmethod
    def to_diffrax(self):
        pass


class WellMixed(AbstractExperiment):
    build_forward_step = _build_bulk_forward_step

    def __init__(self, icrn):
        self._icrn = icrn
        self._forward_step_f = _build_forward_step(icrn)

    def simulate_segments(
        self,
        conc_data,
        rate_constant_data,
        diff_data,
        dt,
        outer_scan_length=1,
        inner_scan_length=1,
    ):

        _simulate_by_segments(
            self._forward_step_f,
            conc_data,
            rate_constant_data,
            diff_data,
            dt,
            outer_scan_length,
            inner_scan_length,
        )

    # want this to be jittable
    def simulate_time(
        self, concs_data, rate_constant_data, diff_data, time, dt, sample_num=1
    ):
        if dt is None:
            dt = self._exp_params["dt"]

        f_apps = int(math.ceil(time / dt))
        scan_length = int(f_apps / sample_num)

        return self.simulate_segments(
            concs_data,
            rate_constant_data,
            diff_data,
            dt,
            segments=sample_num,
            scan_length=scan_length,
        )

    def dict_builder(
        self, concs_spec={}, rate_constant_spec={}, diff_spec={}, batch_size=None
    ):
        spatial_dim = self._exp_params["spatial_dim"]
        spatial_rate_constant = self._exp_params.get("spatial_rate_constant", False)
        shapes_dict = self._icrn.shapes()
        return sjdict_builder(
            shapes_dict,
            concs_spec,
            rate_constant_spec,
            diff_spec,
            batch_size,
            spatial_dim,
            spatial_rate_constant,
        )


class ReactionDiffusion(AbstractExperiment):
    def __init__(self, icrn, spatial_dim, spatial_rate_constant=False):
        self._icrn = icrn
        self._spatial_dim = spatial_dim
        self._spatial_rate_constant = spatial_rate_constant
        self._forward_step_f = _build_forward_step(
            icrn, spatial_dim, spatial_rate_constant=False
        )
