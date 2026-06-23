from .logger import MLLogger

import pynvml
import time


class GPUMetrics:
    """
    A class for keeping track of GPU metrics
    """

    def __init__(self, logger: MLLogger) -> None:
        self.logger = logger
        
        self.pwr_kwh        = 0.0
        self.pwr_lastreport = None

        pynvml.nvmlInit()
        self.gpu_count      = pynvml.nvmlDeviceGetCount()
        self.gpu_handles    = [ pynvml.nvmlDeviceGetHandleByIndex(gpui) for gpui in range(self.gpu_count) ]
        self.gpu_models     = [ pynvml.nvmlDeviceGetName(gpuh) for gpuh in self.gpu_handles ]
        self.pwr_lastvals   = [ 0.0 for gpuh in self.gpu_handles]
        

    def _format_gpu_name(self, gpu_index: int) -> str:
        """
        In the format:
          e.g. "gpu_0:NVIDIA.GeForce.RTX.3070"
          e.g. "gpu_1:NVIDIA.GeForce.RTX.3070"
        
        Or if you have more than 10 GPUs:
          e.g. "gpu_01:NVIDIA.GeForce.RTX.3070"
          e.g. "gpu_42:NVIDIA.GeForce.RTX.3070"
        """

        return f"gpu_{gpu_index:0{1 + (self.gpu_count//10)}d}:{self.gpu_models[gpu_index].replace(' ', '.')}"


    def _tick_gpu_power(self, iteration: int = None) -> None:
        """
        Tick the recording of GPU power. ONLY LOGS if iteration != None
        """

        # Only record after this function has been called at least once
        if (self.pwr_lastreport == None): self.pwr_lastreport = time.time()

        # We need an iteration number to do logging
        log_it = (iteration != None)
        
        # Figure out how much power is being pulled currently
        total_gpus_power = 0.0
        for gpui, gpuh in enumerate(self.gpu_handles):
            # Returns mw
            try:
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(gpuh) / 1000
                self.pwr_lastvals[gpui] = gpu_power
            except:
                gpu_power = self.pwr_lastvals[gpui]

            if (log_it):
                self.logger.cmllogger.report_scalar(title="GPU Power", series=self._format_gpu_name(gpui), iteration=iteration, value=gpu_power)
            
            total_gpus_power += gpu_power

        # Multiply the power by seconds to interpolate usage over time (in J or ws)
        # Then /1000 to get kws, then /60^2 to get kwh
        self.pwr_kwh += ((total_gpus_power * (time.time() - self.pwr_lastreport)) / 1000) / 60**2
        self.pwr_lastreport = time.time()

        if (log_it):
            self.logger.cmllogger.report_scalar(title="GPU Power",       series="total", iteration=iteration, value=total_gpus_power)
            self.logger.cmllogger.report_scalar(title="GPU Power Usage", series="total_kwh", iteration=iteration, value=self.pwr_kwh)


    def gpu_power_report(self, iteration: int):
        """Tick the power odomiter and report values"""
        self._tick_gpu_power(iteration=iteration)

    def gpu_power_tick(self):
        """Tick the power odomoiter, do not report"""
        self._tick_gpu_power(iteration=None)

