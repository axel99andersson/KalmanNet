import torch

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, dt):
        """
        Kp (torch.Tensor) : Size (n,1)
        Ki (torch.Tensor) : Size (n,1)
        Kd (torch.Tensor) : Size (n,1)
        setpoint (torch.Tensor) : Size (batch_size, n,1) or (n,1) (same as measurements, (n,1) is the size of one measurement)
        dt (float) : time discretization
        """
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.dt = dt
        self.integral = torch.zeros_like(setpoint)
        self.prev_error = torch.zeros_like(setpoint)
    
    def compute_control(self, measurement, clip_value=True, clip_min_value=-1.0, clip_max_value=1.0):
        """
        Computes the control signal based on measurements (works for batched and multivariate measurements)

        Args:
            measurement (torch.Tensor) : Size (batch_size, n, 1)

        Returns:
            control signals (for each batch and measurement)
        """
        
        error = self.setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        unclipped_signal =  self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        if not clip_value:
            return unclipped_signal
        
        return torch.clip(unclipped_signal, -1.0, 1.0)
        
    
    def set_setpoint(self, new_setpoint: torch.Tensor):
        """
        Change the setpoint

        """
        self.setpoint = new_setpoint


if __name__ == "__main__":
    measurements = torch.rand(4, 2, 1)
    setpoints = torch.ones(4,2,1)

    Kp = torch.tensor([[2], [1]])
    Ki = torch.tensor([[2], [1]])
    Kd = torch.tensor([[2], [1]])

    dt = 0.1

    controller = PIDController(Kp, Ki, Kd, setpoints, dt)

    u = controller.compute_control(measurements)

    print(f"Control Signal: {u}")
    print(f"Size: {u.size()}")