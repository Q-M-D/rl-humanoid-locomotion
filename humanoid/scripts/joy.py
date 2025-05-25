import pygame
from pygame.locals import *
import torch

def setup_joystick():
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("No joystick detected!")
        return None
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Initialized joystick: {joystick.get_name()}")
    return joystick

def clip_command(command: torch.Tensor) -> torch.Tensor:
    """
    Clip the command to the specified ranges.
    """
    return torch.clamp(command, min=torch.tensor([-0.3, -0.3, -2.0]), max=torch.tensor([0.6, 0.3, 2.0]))

def get_joystick_command(joystick):
    pygame.event.pump()
    
    """
    lin_vel_x = [-0.3, 0.6]   # min max [m/s]
    lin_vel_y = [-0.3, 0.3]   # min max [m/s]
    ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
    """
    
    # Get axis values (normalized to [-1, 1])
    left_x = -joystick.get_axis(1)   # Left stick x-axis to control forward/backward
    left_y = -joystick.get_axis(0)   # Left stick y-axis to control left/right
    right_x = -joystick.get_axis(3)  # Right stick x-axis to control yaw
    
    # Create command vector [x_vel, y_vel, yaw_vel, height]
    command_x = left_x * 0.6  # Forward/backward (max 0.6 m/s)
    command_y = left_y * 0.3   # Left/right (max 0.3 m/s)
    command_yaw = right_x * 0.5  # Rotation (max 0.3 rad/s)
    
    return clip_command(torch.tensor([command_x, command_y, command_yaw])).to('cuda:0')

def main():
    joystick = setup_joystick()
    if joystick is None:
        return
    
    try:
        while True:
            command = get_joystick_command(joystick)
            print(f"Command: {command}")
            
            # Here you would send the command to your robot or simulation
            # For example: env.set_commands(command)
            
            pygame.time.delay(100)  # Adjust delay as needed
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        pygame.quit()
        print("Joystick closed.")
        # Cleanup
        if joystick:
            joystick.quit()
            pygame.joystick.quit()
            pygame.quit()
            print("Joystick cleaned up.")


if __name__ == "__main__":
    main()
