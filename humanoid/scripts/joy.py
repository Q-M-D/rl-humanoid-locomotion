import pygame
from pygame.locals import *
import torch
from tabulate import tabulate

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
    
    reset = joystick.get_button(6)  # Back button to reset
    
    # Create command vector [x_vel, y_vel, yaw_vel, height]
    command_x = left_x * 0.6  # Forward/backward (max 0.6 m/s)
    command_y = left_y * 0.3   # Left/right (max 0.3 m/s)
    command_yaw = right_x * 3.14  # Rotation (max 0.3 rad/s)
    
    return torch.tensor([command_x, command_y, command_yaw, reset]).to('cuda:0')

def get_joystick_command_cpu(joystick):
    pygame.event.pump()
    
    # Get axis values (normalized to [-1, 1])
    left_x = -joystick.get_axis(1)   # Left stick x-axis to control forward/backward
    left_y = -joystick.get_axis(0)   # Left stick y-axis to control left/right
    right_x = -joystick.get_axis(3)  # Right stick x-axis to control yaw
    
    reset = joystick.get_button(6)  # Back button to reset
    
    # Create command vector [x_vel, y_vel, yaw_vel, height]
    command_x = left_x * 0.6  # Forward/backward (max 0.6 m/s)
    command_y = left_y * 0.3   # Left/right (max 0.3 m/s)
    command_yaw = right_x * 1.5  # Rotation (max 0.3 rad/s)
    
    return [command_x, command_y, command_yaw, reset]

def console_joy_command(joystick):
    pygame.event.pump()
    
    # Get all axis values
    headers = ["Input", "Description", "Value"]
    data = [
        # Axes
        [f"Axis 0", "Left stick x-axis", f"{joystick.get_axis(0):.4f}"],
        [f"Axis 1", "Left stick y-axis", f"{joystick.get_axis(1):.4f}"],
        [f"Axis 2", "Left trigger", f"{joystick.get_axis(2):.4f}"],
        [f"Axis 3", "Right stick x-axis", f"{joystick.get_axis(3):.4f}"],
        [f"Axis 4", "Right stick y-axis", f"{joystick.get_axis(4):.4f}"],
        [f"Axis 5", "Right trigger", f"{joystick.get_axis(5):.4f}"],
        [f"Button 0", "B", f"{joystick.get_button(0)}"],
        [f"Button 1", "A", f"{joystick.get_button(1)}"],
        [f"Button 2", "Y", f"{joystick.get_button(2)}"],
        [f"Button 3", "X", f"{joystick.get_button(3)}"],
        [f"Button 4", "Left Bumper", f"{joystick.get_button(4)}"],
        [f"Button 5", "Right Bumper", f"{joystick.get_button(5)}"],
        [f"Button 6", "Back", f"{joystick.get_button(6)}"],
        [f"Button 7", "Start", f"{joystick.get_button(7)}"],
    ]
    
    print(tabulate(data, headers=headers, tablefmt="grid"))
    

def main():
    joystick = setup_joystick()
    if joystick is None:
        return
    
    try:
        while True:
            # command = get_joystick_command(joystick)
            # print(f"Command: {command}")
            console_joy_command(joystick)
            
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
