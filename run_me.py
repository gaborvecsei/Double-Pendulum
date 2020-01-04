import pendulum

# Create pendulums

# Arm 1 properties
L1 = 1.0
M1 = 1.0

# Arm 2 properties
L2 = 1.0
M2 = 1.0

pend1 = pendulum.DoublePendulum(length_1=L1, mass_1=M1, length_2=L2, mass_2=M2)
pend2 = pend1.copy()
pend3 = pend1.copy()

print("Pendulums created")

# Calculate trajectories

start_time = 0
end_time = 20
dt = 0.1

init_angle_1 = 90.0
init_velocity_1 = 0.0

init_angle_2 = 90.0
init_velocity_2 = 0.0

pend1.calculate_trajectory(start_time, end_time, dt, init_angle_1, init_velocity_1, init_angle_2, init_velocity_2)
pend2.calculate_trajectory(start_time, end_time, dt, init_angle_1 + 0.1, init_velocity_1, init_angle_2, init_velocity_2)
pend3.calculate_trajectory(start_time, end_time, dt, init_angle_1 - 0.1, init_velocity_1, init_angle_2, init_velocity_2)

print("Pendulum trajectories calculated")

# Plot the trajectories

pend_plotter = pendulum.DoublePendulumPlotter([pend1, pend2, pend3], dt)
fig = pend_plotter.trajectory_plot_slider(custom_dt=0.1)
fig.write_html("art/trajectories.html")

print("Trajectory plot created")

# Animate pendulums

anim = pend_plotter.anim(frame_interval=60)
anim.save("art/chaotic_pendulums.gif", writer="pillow")

print("Animation created")
