from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np

from helper_functions import euclidean_distance
from robot import RobotStatus
import swarm_simulator
from sim1 import SIM_PARAMS


def build_obstacles(sim_parameters):

    obstacles_list = []
    for idx in range(60, 120, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([50.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )
    for idx in range(-50, 42, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([50.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )

    for idx in range(40, 45, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([20.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )

    for idx in range(-10, 20, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([-60.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )

    for idx in range(-20, 10, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([10.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )

    # Initialize the simulation

    return obstacles_list


def run():
    sim_parameters = SIM_PARAMS

    obstacles_list = build_obstacles(sim_parameters)

    swarm_sim = swarm_simulator.SwarmSimulation(
        sim_parameters, obstacles=obstacles_list
    )

    print(
        f"Starting simulation with {sim_parameters['NUM_ROBOTS']} robots goals {sim_parameters['GOALS_POS']}."
    )
    print(f"Rearranging Regions: {sim_parameters['REARRANGING_REGIONS_RADII']}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Swarm Simulation)")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(sim_parameters["PLOT_X_RANGE"])
    ax.set_ylim(sim_parameters["PLOT_Y_RANGE"])
    ax.grid(True)

    # BACKGROUND (1x)
    rearranging_regions_colors = ["orange", "pink", "red"]
    for idx, goal_pos in enumerate(sim_parameters["GOALS_POS"]):
        ax.plot(
            [goal_pos[0]],
            [goal_pos[1]],
            "x",
            color="purple",
            markersize=10,
            label="Goal" if idx == 0 else None,
        )

        for idx2, R_radius in enumerate(sim_parameters["REARRANGING_REGIONS_RADII"]):
            rearrange_circle = Circle(
                goal_pos,
                R_radius,
                color=rearranging_regions_colors[idx2 % len(rearranging_regions_colors)],
                fill=False,
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                label=("Rearranging Region" if (idx == 0 and idx2 == 0) else None),
            )
            ax.add_patch(rearrange_circle)

    if len(obstacles_list) > 0:
        obs_xy = np.array([obs.position for obs in obstacles_list], dtype=float)
        ax.scatter(
            obs_xy[:, 0], obs_xy[:, 1],
            marker="s", s=40, color="black",
            label="Obstacle"
        )
        for obs in obstacles_list:
            obs_influence_circle = Circle(
                obs.position.tolist(),
                obs.influence_radius,
                color="gray",
                alpha=0.18,
                linewidth=0.0,
            )
            ax.add_patch(obs_influence_circle)

    # DYNAMIC
    pos0 = np.array([r.position for r in swarm_sim.robots], dtype=float)
    scat = ax.scatter(pos0[:, 0], pos0[:, 1], s=75, alpha=0.75, zorder=4)

    inline_lines = LineCollection([], linewidths=1.4, alpha=0.8, zorder=3, colors="red")
    ax.add_collection(inline_lines)

    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def colors_from_states(states):
        cols = []
        for st in states:
            if st == RobotStatus.GROUP:
                cols.append("tab:blue")
            elif st == RobotStatus.IN_LINE:
                cols.append("tab:green")
            elif st == RobotStatus.LEADER:
                cols.append("tab:red")
            else:
                cols.append("black")
        return cols

    def update(_):
        data = swarm_sim.run_simulation_step()

        pos = np.array([p for (_, p, _, _, _) in data], dtype=float)
        front = np.array([fp for (_, _, _, _, fp) in data], dtype=float)
        states = [st for (_, _, _, st, _) in data]

        scat.set_offsets(pos)
        scat.set_color(colors_from_states(states))


        segments = []
        for r in swarm_sim.robots:
            if r.inline_following_robot is not None:
                segments.append([r.position, r.inline_following_robot.position])
        inline_lines.set_segments(segments)

        finished = sum(1 for r in swarm_sim.robots if r.state == RobotStatus.FINISHED)
        text.set_text(f"step: {swarm_sim.time_step_count} | finished: {finished}/{sim_parameters['NUM_ROBOTS']}")

        return scat, inline_lines, text

    ani = FuncAnimation(
        fig,
        update,
        frames=sim_parameters["NUM_SIMULATION_STEPS"],
        interval=20,
        blit=False,
    )

    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    run()
