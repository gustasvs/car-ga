from bisect import bisect_right
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon


CAMERA_FOLLOW_X = 0.34
CAMERA_FOLLOW_Y = 0.52
THROTTLE_HISTORY_SECONDS = 8
THROTTLE_HISTORY_SAMPLES = 160
SMOKE_LIFETIME = 1.5


def build_road_points(road, delta_x):
    """
    road[i] = angle in degrees of segment i.

    delta_x is the LENGTH of the road segment, matching the physics.
    """
    points = [(0.0, 0.0)]

    for angle_deg in road[:-1]:
        angle = math.radians(angle_deg)

        x, y = points[-1]

        x += delta_x * math.cos(angle)
        y += delta_x * math.sin(angle)

        points.append((x, y))

    return points


def simulate(road, throttles, delta_x, k_drag, k_roll):
    """
    Same motion model as measure_individual().

    Returns information needed by the animation.
    """
    velocity = 0.0
    total_time = 0.0

    segments = []

    for i in range(len(throttles)):
        angle = math.radians(road[i])

        f_engine = throttles[i] ** 1.5

        f_grade = 9.81 * math.sin(angle)
        f_roll = k_roll * 9.81 * math.cos(angle)

        predicted_end_velocity = math.sqrt(
            max(
                0.0,
                velocity**2
                + 2
                * (
                    f_engine
                    - f_grade
                    - f_roll
                    - k_drag * velocity**2
                )
                * delta_x
            )
        )

        f_drag = (
            k_drag
            * ((velocity + predicted_end_velocity) / 2) ** 2
        )

        force = (
            f_engine
            - f_grade
            - f_roll
            - f_drag
        )

        end_speed_squared = (
            velocity**2
            + 2 * force * delta_x
        )

        # car stops before finishing this segment
        if end_speed_squared < 0:
            if force >= 0:
                break

            stop_distance = velocity**2 / (-2 * force)

            if velocity == 0:
                break

            segment_time = 2 * stop_distance / velocity

            segments.append({
                "angle": angle,
                "distance": stop_distance,
                "start_velocity": velocity,
                "acceleration": force,
                "start_time": total_time,
                "duration": segment_time,
                "segment": i,
            })

            total_time += segment_time
            break

        end_velocity = math.sqrt(end_speed_squared)

        if velocity + end_velocity == 0:
            break

        segment_time = (
            2 * delta_x
            / (velocity + end_velocity)
        )

        segments.append({
            "angle": angle,
            "distance": delta_x,
            "start_velocity": velocity,
            "acceleration": force,
            "start_time": total_time,
            "duration": segment_time,
            "segment": i,
        })

        total_time += segment_time
        velocity = end_velocity

    return segments, total_time


def visualize(
    road,
    throttles,
    delta_x=10,
    k_drag=0.1,
    k_roll=0.005,
):
    points = build_road_points(road, delta_x)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    segments, total_time = simulate(
        road,
        throttles,
        delta_x,
        k_drag,
        k_roll,
    )

    if total_time == 0:
        raise ValueError("Car cannot move.")

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#101a20")
    ax.set_facecolor("#101a20")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # -----------------------------
    # terrain
    # -----------------------------

    terrain_bottom = min(ys) - 20

    # Filled ground instead of just a graph line
    ax.fill_between(
        xs,
        ys,
        terrain_bottom,
        color="#173c3b",
        alpha=0.95,
    )

    # surface of the road / mountain
    ax.plot(
        xs,
        ys,
        color="#65d6c4",
        linewidth=5,
        alpha=0.22,
        zorder=2,
    )
    ax.plot(
        xs,
        ys,
        color="#c7f4e9",
        linewidth=2.2,
        zorder=3,
    )

    # Optional visible road points
    ax.scatter(
        xs,
        ys,
        s=12,
        color="#d6f1e8",
        alpha=0.7,
        zorder=4,
    )

    # -----------------------------
    # car
    # -----------------------------

    car_length = 4.4
    car_height = 1.35
    car = Polygon(
        [(0, 0)] * 5,
        closed=True,
        facecolor="#ff805f",
        edgecolor="#ffe0d4",
        linewidth=1.1,
        zorder=10,
    )
    rear_wheel = Circle((0, 0), 0.43, facecolor="#101a20", edgecolor="#d6f1e8", linewidth=1.2, zorder=11)
    front_wheel = Circle((0, 0), 0.43, facecolor="#101a20", edgecolor="#d6f1e8", linewidth=1.2, zorder=11)
    ax.add_patch(car)
    ax.add_patch(rear_wheel)
    ax.add_patch(front_wheel)
    smoke = ax.scatter([], [], s=[], zorder=8, linewidths=0)

    # -----------------------------
    # display setup
    # -----------------------------

    view_width = max(70, delta_x * 7)
    view_height = max(38, view_width * 0.48)
    ax.set_xlim(
        xs[0] - view_width * CAMERA_FOLLOW_X,
        xs[0] + view_width * (1 - CAMERA_FOLLOW_X),
    )
    ax.set_ylim(
        ys[0] - view_height * CAMERA_FOLLOW_Y,
        ys[0] + view_height * (1 - CAMERA_FOLLOW_Y),
    )

    # Very important:
    # 20 degrees visually appears as 20 degrees.
    ax.set_aspect("equal", adjustable="box")

    # Make it look like terrain rather than a graph
    ax.axis("off")

    # -----------------------------
    # animation
    # -----------------------------

    FPS = 30
    frame_dt = 1 / FPS
    segment_starts = [segment["start_time"] for segment in segments]
    max_throttle = max(1.0, max(throttles, default=0))
    throttle_history_ax = fig.add_axes((0.735, 0.735, 0.235, 0.19))
    throttle_history_ax.set_facecolor("#17282d")
    throttle_history_ax.set_xlim(0, 1)
    throttle_history_ax.set_ylim(0, max_throttle * 1.2)
    throttle_history_ax.set_xticks([])
    throttle_history_ax.tick_params(axis="y", colors="#9bb7b5", labelsize=7, length=0)
    for spine in throttle_history_ax.spines.values():
        spine.set_color("#355157")
    throttle_history_ax.grid(axis="y", color="#355157", linewidth=0.6, alpha=0.55)
    throttle_history_ax.set_title("THROTTLE HISTORY", loc="left", color="#d6f1e8", fontsize=8, pad=5)
    throttle_history = LineCollection([], linewidths=1.8)
    throttle_history_ax.add_collection(throttle_history)
    throttle_readout = throttle_history_ax.text(
        0.98,
        0.91,
        "0.0",
        transform=throttle_history_ax.transAxes,
        color="#ffb092",
        fontsize=9,
        fontweight="bold",
        ha="right",
        va="top",
    )
    smoke_particles = []
    smoke_emission = 0.0
    last_frame_time = None
    status = ax.text(
        0.025,
        0.94,
        "ROAD RUN  /  SEGMENT 01",
        transform=ax.transAxes,
        color="#d6f1e8",
        fontsize=10,
        fontweight="bold",
        va="top",
        family="sans-serif",
    )

    def oriented_point(car_x, car_y, angle, along, above):
        return (
            car_x + math.cos(angle) * along - math.sin(angle) * above,
            car_y + math.sin(angle) * along + math.cos(angle) * above,
        )

    def update(frame):
        nonlocal smoke_emission, last_frame_time
        # automatically loops
        t = (frame * frame_dt) % total_time
        elapsed = frame_dt if last_frame_time is None else max(frame_dt, (frame - last_frame_time) * frame_dt)
        last_frame_time = frame

        current = segments[-1]

        for segment in segments:
            start = segment["start_time"]
            end = start + segment["duration"]

            if start <= t < end:
                current = segment
                break

        local_t = t - current["start_time"]

        # s = v0*t + 1/2*a*t^2
        distance = (
            current["start_velocity"] * local_t
            + 0.5
            * current["acceleration"]
            * local_t**2
        )

        distance = max(
            0,
            min(distance, current["distance"])
        )

        i = current["segment"]

        x0, y0 = points[i]

        angle = current["angle"]

        # move along the actual angled road
        car_x = (
            x0
            + math.cos(angle) * distance
        )

        car_y = (
            y0
            + math.sin(angle) * distance
        )

        half_length = car_length / 2
        car.set_xy([
            oriented_point(car_x, car_y, angle, -half_length, 0),
            oriented_point(car_x, car_y, angle, -half_length * 0.72, car_height),
            oriented_point(car_x, car_y, angle, half_length * 0.58, car_height),
            oriented_point(car_x, car_y, angle, half_length, car_height * 0.25),
            oriented_point(car_x, car_y, angle, half_length, 0),
        ])
        rear_wheel.center = oriented_point(car_x, car_y, angle, -car_length * 0.29, 0.35)
        front_wheel.center = oriented_point(car_x, car_y, angle, car_length * 0.29, 0.35)

        ax.set_xlim(
            car_x - view_width * CAMERA_FOLLOW_X,
            car_x + view_width * (1 - CAMERA_FOLLOW_X),
        )
        ax.set_ylim(
            car_y - view_height * CAMERA_FOLLOW_Y,
            car_y + view_height * (1 - CAMERA_FOLLOW_Y),
        )
        speed = max(0, current["start_velocity"] + current["acceleration"] * local_t)
        status.set_text(f"ROAD RUN  /  SEGMENT {i + 1:02d}  /  {speed:04.1f} m/s")

        throttle = max(0, throttles[i])
        smoke_emission += throttle * elapsed * 5
        while smoke_emission >= 1:
            smoke_emission -= 1
            rear_x, rear_y = oriented_point(car_x, car_y, angle, -car_length * 0.56, 0.7)
            smoke_particles.append({
                "x": rear_x,
                "y": rear_y,
                "vx": -math.cos(angle) * (1.5 + throttle * 0.25),
                "vy": -math.sin(angle) * (1.5 + throttle * 0.25) + 0.8,
                "age": 0.0,
                "size": 12 + throttle * 3,
            })

        live_particles = []
        smoke_offsets = []
        smoke_sizes = []
        smoke_colors = []
        for particle in smoke_particles:
            particle["age"] += elapsed
            if particle["age"] >= SMOKE_LIFETIME:
                continue
            particle["x"] += particle["vx"] * elapsed
            particle["y"] += particle["vy"] * elapsed
            particle["vy"] += 0.7 * elapsed
            fade = 1 - particle["age"] / SMOKE_LIFETIME
            live_particles.append(particle)
            smoke_offsets.append((particle["x"], particle["y"]))
            smoke_sizes.append(particle["size"] * (1 + particle["age"]))
            smoke_colors.append((0.72, 0.82, 0.81, fade * 0.45))
        smoke_particles[:] = live_particles
        smoke.set_offsets(np.asarray(smoke_offsets, dtype=float).reshape(-1, 2))
        smoke.set_sizes(smoke_sizes)
        smoke.set_facecolor(smoke_colors)

        history_x = [sample / (THROTTLE_HISTORY_SAMPLES - 1) for sample in range(THROTTLE_HISTORY_SAMPLES)]
        history_y = []
        for sample in range(THROTTLE_HISTORY_SAMPLES):
            sample_time = (t - THROTTLE_HISTORY_SECONDS + sample * THROTTLE_HISTORY_SECONDS / (THROTTLE_HISTORY_SAMPLES - 1)) % total_time
            sample_segment_index = max(
                0,
                min(len(segments) - 1, bisect_right(segment_starts, sample_time) - 1),
            )
            history_y.append(throttles[segments[sample_segment_index]["segment"]])
        history_points = list(zip(history_x, history_y))
        throttle_history.set_segments([
            [history_points[index], history_points[index + 1]]
            for index in range(len(history_points) - 1)
        ])
        throttle_history.set_color([
            (0.39, 0.84, 0.77, 0.08 + 0.92 * history_x[index + 1])
            for index in range(len(history_points) - 1)
        ])
        throttle_readout.set_text(f"{throttle:0.1f}")

        return car, rear_wheel, front_wheel, smoke, status

    # KEEP this variable alive
    ani = FuncAnimation(
        fig,
        update,
        interval=1000 / FPS,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()

    return ani