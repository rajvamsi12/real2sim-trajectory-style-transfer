import argparse
import csv
import logging
import math
import random
import time
from datetime import datetime, timezone

import carla


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_vehicle_blueprints(world: carla.World) -> list[carla.ActorBlueprint]:
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    filtered = []
    for bp in blueprints:
        if bp.has_attribute("number_of_wheels"):
            try:
                if int(bp.get_attribute("number_of_wheels").as_int()) == 4:
                    filtered.append(bp)
            except Exception:
                continue
    return filtered if filtered else list(blueprints)


def build_convoy_from_spawn(
    carla_map: carla.Map,
    start_transform: carla.Transform,
    num_vehicles: int,
    gap_m: float,
) -> list[carla.Transform]:
    start_wp = carla_map.get_waypoint(
        start_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if not start_wp:
        return []

    transforms: list[carla.Transform] = []
    current_wp = start_wp
    for _ in range(num_vehicles):
        tf = carla.Transform(current_wp.transform.location, current_wp.transform.rotation)
        tf.location.z += 0.4
        transforms.append(tf)

        prev_candidates = current_wp.previous(gap_m)
        if not prev_candidates:
            break

        prev_candidates.sort(
            key=lambda wp: (
                wp.road_id != current_wp.road_id,
                wp.lane_id != current_wp.lane_id,
                abs(wp.lane_id),
                wp.transform.location.distance(current_wp.transform.location),
            )
        )
        current_wp = prev_candidates[0]

    return transforms


def find_best_convoy_transforms(
    world: carla.World,
    num_vehicles: int,
    base_gap_m: float,
) -> tuple[list[carla.Transform], float]:
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return [], base_gap_m

    candidate_gaps = [base_gap_m, max(6.0, base_gap_m - 2.0), 5.0, 4.0]
    best_transforms: list[carla.Transform] = []
    best_gap = base_gap_m

    for gap in candidate_gaps:
        for spawn in spawn_points:
            convoy = build_convoy_from_spawn(world.get_map(), spawn, num_vehicles, gap)
            if len(convoy) > len(best_transforms):
                best_transforms = convoy
                best_gap = gap
                if len(best_transforms) >= num_vehicles:
                    return best_transforms[:num_vehicles], best_gap

    return best_transforms, best_gap


def compute_speed_difference_percentage(speed_limit_kmh: float, target_speed_kmh: float) -> float:
    speed_limit_kmh = max(speed_limit_kmh, 1.0)
    pct = 100.0 * (speed_limit_kmh - target_speed_kmh) / speed_limit_kmh
    return max(min(pct, 90.0), -80.0)


def configure_vehicle_tm(
    tm: carla.TrafficManager,
    vehicle: carla.Vehicle,
    gap_m: float,
    target_speed_kmh: float,
) -> None:
    tm.auto_lane_change(vehicle, False)
    tm.ignore_lights_percentage(vehicle, 0.0)
    tm.ignore_signs_percentage(vehicle, 0.0)
    tm.ignore_vehicles_percentage(vehicle, 0.0)
    tm.ignore_walkers_percentage(vehicle, 0.0)
    tm.distance_to_leading_vehicle(vehicle, max(4.0, gap_m - 2.0))

    speed_limit = vehicle.get_speed_limit()
    pct = compute_speed_difference_percentage(speed_limit, target_speed_kmh)
    tm.vehicle_percentage_speed_difference(vehicle, pct)


def spawn_convoy(
    world: carla.World,
    tm: carla.TrafficManager,
    tm_port: int,
    num_vehicles: int,
    target_speed_kmh: float,
    gap_m: float,
    seed: int,
) -> tuple[list[carla.Vehicle], float]:
    rng = random.Random(seed)
    transforms, used_gap = find_best_convoy_transforms(world, num_vehicles, gap_m)
    if len(transforms) < num_vehicles:
        raise RuntimeError(
            f"Could only find {len(transforms)} convoy spawn points on one lane, "
            f"but {num_vehicles} are required."
        )

    blueprints = get_vehicle_blueprints(world)
    if not blueprints:
        raise RuntimeError("No vehicle blueprints available in this CARLA world.")

    vehicles_by_index: dict[int, carla.Vehicle] = {}
    for idx in range(num_vehicles - 1, -1, -1):
        spawn_tf = transforms[idx]
        spawned_vehicle = None

        # Try several random 4-wheel blueprints for each slot.
        for _ in range(min(12, len(blueprints))):
            bp = rng.choice(blueprints)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", f"convoy_{idx}")
            if bp.has_attribute("color"):
                colors = bp.get_attribute("color").recommended_values
                if colors:
                    bp.set_attribute("color", rng.choice(colors))
            spawned_vehicle = world.try_spawn_actor(bp, spawn_tf)
            if spawned_vehicle:
                break

        if not spawned_vehicle:
            raise RuntimeError(f"Failed to spawn vehicle at convoy index {idx}.")

        vehicle = spawned_vehicle
        vehicle.set_autopilot(True, tm_port)
        configure_vehicle_tm(tm, vehicle, used_gap, target_speed_kmh)
        vehicles_by_index[idx] = vehicle

    vehicles = [vehicles_by_index[i] for i in range(num_vehicles)]
    return vehicles, used_gap


def run_and_log(
    world: carla.World,
    vehicles: list[carla.Vehicle],
    output_csv: str,
    duration_minutes: float,
    sim_fps: float,
    log_hz: float,
) -> None:
    dt = 1.0 / sim_fps
    total_ticks = int(duration_minutes * 60.0 * sim_fps)
    log_interval_ticks = max(1, int(round(sim_fps / max(log_hz, 0.1))))
    progress_interval_ticks = max(1, int(round(sim_fps * 10.0)))

    logging.info(
        "Simulation started: %d vehicles, %.1f minutes, %.1f FPS (%.3f s dt), logging %.1f Hz",
        len(vehicles),
        duration_minutes,
        sim_fps,
        dt,
        log_hz,
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sim_time_s",
                "wall_time_utc",
                "vehicle_actor_id",
                "fleet_index",
                "x",
                "y",
                "z",
                "pitch",
                "yaw",
                "roll",
                "speed_kmh",
            ]
        )

        sim_time_s = 0.0
        for tick_idx in range(total_ticks):
            world.tick()
            sim_time_s += dt

            if tick_idx % log_interval_ticks == 0:
                wall_iso = datetime.now(timezone.utc).isoformat()
                for fleet_index, vehicle in enumerate(vehicles):
                    if not vehicle.is_alive:
                        continue
                    tf = vehicle.get_transform()
                    vel = vehicle.get_velocity()
                    speed_kmh = 3.6 * math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                    writer.writerow(
                        [
                            f"{sim_time_s:.3f}",
                            wall_iso,
                            vehicle.id,
                            fleet_index,
                            f"{tf.location.x:.3f}",
                            f"{tf.location.y:.3f}",
                            f"{tf.location.z:.3f}",
                            f"{tf.rotation.pitch:.3f}",
                            f"{tf.rotation.yaw:.3f}",
                            f"{tf.rotation.roll:.3f}",
                            f"{speed_kmh:.3f}",
                        ]
                    )

            if tick_idx % progress_interval_ticks == 0:
                elapsed_min = sim_time_s / 60.0
                logging.info("Progress: %.1f / %.1f simulation minutes", elapsed_min, duration_minutes)

            if tick_idx % (progress_interval_ticks * 3) == 0:
                handle.flush()

        handle.flush()

    logging.info("Simulation completed. Log saved to %s", output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spawn a Town10 autopilot convoy and log timestamp/position/orientation."
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    parser.add_argument("--town", default="Town10HD", help="Town map to load (e.g., Town10HD)")
    parser.add_argument("--num-vehicles", type=int, default=20, help="Number of convoy vehicles")
    parser.add_argument("--duration-min", type=float, default=20.0, help="Simulation duration in minutes")
    parser.add_argument("--target-speed-kmh", type=float, default=80.0, help="Target cruise speed in km/h")
    parser.add_argument("--sim-fps", type=float, default=20.0, help="Synchronous simulation FPS")
    parser.add_argument("--log-hz", type=float, default=2.0, help="CSV logging rate in Hz")
    parser.add_argument("--gap-m", type=float, default=8.0, help="Initial convoy gap in meters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="town10_autopilot_convoy_log.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    logging.info("Loading world: %s", args.town)
    world = client.load_world(args.town)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_random_device_seed(args.seed)
    tm.set_hybrid_physics_mode(True)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.sim_fps
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)

    vehicles: list[carla.Vehicle] = []
    try:
        vehicles, used_gap = spawn_convoy(
            world=world,
            tm=tm,
            tm_port=args.tm_port,
            num_vehicles=args.num_vehicles,
            target_speed_kmh=args.target_speed_kmh,
            gap_m=args.gap_m,
            seed=args.seed,
        )
        logging.info(
            "Spawned %d vehicles in convoy formation (gap %.1f m).",
            len(vehicles),
            used_gap,
        )
        run_and_log(
            world=world,
            vehicles=vehicles,
            output_csv=args.output,
            duration_minutes=args.duration_min,
            sim_fps=args.sim_fps,
            log_hz=args.log_hz,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        if vehicles:
            logging.info("Destroying %d convoy vehicles...", len(vehicles))
            client.apply_batch([carla.command.DestroyActor(v.id) for v in vehicles if v.is_alive])

        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)
        time.sleep(0.2)
        logging.info("Finished.")


if __name__ == "__main__":
    main()
