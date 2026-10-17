# import sys
# import math
# import time
# import carla
# import numpy as np

# from agents.navigation.controller import VehiclePIDController

# class DummyWaypoint:
#     def __init__(self, transform):
#         self.transform = transform

# def connect_waypoints_straight(start_wp, end_wp, num_samples=50):
#     start_loc = start_wp.transform.location
#     end_loc = end_wp.transform.location

#     xs = np.linspace(start_loc.x, end_loc.x, num_samples)
#     ys = np.linspace(start_loc.y, end_loc.y, num_samples)
#     zs = np.linspace(start_loc.z, end_loc.z, num_samples)

#     dense_segment = []
#     rot = start_wp.transform.rotation
#     for i in range(num_samples):
#         loc = carla.Location(x=float(xs[i]), y=float(ys[i]), z=float(zs[i]))
#         trans = carla.Transform(location=loc, rotation=rot)
#         dense_segment.append(DummyWaypoint(trans))
#     return dense_segment

# class SimVehicle:
#     def __init__(self, world, bp, name, start, yaw, start_delay=0, wait_for=None):
#         self.world = world
#         self.name = name
#         self.start = start
#         self.start_delay = start_delay
#         self.wait_for = wait_for

#         self.sparse_route = []
#         self.dense_route = []
#         self.wp_index = 0
#         self.speed = 0.0
#         self.vel_ref = 0.0

#         spawn_transform = carla.Transform(
#             carla.Location(x=start[0], y=start[1], z=start[2]),
#             carla.Rotation(yaw=yaw)
#         )
#         vehicle = world.try_spawn_actor(bp, spawn_transform)
#         if vehicle is None:
#             raise RuntimeError(f"Failed to spawn vehicle '{self.name}' at the given location")
#         self.vehicle = vehicle
#         print(f"{self.name} spawned at: ({spawn_transform.location.x:.2f}, {spawn_transform.location.y:.2f}, {spawn_transform.location.z:.2f}) with yaw={yaw}°")

#         self.controller = VehiclePIDController(
#             self.vehicle,
#             args_lateral={'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 1.0 / 20.0},
#             args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': 1.0 / 20.0}
#         )

#     def set_route_from_points(self, route, yaw, custom_speed=None):
#         def create_wp(coord, yaw):
#             loc = carla.Location(x=coord[0], y=coord[1], z=coord[2])
#             rot = carla.Rotation(yaw=yaw)
#             return DummyWaypoint(carla.Transform(loc, rot))

#         self.sparse_route = [create_wp(pt, yaw) for pt in route]

#         self.dense_route = []
#         for i in range(len(self.sparse_route) - 1):
#             segment = connect_waypoints_straight(self.sparse_route[i], self.sparse_route[i + 1], num_samples=50)
#             if i < len(self.sparse_route) - 2:
#                 self.dense_route += segment[:-1]
#             else:
#                 self.dense_route += segment

#         total_distance = 0.0
#         for i in range(len(self.sparse_route) - 1):
#             p1 = self.sparse_route[i].transform.location
#             p2 = self.sparse_route[i + 1].transform.location
#             total_distance += p1.distance(p2)

#         # ✅ Convert km/h to m/s
#         self.speed = (custom_speed / 3.6) if custom_speed is not None else total_distance / 1.5
#         self.vel_ref = self.speed

#     def _compute_corners(self):
#         """
#         Compute the world‐coordinates of the 4 corners of the vehicle's bounding box.
#         Returns [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in FL, FR, RR, RL order.
#         """
#         tf = self.vehicle.get_transform()
#         cx = tf.location.x
#         cy = tf.location.y
#         yaw = math.radians(tf.rotation.yaw)
#         extent = self.vehicle.bounding_box.extent
#         a = extent.x  # half‐length
#         b = extent.y  # half‐width
#         hd = math.hypot(a, b)
#         alpha = math.atan2(b, a)
#         phis = [alpha, -alpha, math.pi + alpha, math.pi - alpha]
#         corners = []
#         for phi in phis:
#             corners.append((
#                 cx + hd * math.cos(yaw + phi),
#                 cy + hd * math.sin(yaw + phi)
#             ))
#         return corners

#     def is_ahead(self, other):
#         """
#         Return True if `other` vehicle's center lies in front of this vehicle,
#         within the lateral bounds defined by the left and right edges.
#         """
#         C1, C2, C3, C4 = self._compute_corners()
#         tf_o = other.vehicle.get_transform()
#         px, py = tf_o.location.x, tf_o.location.y

#         # Left edge test: C1 -> C4
#         x41 = C4[0] - C1[0]; y41 = C4[1] - C1[1]
#         xP1 = px - C1[0]; yP1 = py - C1[1]
#         L14 = x41 * yP1 - y41 * xP1

#         # Right edge test: C3 -> C2
#         x23 = C2[0] - C3[0]; y23 = C2[1] - C3[1]
#         xP3 = px - C3[0]; yP3 = py - C3[1]
#         L32 = x23 * yP3 - y23 * xP3

#         if L14 > 0 and L32 < 0:
#             # Optional: longitudinal (forward) check
#             tf_s = self.vehicle.get_transform()
#             dx = px - tf_s.location.x
#             dy = py - tf_s.location.y
#             heading = math.radians(tf_s.rotation.yaw)
#             return (dx * math.cos(heading) + dy * math.sin(heading)) >= 0

#         return False


#     def update(self, simulation_time, sim_vehicles):
#         if simulation_time < self.start_delay:
#             return

#         if self.wp_index >= len(self.dense_route):
#             control = carla.VehicleControl(throttle=0.0, brake=1.0)
#             self.vehicle.apply_control(control)
#             return

#         current_transform = self.vehicle.get_transform()
#         threshold = 5
#         while self.wp_index < len(self.dense_route) and \
#                 current_transform.location.distance(self.dense_route[self.wp_index].transform.location) < threshold:
#             self.wp_index += 1

#         if self.wp_index >= len(self.dense_route):
#             control = carla.VehicleControl(throttle=0.0, brake=1.0)
#             self.vehicle.apply_control(control)
#             return

#         target_wp = self.dense_route[self.wp_index]
#         control = self.controller.run_step(self.speed, target_wp)
#         self.vehicle.apply_control(control)

        

#         # 🏁 Detect and print vehicles ahead
#         ahead_vehicles = []
#         closest_distance = float('inf')
#         closest_name = None
        
#         for other in sim_vehicles.values():
#             if other is not self and self.is_ahead(other):
#                 # Calculate distance between vehicles
#                 self_loc = self.vehicle.get_transform().location
#                 other_loc = other.vehicle.get_transform().location
#                 distance = self_loc.distance(other_loc)
                
#                 ahead_vehicles.append(other.name)
                
#                 # Track closest vehicle
#                 if distance < closest_distance:
#                     closest_distance = distance
#                     closest_name = other.name
        
#         # Logging with closest vehicle info
#         if ahead_vehicles:
#             print(f"[{self.name}] Vehicles ahead: {ahead_vehicles}")
#             print(f"[{self.name}] Closest: {closest_name} ({closest_distance:.2f}m)")
#         else:
#             print(f"[{self.name}] No vehicles ahead")

# def main():
#     client = carla.Client("localhost", 2000)
#     client.set_timeout(10.0)
#     world = client.load_world("Town03")
#     settings = world.get_settings()
#     settings.synchronous_mode = True
#     settings.fixed_delta_seconds = 1.0 / 20.0
#     world.apply_settings(settings)

#     spectator = world.get_spectator()
#     spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=100), carla.Rotation(pitch=-90)))

#     blueprint_library = world.get_blueprint_library()

#     vehicles_data = [
#         {"name": "vehicle1", "route": [(50,5,0.6), (40,6,0), (30,6.5,0), (25,7,0), (20,7.5,0), (10,25,0), (5,45,0), (2,100,0)], "yaw": 180},
#         {"name": "vehicle2", "route": [(40,3,0.6), (35,3,0), (30,3,0), (25,5,0), (22,6,0), (18,15,0), (13,20,0), (10,23,0), (3,20,0)], "yaw": 180},
#         {"name": "vehicle3", "route": [(35,7,0.6), (25,7,0), (22,6,0), (18,15,0), (10,20,0), (-3,25,0), (-15,20,0), (-18,0,0)], "yaw": 180},
#         {"name": "vehicle4", "route": [(15,15,0.6), (5,25,0), (2,50,0), (2,80,0), (2,110,0)], "yaw": 135},
#         {"name": "bike1",    "route": [(7,30,0.6), (7,50,0), (7,80,0), (7,100,0), (7,130,0)],               "yaw": 90},
#         {"name": "vehicle5", "route": [(-8,30,0.6), (-10,20,0), (-15,15,0), (-19,5,0), (-18,-5,0), (-15,-15,0), (-10,-23,0), (0,-25,0)], "yaw": 270},
#         {"name": "vehicle6", "route": [(-11,40,0.6), (-10,28,0), (-10,20,0), (-15,15,0), (-19,5,0), (-18,-5,0), (-15,-15,0), (-10,28,0), (-5,40,0)], "yaw": 270},
#         {"name": "vehicle7", "route": [(-8,110,0.6), (-8,90,0), (-8,60,0), (-8,40,0), (-8,30,0)],                         "yaw": 270},
#         {"name": "vehicle8", "route": [(7,-30,0.6), (12,-20,0), (17,-15,0), (22,-11,0), (30,-5,0), (40,-5,0), (60,-5,0)],    "yaw": 70},
#         {"name": "vehicle9", "route": [(30,-4,0.6), (40,-4,0), (60,-4,0), (90,-4,0), (110,-4,0)],                      "yaw":   0},
#         {"name": "bike2",    "route": [(20,-15,0.6), (25,-10,0), (45,-10,0), (75,-10,0), (100,-10,0)],               "yaw":  45},
#         {"name": "vehicle10","route": [(7,-90,0.6), (7,-50,0), (7,-30,0), (5,-20,0)],                                "yaw":  90},
#     ]

#     sim_vehicles = {}
#     for veh_data in vehicles_data:
#         name = veh_data["name"]
#         if name.lower().startswith("bike"):
#             bp = blueprint_library.find("vehicle.bh.crossbike")
#         else:
#             bp = blueprint_library.find("vehicle.bmw.grandtourer")

#         sim_vehicle = SimVehicle(world, bp, name, veh_data["route"][0], veh_data["yaw"], 0)

#         if name == "vehicle1":
#             speed = 100
#         elif name == "vehicle3":
#             speed = 80
#         elif name == "vehicle2":
#             speed = 40
#         else:
#             speed = 60 
#         sim_vehicle.set_route_from_points(veh_data["route"], veh_data["yaw"], custom_speed=speed)

#         sim_vehicles[name] = sim_vehicle

#     simulation_time = 0.0
#     max_simulation_time = 20  
#     dt = settings.fixed_delta_seconds

#     try:
#         while simulation_time < max_simulation_time:
#             world.tick()
#             simulation_time += dt
#             for sim_vehicle in sim_vehicles.values():
#                 sim_vehicle.update(simulation_time, sim_vehicles)
#             time.sleep(dt)
#     finally:
#         print("\nFinal vehicle locations:")
#         for name, sim_vehicle in sim_vehicles.items():
#             loc = sim_vehicle.vehicle.get_transform().location
#             print(f"{name}: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")
#         print("Destroying actors")
#         actor_list = [veh.vehicle for veh in sim_vehicles.values()]
#         actor_list.append(spectator)
#         client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
#         settings = world.get_settings()
#         settings.synchronous_mode = False
#         settings.fixed_delta_seconds = None
#         world.apply_settings(settings)

# if __name__ == '__main__':
#     main()


import sys
import math
import time
import carla
import numpy as np

from agents.navigation.controller import VehiclePIDController

class DummyWaypoint:
    """
    A simple waypoint class that holds a CARLA transform.
    Used to create routes from raw coordinates.
    """
    def __init__(self, transform):
        self.transform = transform

def connect_waypoints_straight(start_wp, end_wp, num_samples=20):
    """
    Creates a dense list of DummyWaypoints by interpolating
    linearly between a start and end waypoint.
    """
    start_loc = start_wp.transform.location
    end_loc = end_wp.transform.location

    xs = np.linspace(start_loc.x, end_loc.x, num_samples)
    ys = np.linspace(start_loc.y, end_loc.y, num_samples)
    zs = np.linspace(start_loc.z, end_loc.z, num_samples)

    dense_segment = []
    rot = start_wp.transform.rotation
    for i in range(num_samples):
        loc = carla.Location(x=float(xs[i]), y=float(ys[i]), z=float(zs[i]))
        trans = carla.Transform(location=loc, rotation=rot)
        dense_segment.append(DummyWaypoint(trans))
    return dense_segment

class SimVehicle:
    """
    A simulation vehicle class that can follow a route with dynamic speeds.

    Two ways to drive it:
      - set_dynamic_route(): legacy distance-indexed route (next waypoint once within 3 m).
      - set_timed_route():   time-indexed route; the target is where the real vehicle was at
                             t + TIMED_LOOKAHEAD_S, so both vehicles of a scenario stay in sync.
    """
    TIMED_LOOKAHEAD_S = 0.5
    TIMED_MIN_LOOKAHEAD_M = 2.5
    TIMED_SPEED_GAIN = 0.8      # m/s of extra target speed per metre of along-track lag

    def __init__(self, world, bp, name, start, yaw, start_delay=0, max_brake=0.3, max_throttle=0.75):
        self.world = world
        self.name = name
        self.start_delay = start_delay
        self.vehicle = None # Initialize vehicle as None

        # Route and speed variables
        self.dense_route = [] # Will store tuples of (DummyWaypoint, target_speed_kmh)
        self.wp_index = 0
        self.timed_route = None # (times_s, xy[N,2], speeds_mps) when set_timed_route() is used
        self._constant_velocity_until = None # sim time at which the launch speed hold ends

        # Per-tick overrides set by an avoidance layer (reset by it every tick)
        self.speed_cap_mps = None   # upper bound on target speed
        self.lateral_offset_m = 0.0 # shift of the steering target, + = left of heading (timed route only)

        # Spawn the vehicle
        spawn_transform = carla.Transform(
            carla.Location(x=start[0], y=start[1], z=start[2]),
            carla.Rotation(yaw=yaw)
        )
        self.vehicle = world.try_spawn_actor(bp, spawn_transform)
        if self.vehicle is None:
            raise RuntimeError(f"Failed to spawn vehicle '{self.name}' at the given location")
        
        print(f"[{self.name}] Spawned at: ({spawn_transform.location.x:.2f}, {spawn_transform.location.y:.2f}) with yaw={yaw:.2f}°")

        # Initialize the PID controller
        # max_brake/max_throttle default to CARLA's 0.3/0.75; the scenario replay passes 1.0 for both
        self.controller = VehiclePIDController(
            self.vehicle,
            args_lateral={'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 1.0 / 20.0},
            args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': 1.0 / 20.0},
            max_brake=max_brake,
            max_throttle=max_throttle,
        )

    def hold_in_gear(self):
        """Before the scenario clock starts: hold still in 1st gear while the car drops onto the road.
        A fresh CARLA vehicle otherwise sits in gear 0 for ~1.2 s and loses any launch velocity."""
        if self.vehicle:
            self.vehicle.apply_control(carla.VehicleControl(hand_brake=True, manual_gear_shift=True, gear=1))

    def set_initial_speed(self, speed_mps, hold_s=0.3):
        """Launch the vehicle at its real speed (after hold_in_gear() + settle ticks).
        A plain set_target_velocity bleeds off within a few ticks because the wheels are not
        spinning yet, so the speed is held constant for hold_s while the drivetrain catches up."""
        if not self.vehicle:
            return
        self.vehicle.apply_control(carla.VehicleControl(manual_gear_shift=False))
        if speed_mps <= 0.0:
            return
        self.vehicle.enable_constant_velocity(carla.Vector3D(speed_mps, 0.0, 0.0))  # local frame: forward
        self._constant_velocity_until = hold_s

    def set_timed_route(self, times_s, points_xy, speeds_mps, z=0.6):
        """Time-indexed reference: points_xy[i] is where the real vehicle was at times_s[i]."""
        self.timed_route = (np.asarray(times_s, dtype=float), np.asarray(points_xy, dtype=float),
                            np.asarray(speeds_mps, dtype=float))
        self._route_z = z
        # Cumulative arc length, used to measure how far ahead/behind schedule the vehicle is
        seg = np.linalg.norm(np.diff(self.timed_route[1], axis=0), axis=1)
        self._route_s = np.concatenate([[0.0], np.cumsum(seg)])

    def reference(self, t):
        """(x, y, speed_mps, arc_length) of the real vehicle at time t (clamped to the route)."""
        times, xy, speeds = self.timed_route
        x = np.interp(t, times, xy[:, 0])
        y = np.interp(t, times, xy[:, 1])
        return x, y, float(np.interp(t, times, speeds)), float(np.interp(t, times, self._route_s))

    def _arc_length_at(self, x, y):
        """Arc length of the vehicle's projection onto the closest segment of the timed route."""
        xy, s = self.timed_route[1], self._route_s
        if len(xy) < 2:
            return 0.0
        a, b = xy[:-1], xy[1:]
        ab = b - a
        denom = np.maximum((ab ** 2).sum(axis=1), 1e-9)
        u = np.clip(((np.array([x, y]) - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
        proj = a + ab * u[:, None]
        i = int(np.argmin(np.hypot(proj[:, 0] - x, proj[:, 1] - y)))
        return float(s[i] + u[i] * (s[i + 1] - s[i]))

    def _point_at_arc_length(self, s):
        """Route point at arc length s; past the end, extend straight along the last segment."""
        xy, route_s = self.timed_route[1], self._route_s
        if len(xy) < 2 or s <= route_s[-1]:
            return float(np.interp(s, route_s, xy[:, 0])), float(np.interp(s, route_s, xy[:, 1]))
        d = xy[-1] - xy[-2]
        d = d / max(np.linalg.norm(d), 1e-9)
        extra = s - route_s[-1]
        return float(xy[-1, 0] + d[0] * extra), float(xy[-1, 1] + d[1] * extra)

    def route_length(self):
        return float(self._route_s[-1]) if self.timed_route is not None else 0.0

    def progress(self):
        """Fraction of the timed route's arc length covered."""
        if self.timed_route is None or not self.vehicle or self._route_s[-1] < 1e-6:
            return 1.0
        loc = self.vehicle.get_location()
        return self._arc_length_at(loc.x, loc.y) / self._route_s[-1]

    def release_launch_hold(self):
        """End the constant-velocity launch hold early (e.g. before an emergency brake)."""
        if self._constant_velocity_until is not None and self.vehicle:
            self.vehicle.disable_constant_velocity()
        self._constant_velocity_until = None

    def _timed_update(self, simulation_time):
        if self._constant_velocity_until is not None and simulation_time >= self._constant_velocity_until:
            self.release_launch_hold()
        times = self.timed_route[0]
        if simulation_time > times[-1]:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return False
        _, _, ref_speed, ref_s = self.reference(simulation_time)
        _, _, _, ahead_s = self.reference(simulation_time + self.TIMED_LOOKAHEAD_S)
        loc = self.vehicle.get_location()
        own_s = self._arc_length_at(loc.x, loc.y)

        # Steer at a point at least TIMED_MIN_LOOKAHEAD_M ahead along the route; a target closer
        # than that (slow vehicles) makes the lateral PID steer erratically.
        tx, ty = self._point_at_arc_length(max(ahead_s, own_s + self.TIMED_MIN_LOOKAHEAD_M))
        if self.lateral_offset_m:
            fwd = self.vehicle.get_transform().get_forward_vector()
            tx -= fwd.y * self.lateral_offset_m
            ty += fwd.x * self.lateral_offset_m
        target_wp = DummyWaypoint(carla.Transform(carla.Location(x=tx, y=ty, z=self._route_z)))

        # Speed up when behind schedule, slow down when ahead (along the route, not straight-line)
        lag_m = ref_s - own_s
        speed_mps = max(0.0, ref_speed + self.TIMED_SPEED_GAIN * lag_m)
        if self.speed_cap_mps is not None:
            speed_mps = min(speed_mps, self.speed_cap_mps)
        self.vehicle.apply_control(self.controller.run_step(speed_mps * 3.6, target_wp))
        return True

    def set_dynamic_route(self, route_with_speeds):
        """
        Sets a route where each waypoint has an associated target speed.
        """
        if not route_with_speeds:
            return

        # Create a sparse route of DummyWaypoints and their speeds in km/h
        sparse_route = []
        for point_data in route_with_speeds:
            loc = carla.Location(x=point_data[0], y=point_data[1], z=point_data[2])
            rot = carla.Rotation() # Yaw is handled by controller steering
            wp = DummyWaypoint(carla.Transform(loc, rot))
            speed_kmh = point_data[3]
            sparse_route.append((wp, speed_kmh))

        # Create a dense route by interpolating between sparse points
        self.dense_route = []
        for i in range(len(sparse_route) - 1):
            start_wp, start_speed = sparse_route[i]
            end_wp, _ = sparse_route[i+1]

            segment_waypoints = connect_waypoints_straight(start_wp, end_wp)
            
            # Assign the speed of the segment's start to all interpolated points
            segment_with_speeds = [(wp, start_speed) for wp in segment_waypoints]

            if i < len(sparse_route) - 2:
                self.dense_route.extend(segment_with_speeds[:-1])
            else:
                self.dense_route.extend(segment_with_speeds)

    def update(self, simulation_time, sim_vehicles):
        """
        Updates the vehicle's state, applying control to follow the
        dense route with continuously changing target speeds.
        Returns False if the route is complete.
        """
        if not self.vehicle or simulation_time < self.start_delay:
            return True # Still running

        if self.timed_route is not None:
            return self._timed_update(simulation_time)

        if self.wp_index >= len(self.dense_route):
            # Route is complete, stop the vehicle
            control = carla.VehicleControl(throttle=0.0, brake=1.0)
            self.vehicle.apply_control(control)
            return False 

        current_transform = self.vehicle.get_transform()
        threshold = 3.0 # Proximity to waypoint to consider it "reached"

        # Advance waypoint index if we are close enough to the target
        while self.wp_index < len(self.dense_route) and \
              current_transform.location.distance(self.dense_route[self.wp_index][0].transform.location) < threshold:
            self.wp_index += 1

        if self.wp_index >= len(self.dense_route):
            return False # Route just finished

        # Get the current target waypoint and its associated speed (km/h)
        target_wp, target_speed_kmh = self.dense_route[self.wp_index]
        if self.speed_cap_mps is not None:
            target_speed_kmh = min(target_speed_kmh, self.speed_cap_mps * 3.6)

        # Feed the dynamic target speed to the PID controller
        control = self.controller.run_step(target_speed_kmh, target_wp)
        self.vehicle.apply_control(control)

        return True # Still running

    def destroy(self):
        """Safely destroys the vehicle actor."""
        if self.vehicle:
            # Check if the actor is still alive before destroying
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
