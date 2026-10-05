#!/usr/bin/env python3
"""
Interactive CARLA Town10HD Map Coordinate Picker.
Features:
- Live Pygame camera view of Town10HD map.
- Mouse Wheel / (+/-) keys to ZOOM IN and ZOOM OUT.
- Arrow keys / WASD / Right-Click Drag to PAN around the map.
- Left-Click anywhere to query and print exact CARLA 3D World (x, y, z) coordinates.
- On-Screen Coordinates auto-fade and disappear after 30 seconds.
- Coordinates remain printed permanently in the terminal log.
"""

import os
import sys
import math
import time
import pygame
import numpy as np
import carla

CARLA_HOST = "localhost"
CARLA_PORT = 2000
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAM_FOV = 90.0

def get_camera_intrinsic_matrix(w, h, fov):
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def pixel_to_world_on_ground(u, v, cam_transform, K, ground_z=0.5):
    """
    Raycasts a 2D window pixel (u, v) onto the 3D ground plane z = ground_z.
    Returns carla.Location(x, y, z).
    """
    K_inv = np.linalg.inv(K)
    px_h = np.array([u, v, 1.0])
    
    # Ray in camera OpenCV frame (X-right, Y-down, Z-forward)
    ray_cv = K_inv @ px_h
    
    # Convert OpenCV camera frame to CARLA camera frame (X-forward, Y-right, Z-up)
    ray_carla_cam = np.array([ray_cv[2], ray_cv[0], -ray_cv[1]])
    
    # Rotate ray to world frame using camera rotation matrix
    matrix = np.array(cam_transform.get_matrix())
    R = matrix[:3, :3]
    cam_pos = np.array([cam_transform.location.x, cam_transform.location.y, cam_transform.location.z])
    
    ray_world = R @ ray_carla_cam
    
    if abs(ray_world[2]) < 1e-6:
        return None
        
    t = (ground_z - cam_pos[2]) / ray_world[2]
    world_pt = cam_pos + t * ray_world
    
    return carla.Location(x=float(world_pt[0]), y=float(world_pt[1]), z=ground_z)

def world_to_pixel(loc, cam_transform, K):
    """
    Projects a 3D CARLA Location(x, y, z) onto the 2D window pixel (u, v).
    """
    matrix = np.array(cam_transform.get_matrix())
    R = matrix[:3, :3]
    cam_pos = np.array([cam_transform.location.x, cam_transform.location.y, cam_transform.location.z])
    pt_world = np.array([loc.x, loc.y, loc.z])
    
    pt_cam_carla = R.T @ (pt_world - cam_pos)
    pt_cv = np.array([pt_cam_carla[1], -pt_cam_carla[2], pt_cam_carla[0]])
    
    if pt_cv[2] <= 0:
        return None
        
    px_h = K @ pt_cv
    u = int(px_h[0] / px_h[2])
    v = int(px_h[1] / px_h[2])
    return (u, v)

def main():
    print("Connecting to CARLA Server at localhost:2000...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(15.0)
    world = client.load_world("Town10HD")
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # Camera position state
    cam_x = -50.0
    cam_y = 10.0
    cam_z = 220.0
    
    bp_library = world.get_blueprint_library()
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
    cam_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
    cam_bp.set_attribute("fov", str(CAM_FOV))
    
    def make_cam_tf(x, y, z):
        return carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        )
        
    cam_tf = make_cam_tf(cam_x, cam_y, cam_z)
    camera = world.spawn_actor(cam_bp, cam_tf)
    
    pygame.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Town10HD Interactive Map Coordinate Picker")
    font = pygame.font.SysFont("monospace", 17, bold=True)
    small_font = pygame.font.SysFont("monospace", 14, bold=True)
    
    latest_surface = None
    K = get_camera_intrinsic_matrix(WINDOW_WIDTH, WINDOW_HEIGHT, CAM_FOV)
    
    # Stores list of dicts: {"loc": carla.Location, "time": float, "id": int}
    clicked_points = []
    click_counter = 0
    
    def on_image(image):
        nonlocal latest_surface
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] # BGR to RGB
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        latest_surface = surface

    camera.listen(on_image)
    
    print("\n" + "="*75)
    print("  CARLA Town10HD Interactive Map Coordinate Picker Active!")
    print("  - LEFT-CLICK anywhere on map to print CARLA (x, y) coordinates.")
    print("  - MOUSE WHEEL / (+ / -): ZOOM IN and ZOOM OUT.")
    print("  - ARROW KEYS / WASD / RIGHT-CLICK DRAG: PAN Camera around map.")
    print("  - Coordinates disappear from Pygame screen after 30 seconds.")
    print("  - Coordinates remain printed permanently in terminal output.")
    print("  - Press ESC to exit.")
    print("="*75 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    dragging = False
    drag_start_pos = (0, 0)
    cam_start_pos = (cam_x, cam_y)
    
    try:
        while running:
            clock.tick(30)
            world.tick()
            
            cam_updated = False
            
            # Continuous key panning
            keys = pygame.key.get_pressed()
            pan_speed = (cam_z / 200.0) * 3.5
            
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                cam_x -= pan_speed
                cam_updated = True
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                cam_x += pan_speed
                cam_updated = True
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                cam_y += pan_speed  # Top-down camera: +Y is Up on map
                cam_updated = True
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                cam_y -= pan_speed
                cam_updated = True
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_PAGEUP):
                        # Zoom In
                        cam_z = max(15.0, cam_z - 25.0)
                        cam_updated = True
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_PAGEDOWN):
                        # Zoom Out
                        cam_z = min(550.0, cam_z + 25.0)
                        cam_updated = True
                        
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left Click -> Query Coordinate
                        u, v = event.pos
                        loc = pixel_to_world_on_ground(u, v, cam_tf, K)
                        if loc:
                            click_counter += 1
                            clicked_points.append({
                                "id": click_counter,
                                "loc": loc,
                                "time": time.time(),
                                "px": (u, v)
                            })
                            print(f"👉 CLICK #{click_counter:02d} AT ({u}, {v})px --> CARLA Location: X = {loc.x:+.2f} m, Y = {loc.y:+.2f} m, Z = {loc.z:.1f} m")
                            
                    elif event.button == 3:  # Right Click -> Start Panning Drag
                        dragging = True
                        drag_start_pos = event.pos
                        cam_start_pos = (cam_x, cam_y)
                        
                    elif event.button == 4:  # Mouse Wheel Up -> Zoom In
                        cam_z = max(15.0, cam_z - 15.0)
                        cam_updated = True
                        
                    elif event.button == 5:  # Mouse Wheel Down -> Zoom Out
                        cam_z = min(550.0, cam_z + 15.0)
                        cam_updated = True
                        
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:
                        dragging = False
                        
                elif event.type == pygame.MOUSEMOTION and dragging:
                    dx = event.pos[0] - drag_start_pos[0]
                    dy = event.pos[1] - drag_start_pos[1]
                    scale = (cam_z / 220.0) * 0.25
                    cam_x = cam_start_pos[0] - dx * scale
                    cam_y = cam_start_pos[1] + dy * scale
                    cam_updated = True

            if cam_updated:
                cam_tf = make_cam_tf(cam_x, cam_y, cam_z)
                camera.set_transform(cam_tf)
            
            if latest_surface:
                display.blit(latest_surface, (0, 0))
                
                # Draw top instructions banner
                banner_rect = pygame.Rect(10, 10, 720, 65)
                pygame.draw.rect(display, (20, 20, 30), banner_rect)
                pygame.draw.rect(display, (0, 255, 255), banner_rect, 2)
                
                t1 = font.render("Click: Query Coord | Wheel/(+/-): Zoom In/Out | WASD/Right-Drag: Pan", True, (0, 255, 255))
                t2 = small_font.render(f"Camera Height (Zoom): {cam_z:.0f}m | Screen markers auto-fade after 30 sec", True, (255, 255, 255))
                display.blit(t1, (20, 18))
                display.blit(t2, (20, 45))
                
                # Draw active clicked markers (only if age < 30 seconds)
                now = time.time()
                for item in clicked_points:
                    age = now - item["time"]
                    if age < 30.0:  # Display on screen for 30 seconds
                        loc = item["loc"]
                        px = world_to_pixel(loc, cam_tf, K)
                        if px and 0 <= px[0] < WINDOW_WIDTH and 0 <= px[1] < WINDOW_HEIGHT:
                            u, v = px
                            remaining_sec = int(30.0 - age)
                            
                            # Marker circles
                            pygame.draw.circle(display, (0, 0, 0), (u, v), 9)
                            pygame.draw.circle(display, (0, 255, 255), (u, v), 6)
                            
                            # Coordinate label
                            lbl = font.render(f"#{item['id']} ({loc.x:+.1f}m, {loc.y:+.1f}m) [{remaining_sec}s]", True, (255, 255, 0))
                            lbl_bg = pygame.Rect(u + 12, v - 12, lbl.get_width() + 6, lbl.get_height() + 2)
                            pygame.draw.rect(display, (0, 0, 0), lbl_bg)
                            display.blit(lbl, (u + 15, v - 11))
                
                pygame.display.flip()
                
    finally:
        try: camera.stop(); camera.destroy()
        except Exception: pass
        pygame.quit()
        print("\nExited CARLA Coordinate Picker.")

if __name__ == "__main__":
    main()
