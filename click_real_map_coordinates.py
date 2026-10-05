#!/usr/bin/env python3
"""
Interactive Real Drone Map Coordinate Picker for default.json Affine Matrix Calibration.
Loads high-res aerial map (drone_background_map.png).

Outputs ONLY the exact default.json coordinate format (X, -Y) so clicked points
can be directly copied for calculating/re-doing Affine Transformation Matrices.

Features:
- Live Pygame view of the Real Drone Aerial Map.
- Mouse Wheel / (+/-) keys to ZOOM IN and ZOOM OUT.
- Arrow keys / WASD / Right-Click Drag to PAN around the map.
- Left-Click anywhere on map to query exact default.json coordinate: (X, -Y).
- On-Screen Coordinates auto-fade and disappear after 30 seconds.
- Coordinates remain printed permanently in terminal output for easy copying.
"""

import os
import sys
import time
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MAP_IMAGE_PATH = os.path.join(SCRIPT_DIR, "drone_background_map.png")

def main():
    if not os.path.exists(MAP_IMAGE_PATH):
        print(f"Error: Map image not found at {MAP_IMAGE_PATH}")
        return

    pygame.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("Real Drone Map Coordinate Picker (Exact default.json Format for Affine Matrices)")
    font = pygame.font.SysFont("monospace", 17, bold=True)
    small_font = pygame.font.SysFont("monospace", 14, bold=True)

    print(f"Loading Real Drone Aerial Map from {MAP_IMAGE_PATH}...")
    orig_map_surf = pygame.image.load(MAP_IMAGE_PATH).convert()
    orig_w, orig_h = orig_map_surf.get_size()
    print(f"Loaded Real Map Image: {orig_w} x {orig_h} pixels.")

    # Viewport state
    zoom_scale = 0.45  # Initial fit scale
    offset_x = (WINDOW_WIDTH - orig_w * zoom_scale) / 2.0
    offset_y = (WINDOW_HEIGHT - orig_h * zoom_scale) / 2.0

    # Stores list of dicts: {"real_px": (x, y), "json_coord": (x, -y), "time": float, "id": int}
    clicked_points = []
    click_counter = 0

    print("\n" + "="*75)
    print("  Real Drone Map Coordinate Picker Active (default.json Format)!")
    print("  - LEFT-CLICK anywhere on map to print exact default.json coordinate (X, -Y).")
    print("  - MOUSE WHEEL / (+ / -): ZOOM IN and ZOOM OUT.")
    print("  - ARROW KEYS / WASD / RIGHT-CLICK DRAG: PAN around map.")
    print("  - On-screen coordinates auto-fade after 30 seconds.")
    print("  - Coordinates remain printed permanently in terminal output for copying.")
    print("  - Press ESC to exit.")
    print("="*75 + "\n")

    clock = pygame.time.Clock()
    running = True
    dragging = False
    drag_start_pos = (0, 0)
    offset_start_pos = (offset_x, offset_y)

    def screen_to_map_pixel(sx, sy):
        mx = (sx - offset_x) / zoom_scale
        my = (sy - offset_y) / zoom_scale
        return int(round(mx)), int(round(my))

    def map_pixel_to_screen(mx, my):
        sx = int(round(mx * zoom_scale + offset_x))
        sy = int(round(my * zoom_scale + offset_y))
        return sx, sy

    cached_scale = None
    scaled_surf = None

    def get_scaled_surface():
        nonlocal cached_scale, scaled_surf
        if cached_scale != zoom_scale:
            target_w = int(max(50, orig_w * zoom_scale))
            target_h = int(max(50, orig_h * zoom_scale))
            scaled_surf = pygame.transform.smoothscale(orig_map_surf, (target_w, target_h))
            cached_scale = zoom_scale
        return scaled_surf

    try:
        while running:
            clock.tick(60)

            # Continuous key panning
            keys = pygame.key.get_pressed()
            pan_speed = 12.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                offset_x += pan_speed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                offset_x -= pan_speed
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                offset_y += pan_speed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                offset_y -= pan_speed

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    display = pygame.display.set_mode((event.w, event.h), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_PAGEUP):
                        old_scale = zoom_scale
                        zoom_scale = min(5.0, zoom_scale * 1.25)
                        cx, cy = display.get_width() / 2.0, display.get_height() / 2.0
                        offset_x = cx - (cx - offset_x) * (zoom_scale / old_scale)
                        offset_y = cy - (cy - offset_y) * (zoom_scale / old_scale)
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_PAGEDOWN):
                        old_scale = zoom_scale
                        zoom_scale = max(0.1, zoom_scale / 1.25)
                        cx, cy = display.get_width() / 2.0, display.get_height() / 2.0
                        offset_x = cx - (cx - offset_x) * (zoom_scale / old_scale)
                        offset_y = cy - (cy - offset_y) * (zoom_scale / old_scale)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left Click -> Query default.json Coordinate
                        sx, sy = event.pos
                        mx, my = screen_to_map_pixel(sx, sy)
                        if 0 <= mx < orig_w and 0 <= my < orig_h:
                            json_x = float(mx)
                            json_y = float(-my)
                            click_counter += 1
                            item = {
                                "id": click_counter,
                                "real_px": (mx, my),
                                "json_coord": (json_x, json_y),
                                "time": time.time(),
                            }
                            clicked_points.append(item)

                            # Print ONLY clean default.json coordinate tuple
                            print(f"👉 CLICK #{click_counter:02d} -> default.json coordinate: ({json_x:.1f}, {json_y:.1f})")

                    elif event.button == 3:  # Right Click -> Start Panning Drag
                        dragging = True
                        drag_start_pos = event.pos
                        offset_start_pos = (offset_x, offset_y)

                    elif event.button == 4:  # Mouse Wheel Up -> Zoom In
                        mx, my = event.pos
                        old_scale = zoom_scale
                        zoom_scale = min(5.0, zoom_scale * 1.15)
                        offset_x = mx - (mx - offset_x) * (zoom_scale / old_scale)
                        offset_y = my - (my - offset_y) * (zoom_scale / old_scale)

                    elif event.button == 5:  # Mouse Wheel Down -> Zoom Out
                        mx, my = event.pos
                        old_scale = zoom_scale
                        zoom_scale = max(0.1, zoom_scale / 1.15)
                        offset_x = mx - (mx - offset_x) * (zoom_scale / old_scale)
                        offset_y = my - (my - offset_y) * (zoom_scale / old_scale)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:
                        dragging = False

                elif event.type == pygame.MOUSEMOTION and dragging:
                    dx = event.pos[0] - drag_start_pos[0]
                    dy = event.pos[1] - drag_start_pos[1]
                    offset_x = offset_start_pos[0] + dx
                    offset_y = offset_start_pos[1] + dy

            # Render Screen
            display.fill((15, 15, 20))
            map_surf = get_scaled_surface()
            display.blit(map_surf, (int(offset_x), int(offset_y)))

            # Draw outer map boundary outline
            map_rect = pygame.Rect(int(offset_x), int(offset_y), int(orig_w * zoom_scale), int(orig_h * zoom_scale))
            pygame.draw.rect(display, (0, 255, 255), map_rect, 2)

            # Draw top instructions banner
            curr_w, curr_h = display.get_width(), display.get_height()
            banner_rect = pygame.Rect(10, 10, min(760, curr_w - 20), 65)
            pygame.draw.rect(display, (20, 20, 30), banner_rect)
            pygame.draw.rect(display, (0, 255, 255), banner_rect, 2)

            t1 = font.render("REAL MAP COORD PICKER (default.json Format) | Click: Query | Wheel: Zoom", True, (0, 255, 255))
            t2 = small_font.render(f"Zoom Level: {zoom_scale*100:.0f}% | On-screen coordinates auto-fade after 30 sec", True, (255, 255, 255))
            display.blit(t1, (20, 18))
            display.blit(t2, (20, 45))

            # Draw active clicked markers (only if age < 30 seconds)
            now = time.time()
            for item in clicked_points:
                age = now - item["time"]
                if age < 30.0:  # Display on Pygame screen for 30 seconds
                    mx, my = item["real_px"]
                    jx, jy = item["json_coord"]
                    sx, sy = map_pixel_to_screen(mx, my)

                    if 0 <= sx < curr_w and 0 <= sy < curr_h:
                        remaining_sec = int(30.0 - age)

                        # Marker circles
                        pygame.draw.circle(display, (0, 0, 0), (sx, sy), 9)
                        pygame.draw.circle(display, (0, 255, 255), (sx, sy), 6)

                        # Label showing ONLY exact default.json coordinate
                        lbl_text = f"#{item['id']} ({jx:.1f}, {jy:.1f}) [{remaining_sec}s]"
                        lbl = font.render(lbl_text, True, (255, 255, 0))
                        lbl_bg = pygame.Rect(sx + 12, sy - 12, lbl.get_width() + 6, lbl.get_height() + 2)
                        pygame.draw.rect(display, (0, 0, 0), lbl_bg)
                        display.blit(lbl, (sx + 15, sy - 11))

            pygame.display.flip()

    finally:
        pygame.quit()
        print("\nExited Real Map Coordinate Picker.")

if __name__ == "__main__":
    main()
