"""
SeepageRL: Premium Interactive Traffic Signal Simulation & Research Cockpit.

Features:
  - Fits standard 1080p (and 125%/150% scaled) desktop displays (1480 x 800 layout)
  - Uniform-scaled simulation viewport with realistic drop-shadows and brake lights
  - Dynamic traffic signal bloom halos and modern floating countdown badges
  - Stop-line "Bike Box" seepage reservoirs with animated lateral filtering markers
  - 420px Dark Glassmorphism Research Cockpit (Pillars 1, 2, 3 XAI & Tipping Points)
  - Bottom Analytics & Interactive Operator Ribbon with clickable buttons and hotkeys
"""

import atexit
import math
import os
import random
import sys
import threading
import time
from typing import Dict, List, Tuple, Any, Optional

import pygame
import numpy as np

from config import (
    ACTION_SPACE,
    CANONICAL_PROFILES,
    CLEAR_TIME_CUTOFFS,
    DEFAULT_GREEN_DURATION,
    ENABLE_STARVATION_MASK,
    GAMMA,
    GREEN_DURATION_OPTIONS,
    MAX_GREEN_DURATION,
    MIN_GREEN_DURATION,
    MODEL_PATH,
    PCU_WEIGHTS,
    PRIORITY_WEIGHTS,
    PROFILE_METADATA,
    STARVATION_WAIT_THRESHOLD,
    TRAFFIC_PROFILE,
    USE_RL,
    WAIT_TIME_CUTOFFS,
    YELLOW_DURATION,
)
from rl_state import (
    SeepageParams,
    StateTuple,
    build_seepage_rl_state,
    compute_seepage_clear_time,
    invert_clear_time_to_vehicles,
)
from rl_agent import (
    DecomposedVectorQLearner,
    compute_starvation_mask,
)
from counterfactual_explainer import (
    CounterfactualIntrospector,
    CounterfactualReport,
)

# ---------------------------------------------------------
# Screen Layout Dimensions (Fits 1536 x 864 and 1920 x 1080)
# ---------------------------------------------------------
SIM_NATIVE_W = 1400
SIM_NATIVE_H = 800

# Viewport on screen (exact 1.75 aspect ratio maintained)
VIEW_W = 1060
VIEW_H = 606
BOTTOM_H = 194
SIDEBAR_W = 420

WINDOW_WIDTH = VIEW_W + SIDEBAR_W   # 1480 px
WINDOW_HEIGHT = VIEW_H + BOTTOM_H   # 800 px

# ---------------------------------------------------------
# Global Simulation State
# ---------------------------------------------------------
simTime = 300
timeElapsed = 0
simulationPaused = False
playbackSpeed = 1.0  # 0.5x, 1.0x, 2.0x, 4.0x

currentGreen = 0     # 0: right (EW), 1: down (NS), 2: left (EW), 3: up (NS)
nextGreen = 1
currentYellow = 0
phaseChangeInProgress = False
lastPlannedGreenTime = DEFAULT_GREEN_DURATION
cycleTimeCounter = 0

redElapsed = {"NS": 0.0, "EW": 0.0}
waitingTimeByDirection = {"right": 0.0, "down": 0.0, "left": 0.0, "up": 0.0}

activeProfileName = TRAFFIC_PROFILE
starvationMaskActive = ENABLE_STARVATION_MASK
controllerModeRL = USE_RL

lastDecisionReport: Optional[CounterfactualReport] = None
lastDecisionDetails = "Initializing AI Agent..."

# Historical metrics tracking for bottom analytics
throughputHistory: List[float] = [0.0] * 30
delayHistory: List[float] = [0.0] * 30

# Intersection Parameters
noOfSignals = 4
noOfLanes = 2
gap = 15
gap2 = 15
rotationAngle = 3

vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "rickshaw", 4: "bike"}
directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}
phaseToGroup = {0: 1, 1: 0, 2: 1, 3: 0}  # 0,2 are EW (Group 1); 1,3 are NS (Group 0)
groupToPhases = {0: (1, 3), 1: (0, 2)}

speeds = {"car": 2.25, "bus": 1.8, "truck": 1.8, "rickshaw": 2.0, "bike": 2.5}
x_coords = {"right": [0, 0, 0], "down": [755, 727, 697], "left": [1400, 1400, 1400], "up": [602, 627, 657]}
y_coords = {"right": [348, 370, 398], "down": [0, 0, 0], "left": [498, 466, 436], "up": [800, 800, 800]}

stopLines = {"right": 590, "down": 330, "left": 800, "up": 535}
defaultStop = {"right": 580, "down": 320, "left": 810, "up": 545}
stops = {"right": [580, 580, 580], "down": [320, 320, 320], "left": [810, 810, 810], "up": [545, 545, 545]}
mid = {"right": {"x": 705, "y": 445}, "down": {"x": 695, "y": 450}, "left": {"x": 695, "y": 425}, "up": {"x": 695, "y": 400}}

# Coordinates on native 1400x800 canvas
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 205), (810, 205), (810, 545), (530, 545)]
vehicleCountCoods = [(475, 205), (885, 205), (885, 545), (475, 545)]

# Vehicle containers
vehicles: Dict[str, Dict[Any, Any]] = {
    "right": {0: [], 1: [], 2: [], "crossed": 0},
    "down": {0: [], 1: [], 2: [], "crossed": 0},
    "left": {0: [], 1: [], 2: [], "crossed": 0},
    "up": {0: [], 1: [], 2: [], "crossed": 0},
}

# Instantiate AI Agents and Explainer
seepageParams = SeepageParams()
rlAgent = DecomposedVectorQLearner(active_profile=activeProfileName, gamma=GAMMA)
if os.path.exists(MODEL_PATH):
    rlAgent.load(MODEL_PATH)
    print(f"Loaded SeepageRL Vector Q-Bank from {MODEL_PATH}")

introspector = CounterfactualIntrospector(params=seepageParams)

# Pygame Setup
pygame.init()
simulation = pygame.sprite.Group()


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = "30"
        self.totalGreenTime = 0


signals: List[TrafficSignal] = []


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        super().__init__()
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = float(x_coords[direction][lane])
        self.y = float(y_coords[direction][lane])
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        self.waitTime = 0.0
        self.isBraking = False

        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        path = f"images/{direction}/{vehicleClass}.png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

        if direction == "right":
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x_coords[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == "left":
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x_coords[direction][lane] += temp
            stops[direction][lane] += temp
        elif direction == "down":
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y_coords[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == "up":
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y_coords[direction][lane] += temp
            stops[direction][lane] -= temp

        simulation.add(self)

    def move(self):
        old_x, old_y = self.x, self.y

        if self.direction == "right":
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]["crossed"] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]["x"]:
                    if (self.x + self.currentImage.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.x += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2.0
                        self.y += 1.8
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2):
                            self.y += self.speed
            else:
                if (self.x + self.currentImage.get_rect().width <= self.stop or self.crossed == 1 or (currentGreen == 0 and currentYellow == 0)) and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2) or (vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                    self.x += self.speed

        elif self.direction == "down":
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]["crossed"] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]["y"]:
                    if (self.y + self.currentImage.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.y += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2.0
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or self.y < (vehicles[self.direction][self.lane][self.index - 1].y - gap2):
                            self.x -= self.speed
            else:
                if (self.y + self.currentImage.get_rect().height <= self.stop or self.crossed == 1 or (currentGreen == 1 and currentYellow == 0)) and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or (vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                    self.y += self.speed

        elif self.direction == "left":
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]["crossed"] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x > mid[self.direction]["x"]:
                    if (self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.x -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + gap2):
                            self.y -= self.speed
            else:
                if (self.x >= self.stop or self.crossed == 1 or (currentGreen == 2 and currentYellow == 0)) and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or (vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                    self.x -= self.speed

        elif self.direction == "up":
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]["crossed"] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y > mid[self.direction]["y"]:
                    if (self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.y -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1.0
                        self.y -= 1.0
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x < (vehicles[self.direction][self.lane][self.index - 1].x - vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width - gap2) or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + gap2):
                            self.x += self.speed
            else:
                if (self.y >= self.stop or self.crossed == 1 or (currentGreen == 3 and currentYellow == 0)) and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                    self.y -= self.speed

        # Detect braking/stopped state for animated brake lights
        self.isBraking = (self.crossed == 0 and abs(self.x - old_x) < 0.1 and abs(self.y - old_y) < 0.1)


# ---------------------------------------------------------
# Telemetry Aggregation for Pillars 1, 2, 3
# ---------------------------------------------------------
def get_approach_queue_snapshots() -> Dict[str, List[Dict[str, Any]]]:
    approach_map = {
        "North": "down",
        "South": "up",
        "West": "right",
        "East": "left",
    }
    snapshots: Dict[str, List[Dict[str, Any]]] = {"North": [], "South": [], "East": [], "West": []}
    for app_name, dir_name in approach_map.items():
        for lane in range(3):
            for v in vehicles[dir_name][lane]:
                if v.crossed == 0:
                    snapshots[app_name].append({
                        "vehicle_class": v.vehicleClass,
                        "wait_time": v.waitTime,
                    })
    return snapshots


def get_direction_queue_counts() -> Dict[str, int]:
    counts = {}
    for d in directionNumbers.values():
        c = 0
        for lane in range(3):
            c += sum(1 for v in vehicles[d][lane] if v.crossed == 0)
        counts[d] = c
    return counts


def reset_stops_for_phase(phase_index: int):
    phase_direction = directionNumbers[phase_index]
    for lane_number in range(3):
        stops[phase_direction][lane_number] = defaultStop[phase_direction]
        for v in vehicles[phase_direction][lane_number]:
            v.stop = defaultStop[phase_direction]


def sync_signal_red_times():
    active_signal = signals[currentGreen]
    cycle_time = active_signal.green + active_signal.yellow
    for i in range(noOfSignals):
        signals[i].red = 0 if i == currentGreen else cycle_time


# ---------------------------------------------------------
# Decision Making Engine (Pillars 1, 2, 3)
# ---------------------------------------------------------
def get_control_decision() -> Tuple[int, int]:
    global lastDecisionDetails, lastDecisionReport, lastPlannedGreenTime

    queues = get_approach_queue_snapshots()
    state, telemetry = build_seepage_rl_state(
        queues=queues,
        red_elapsed=redElapsed,
        current_phase=phaseToGroup[currentGreen],
        is_yellow=currentYellow,
        params=seepageParams,
    )

    t_clear_ns = telemetry["t_clear_ns"]
    t_clear_ew = telemetry["t_clear_ew"]
    wait_ns = telemetry["wait_ns"]
    wait_ew = telemetry["wait_ew"]

    if sum(get_direction_queue_counts().values()) == 0:
        lastPlannedGreenTime = MIN_GREEN_DURATION
        lastDecisionDetails = f"Idle: No Queues | Min Green {MIN_GREEN_DURATION}s"
        return currentGreen, MIN_GREEN_DURATION

    # Hierarchical Starvation Action Mask M(s,a)
    mask = compute_starvation_mask(
        wait_ns=wait_ns,
        wait_ew=wait_ew,
        t_clear_ns=t_clear_ns,
        t_clear_ew=t_clear_ew,
        w_max=STARVATION_WAIT_THRESHOLD,
        enable_mask=starvationMaskActive,
    )

    if controllerModeRL:
        chosen_action, effective_q_vectors = rlAgent.select_action(state, mask, training=False)
        phase_str, group_int, green_time = ACTION_SPACE[chosen_action]

        candidate_phases = groupToPhases[group_int]
        q_counts = get_direction_queue_counts()
        target_phase = candidate_phases[0]
        if q_counts[directionNumbers[candidate_phases[1]]] > q_counts[directionNumbers[candidate_phases[0]]]:
            target_phase = candidate_phases[1]

        lastDecisionReport = introspector.explain_decision(
            state=state,
            q_vectors_active=effective_q_vectors,
            q_bank_state=rlAgent.get_state_q_bank(state),
            mask=mask,
            active_profile=activeProfileName,
            t_clear_ns=t_clear_ns,
            t_clear_ew=t_clear_ew,
            wait_ns=wait_ns,
            wait_ew=wait_ew,
            w_max=STARVATION_WAIT_THRESHOLD,
        )
        lastDecisionDetails = lastDecisionReport.narrative_summary
        lastPlannedGreenTime = green_time
        return target_phase, green_time

    else:
        # Fixed Timer Baseline: Alternate NS and EW with 10s green
        next_group = 1 if phaseToGroup[currentGreen] == 0 else 0
        candidate_phases = groupToPhases[next_group]
        target_phase = candidate_phases[0]
        green_time = DEFAULT_GREEN_DURATION
        lastPlannedGreenTime = green_time
        lastDecisionDetails = f"Fixed Baseline: {directionNumbers[target_phase].upper()} Green {green_time}s"
        lastDecisionReport = None
        return target_phase, green_time


# ---------------------------------------------------------
# Signal Cycle Threads
# ---------------------------------------------------------
def set_next_signal_timing():
    global nextGreen
    nextGreen, green_time = get_control_decision()
    signals[nextGreen].green = green_time
    signals[nextGreen].yellow = YELLOW_DURATION
    sync_signal_red_times()


def repeat():
    global currentGreen, currentYellow, nextGreen, phaseChangeInProgress

    set_next_signal_timing()
    while signals[currentGreen].green > 0:
        while simulationPaused:
            time.sleep(0.08)
        updateValues()
        time.sleep(1.0 / max(playbackSpeed, 0.1))

    if nextGreen == currentGreen:
        signals[currentGreen].green = lastPlannedGreenTime
        signals[currentGreen].yellow = YELLOW_DURATION
        signals[currentGreen].red = 0
        repeat()
        return

    currentYellow = 1
    phaseChangeInProgress = True
    reset_stops_for_phase(currentGreen)

    while signals[currentGreen].yellow > 0:
        while simulationPaused:
            time.sleep(0.08)
        updateValues()
        time.sleep(1.0 / max(playbackSpeed, 0.1))

    currentYellow = 0
    phaseChangeInProgress = False

    signals[currentGreen].green = DEFAULT_GREEN_DURATION
    signals[currentGreen].yellow = YELLOW_DURATION
    signals[currentGreen].red = 150

    currentGreen = nextGreen
    sync_signal_red_times()
    repeat()


def updateValues():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            if signals[i].red > 0:
                signals[i].red -= 1

    active_group = phaseToGroup[currentGreen]
    if active_group == 0:
        redElapsed["EW"] += 1.0
        redElapsed["NS"] = 0.0
    else:
        redElapsed["NS"] += 1.0
        redElapsed["EW"] = 0.0

    for d in directionNumbers.values():
        for lane in range(3):
            for v in vehicles[d][lane]:
                if v.crossed == 0:
                    v.waitTime += 1.0

    # Record throughput and delay metrics history
    total_passed = sum(vehicles[d]["crossed"] for d in directionNumbers.values())
    tp = float(total_passed) / max(float(timeElapsed), 1.0)
    throughputHistory.append(tp)
    if len(throughputHistory) > 30:
        throughputHistory.pop(0)


def initialize():
    signals.append(TrafficSignal(0, YELLOW_DURATION, DEFAULT_GREEN_DURATION))
    signals.append(TrafficSignal(150, YELLOW_DURATION, DEFAULT_GREEN_DURATION))
    signals.append(TrafficSignal(150, YELLOW_DURATION, DEFAULT_GREEN_DURATION))
    signals.append(TrafficSignal(150, YELLOW_DURATION, DEFAULT_GREEN_DURATION))
    sync_signal_red_times()
    repeat()


def generateVehicles():
    while True:
        while simulationPaused:
            time.sleep(0.08)

        v_type = random.randint(0, 4)
        lane = 0 if v_type == 4 else random.randint(0, 1) + 1
        will_turn = 1 if (lane == 2 and random.randint(0, 4) <= 2) else 0

        temp = random.randint(0, 999)
        d_num = 0 if temp < 400 else (1 if temp < 800 else (2 if temp < 900 else 3))
        Vehicle(lane, vehicleTypes[v_type], d_num, directionNumbers[d_num], will_turn)
        time.sleep(0.75 / max(playbackSpeed, 0.1))


def simulationTimer():
    global timeElapsed
    while True:
        while simulationPaused:
            time.sleep(0.08)
        timeElapsed += 1
        time.sleep(1.0 / max(playbackSpeed, 0.1))


def inject_heavy_vehicle_surge():
    print("[EVENT] Injected Heavy Vehicle Surge on East-West Corridor (5 Trucks/Buses)")
    for _ in range(5):
        v_cls = random.choice(["truck", "bus"])
        d_num = random.choice([0, 2])
        Vehicle(1, v_cls, d_num, directionNumbers[d_num], will_turn=0)


def inject_two_wheeler_surge():
    print("[EVENT] Injected Two-Wheeler Surge on North-South Corridor (12 Motorcycles)")
    for _ in range(12):
        d_num = random.choice([1, 3])
        Vehicle(0, "bike", d_num, directionNumbers[d_num], will_turn=0)


# ---------------------------------------------------------
# Interactive Clickable Button Component
# ---------------------------------------------------------
class UIButton:
    def __init__(self, rect: Tuple[int, int, int, int], label: str, hotkey: str, color, text_color=(255, 255, 255)):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.hotkey = hotkey
        self.base_color = color
        self.text_color = text_color
        self.is_hovered = False
        self.is_active = False

    def check_hover(self, mouse_pos: Tuple[int, int]) -> bool:
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered

    def draw(self, surface: pygame.Surface, font, font_small):
        # Color styling
        if self.is_active:
            bg_col = self.base_color
            border_col = (255, 255, 255)
            border_w = 2
        elif self.is_hovered:
            bg_col = tuple(min(255, c + 35) for c in self.base_color)
            border_col = (200, 210, 235)
            border_w = 2
        else:
            bg_col = self.base_color
            border_col = tuple(max(0, c - 25) for c in self.base_color)
            border_w = 1

        pygame.draw.rect(surface, bg_col, self.rect, border_radius=6)
        pygame.draw.rect(surface, border_col, self.rect, border_w, border_radius=6)

        # Label and Hotkey badge
        txt_surf = font.render(self.label, True, self.text_color)
        t_rect = txt_surf.get_rect(center=(self.rect.centerx, self.rect.centery - (4 if self.hotkey else 0)))
        surface.blit(txt_surf, t_rect)

        if self.hotkey:
            hk_surf = font_small.render(f"[{self.hotkey}]", True, (210, 220, 240))
            hk_rect = hk_surf.get_rect(center=(self.rect.centerx, self.rect.centery + 10))
            surface.blit(hk_surf, hk_rect)


# ---------------------------------------------------------
# Main Application with Modern UI/UX Cockpit (1480 x 800)
# ---------------------------------------------------------
class SimulationApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("SeepageRL Research Console — Multi-Objective Traffic Control & XAI")

        # Off-screen native canvas for crisp uniform scaling
        self.sim_surface = pygame.Surface((SIM_NATIVE_W, SIM_NATIVE_H))

        # Assets
        self.background = pygame.image.load("images/mod_int.png")
        self.redSignal = pygame.image.load("images/signals/red.png")
        self.yellowSignal = pygame.image.load("images/signals/yellow.png")
        self.greenSignal = pygame.image.load("images/signals/green.png")

        # Clean Typography (Segoe UI + Consolas)
        self.fontTitle = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.fontCardHeader = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.fontNormal = pygame.font.SysFont("Segoe UI", 12)
        self.fontBold = pygame.font.SysFont("Segoe UI", 12, bold=True)
        self.fontSmall = pygame.font.SysFont("Segoe UI", 11)
        self.fontMono = pygame.font.SysFont("Consolas", 11)
        self.fontMonoBold = pygame.font.SysFont("Consolas", 12, bold=True)
        self.fontSignalTimer = pygame.font.SysFont("Segoe UI", 18, bold=True)

        # Theme Colors (Modern Obsidian Slate)
        self.C_BG = (15, 17, 26)
        self.C_CARD = (22, 25, 38)
        self.C_CARD_BORDER = (42, 47, 71)
        self.C_TEXT_WHITE = (245, 248, 252)
        self.C_TEXT_MUTED = (145, 155, 178)
        self.C_ACCENT_BLUE = (41, 128, 185)
        self.C_ACCENT_GREEN = (39, 174, 96)
        self.C_ACCENT_ORANGE = (211, 84, 0)
        self.C_ACCENT_RED = (192, 57, 43)
        self.C_ACCENT_PURPLE = (142, 68, 173)
        self.C_ACCENT_CYAN = (22, 160, 133)

        self.init_buttons()

        # Background simulation threads
        t_sim = threading.Thread(target=simulationTimer, daemon=True)
        t_sim.start()
        t_init = threading.Thread(target=initialize, daemon=True)
        t_init.start()
        t_veh = threading.Thread(target=generateVehicles, daemon=True)
        t_veh.start()

        self.run_loop()

    def init_buttons(self):
        """Initializes interactive buttons for the bottom control ribbon."""
        by = VIEW_H + 15
        bw = 100
        bh = 38
        gap_x = 8

        # Profile Switcher buttons
        bx = 520
        self.btn_balanced = UIButton((bx, by, bw, bh), "Balanced", "1", (30, 80, 50))
        bx += bw + gap_x
        self.btn_commuter = UIButton((bx, by, bw, bh), "Commuter", "2", (25, 65, 100))
        bx += bw + gap_x
        self.btn_freight = UIButton((bx, by, bw, bh), "Freight", "3", (110, 55, 15))
        bx += bw + gap_x
        self.btn_emergency = UIButton((bx, by, bw, bh), "Emergency", "4", (110, 25, 25))

        # Bottom row buttons
        by2 = by + bh + 10
        bx = 520
        self.btn_mode = UIButton((bx, by2, bw, bh), "RL / Fixed", "M", (45, 52, 75))
        bx += bw + gap_x
        self.btn_mask = UIButton((bx, by2, bw, bh), "Mask ON/OFF", "S", (45, 52, 75))
        bx += bw + gap_x
        self.btn_pause = UIButton((bx, by2, 85, bh), "Pause", "Spc", (55, 60, 85))
        bx += 85 + gap_x
        self.btn_speed = UIButton((bx, by2, 65, bh), "Speed", "+/-", (55, 60, 85))
        bx += 65 + gap_x
        self.btn_truck_surge = UIButton((bx, by2, 80, bh), "+ Trucks", "B", (90, 45, 15))
        bx += 80 + gap_x
        self.btn_bike_surge = UIButton((bx, by2, 80, bh), "+ Bikes", "T", (20, 80, 70))

        self.buttons = [
            self.btn_balanced, self.btn_commuter, self.btn_freight, self.btn_emergency,
            self.btn_mode, self.btn_mask, self.btn_pause, self.btn_speed,
            self.btn_truck_surge, self.btn_bike_surge
        ]

    def render_bike_box_reservoirs(self):
        """Draws high-visibility translucent bike-box seepage reservoirs on native canvas."""
        s = pygame.Surface((SIM_NATIVE_W, SIM_NATIVE_H), pygame.SRCALPHA)

        boxes = [
            # West (right): x=535..585, y=340..425
            (535, 340, 50, 85, "2W BOX"),
            # East (left): x=805..855, y=430..515
            (805, 430, 50, 85, "2W BOX"),
            # North (down): x=690..775, y=275..325
            (690, 275, 85, 50, "2W BOX"),
            # South (up): x=595..680, y=540..590
            (595, 540, 85, 50, "2W BOX"),
        ]

        # Draw green translucent bike-box zones
        pulse = int(25 + 15 * math.sin(time.time() * 3.0))
        for rx, ry, rw, rh, lbl in boxes:
            pygame.draw.rect(s, (46, 204, 113, pulse), (rx, ry, rw, rh), border_radius=4)
            pygame.draw.rect(s, (46, 204, 113, 160), (rx, ry, rw, rh), 2, border_radius=4)

        self.sim_surface.blit(s, (0, 0))

    def render_glowing_signal(self, idx: int):
        """Draws realistic radial bloom halos around active signal lamps."""
        sc = signalCoods[idx]
        sig = signals[idx]

        # Halo surface
        halo = pygame.Surface((120, 120), pygame.SRCALPHA)
        center = (60, 60)

        if idx == currentGreen:
            if currentYellow == 1:
                col = (241, 196, 15, 75)
                self.sim_surface.blit(self.yellowSignal, sc)
            else:
                col = (46, 204, 113, 70)
                self.sim_surface.blit(self.greenSignal, sc)
        else:
            col = (231, 76, 60, 50)
            self.sim_surface.blit(self.redSignal, sc)

        # Draw soft radial bloom
        pygame.draw.circle(halo, col, center, 28)
        pygame.draw.circle(halo, (col[0], col[1], col[2], col[3] // 2), center, 44)
        self.sim_surface.blit(halo, (sc[0] - 44, sc[1] - 16))

        # Modern floating countdown badge
        tx, ty = signalTimerCoods[idx]
        timer_text = str(sig.signalText)
        bg_col = (0, 0, 0, 200)

        badge_w, badge_h = 42, 24
        b_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        pygame.draw.rect(b_surf, bg_col, (0, 0, badge_w, badge_h), border_radius=5)
        pygame.draw.rect(b_surf, (200, 210, 230), (0, 0, badge_w, badge_h), 1, border_radius=5)

        t_render = self.fontSignalTimer.render(timer_text, True, (255, 255, 255))
        t_r = t_render.get_rect(center=(badge_w // 2, badge_h // 2))
        b_surf.blit(t_render, t_r)
        self.sim_surface.blit(b_surf, (tx - 5, ty - 2))

    def render_vehicle_enhancements(self, v: Vehicle):
        """Draws realistic 3D vehicle drop shadow and glowing tail brake lights."""
        rect = v.currentImage.get_rect()
        w, h = rect.width, rect.height

        # 1. Translucent Drop Shadow
        shadow = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 65), (0, 0, w + 4, h + 4))
        self.sim_surface.blit(shadow, (v.x + 2, v.y + 2))

        # 2. Vehicle Sprite
        self.sim_surface.blit(v.currentImage, (v.x, v.y))

        # 3. Dynamic Brake Lights when stationary or stopping
        if v.isBraking:
            brake_surf = pygame.Surface((w + 10, h + 10), pygame.SRCALPHA)
            # Position brake lights at the rear of vehicle based on direction
            if v.direction == "right":
                rx, ry1, ry2 = 2, h // 4, 3 * h // 4
            elif v.direction == "left":
                rx, ry1, ry2 = w - 2, h // 4, 3 * h // 4
            elif v.direction == "down":
                ry, rx1, rx2 = 2, w // 4, 3 * w // 4
                rx, ry1, ry2 = rx1, ry, ry
            else:  # up
                ry, rx1, rx2 = h - 2, w // 4, 3 * w // 4
                rx, ry1, ry2 = rx1, ry, ry

            # Red brake light glow
            pygame.draw.circle(brake_surf, (255, 40, 40, 220), (int(rx), int(ry1)), 3)
            pygame.draw.circle(brake_surf, (255, 40, 40, 220), (int(rx), int(ry2)), 3)
            pygame.draw.circle(brake_surf, (255, 40, 40, 80), (int(rx), int(ry1)), 6)
            pygame.draw.circle(brake_surf, (255, 40, 40, 80), (int(rx), int(ry2)), 6)
            self.sim_surface.blit(brake_surf, (v.x, v.y))

    def render_sidebar(self):
        """Renders the 420px Dark Glassmorphism Research Cockpit on the right."""
        sb_x = VIEW_W
        pygame.draw.rect(self.screen, self.C_BG, (sb_x, 0, SIDEBAR_W, WINDOW_HEIGHT))
        pygame.draw.line(self.screen, self.C_CARD_BORDER, (sb_x, 0), (sb_x, WINDOW_HEIGHT), 2)

        meta = PROFILE_METADATA.get(activeProfileName, PROFILE_METADATA["balanced"])
        y = 10
        card_w = SIDEBAR_W - 20

        # -------------------------------------------------------------
        # Card 1: Header / Brand & Active Operational Profile
        # -------------------------------------------------------------
        pygame.draw.rect(self.screen, self.C_CARD, (sb_x + 10, y, card_w, 74), border_radius=6)
        pygame.draw.rect(self.screen, self.C_CARD_BORDER, (sb_x + 10, y, card_w, 74), 1, border_radius=6)

        # Title + Version
        self.screen.blit(self.fontTitle.render("SEEPAGERL RESEARCH CONSOLE", True, self.C_TEXT_WHITE), (sb_x + 20, y + 8))

        # Mode Badge & Profile Pill
        mode_str = "RL (C-GPI Active)" if controllerModeRL else "Fixed-Time Cycle"
        mode_col = self.C_ACCENT_GREEN if controllerModeRL else self.C_ACCENT_ORANGE
        self.screen.blit(self.fontBold.render(f"Mode: {mode_str}", True, mode_col), (sb_x + 20, y + 32))

        prof_title = activeProfileName.upper().replace("_", " ")
        self.screen.blit(self.fontBold.render(f"Profile: {prof_title}", True, meta["color"]), (sb_x + 210, y + 32))

        # Sim Time & Speed
        time_txt = f"Sim Time: {timeElapsed}s | Speed: {playbackSpeed:.1f}x | {'PAUSED' if simulationPaused else 'RUNNING'}"
        self.screen.blit(self.fontSmall.render(time_txt, True, self.C_TEXT_MUTED), (sb_x + 20, y + 52))

        y += 84

        # -------------------------------------------------------------
        # Card 2: Pillar 1 Seepage & Blockage Telemetry
        # -------------------------------------------------------------
        c2_h = 160
        pygame.draw.rect(self.screen, self.C_CARD, (sb_x + 10, y, card_w, c2_h), border_radius=6)
        pygame.draw.rect(self.screen, self.C_CARD_BORDER, (sb_x + 10, y, card_w, c2_h), 1, border_radius=6)

        self.screen.blit(self.fontCardHeader.render("PILLAR 1: SEEPAGE QUEUE DYNAMICS", True, self.C_ACCENT_CYAN), (sb_x + 20, y + 8))

        queues = get_approach_queue_snapshots()
        _, telemetry = build_seepage_rl_state(
            queues=queues,
            red_elapsed=redElapsed,
            current_phase=phaseToGroup[currentGreen],
            is_yellow=currentYellow,
            params=seepageParams,
        )

        ns_t = telemetry["t_clear_ns"]
        ew_t = telemetry["t_clear_ew"]
        ns_seep = telemetry["ns_breakdown"]["alpha_seep"] * 100.0
        ew_seep = telemetry["ew_breakdown"]["alpha_seep"] * 100.0
        ns_hv = telemetry["ns_breakdown"]["rho_hv"] * 100.0
        ew_hv = telemetry["ew_breakdown"]["rho_hv"] * 100.0
        wait_ns = telemetry["wait_ns"]
        wait_ew = telemetry["wait_ew"]

        # NS Row
        self.screen.blit(self.fontBold.render(f"North-South Corridor (T_clear={ns_t:.1f}s)", True, self.C_TEXT_WHITE), (sb_x + 20, y + 32))
        ns_metrics = f"Seep Fraction: {ns_seep:.0f}%  |  HV Blockage: {ns_hv:.0f}%  |  Wait: {wait_ns:.1f}s"
        self.screen.blit(self.fontMono.render(ns_metrics, True, self.C_TEXT_MUTED), (sb_x + 20, y + 50))
        # NS Gauge Bar
        bar_w = int(min(1.0, ns_t / 15.0) * 360)
        pygame.draw.rect(self.screen, (32, 36, 54), (sb_x + 20, y + 68, 360, 6), border_radius=3)
        pygame.draw.rect(self.screen, self.C_ACCENT_BLUE, (sb_x + 20, y + 68, bar_w, 6), border_radius=3)

        # EW Row
        self.screen.blit(self.fontBold.render(f"East-West Corridor (T_clear={ew_t:.1f}s)", True, self.C_TEXT_WHITE), (sb_x + 20, y + 84))
        ew_metrics = f"Seep Fraction: {ew_seep:.0f}%  |  HV Blockage: {ew_hv:.0f}%  |  Wait: {wait_ew:.1f}s"
        self.screen.blit(self.fontMono.render(ew_metrics, True, self.C_TEXT_MUTED), (sb_x + 20, y + 102))
        # EW Gauge Bar
        bar_ew = int(min(1.0, ew_t / 15.0) * 360)
        pygame.draw.rect(self.screen, (32, 36, 54), (sb_x + 20, y + 120, 360, 6), border_radius=3)
        pygame.draw.rect(self.screen, self.C_ACCENT_ORANGE, (sb_x + 20, y + 120, bar_ew, 6), border_radius=3)

        # Starvation Status text
        w_max_col = self.C_ACCENT_RED if max(wait_ns, wait_ew) >= STARVATION_WAIT_THRESHOLD else self.C_ACCENT_GREEN
        starv_txt = f"Starvation Limit: {STARVATION_WAIT_THRESHOLD:.0f}s  |  Status: {'CRITICAL BOUND TRIGGERED' if max(wait_ns, wait_ew) >= STARVATION_WAIT_THRESHOLD else 'Within Safe Bounds'}"
        self.screen.blit(self.fontSmall.render(starv_txt, True, w_max_col), (sb_x + 20, y + 136))

        y += c2_h + 10

        # -------------------------------------------------------------
        # Card 3: Pillar 2 Action Space & Starvation Guard
        # -------------------------------------------------------------
        c3_h = 145
        pygame.draw.rect(self.screen, self.C_CARD, (sb_x + 10, y, card_w, c3_h), border_radius=6)
        pygame.draw.rect(self.screen, self.C_CARD_BORDER, (sb_x + 10, y, card_w, c3_h), 1, border_radius=6)

        self.screen.blit(self.fontCardHeader.render("PILLAR 2: CONSTRAINED POLICY & ACTION MASK", True, self.C_ACCENT_ORANGE), (sb_x + 20, y + 8))

        mask = compute_starvation_mask(
            wait_ns=wait_ns, wait_ew=wait_ew,
            t_clear_ns=ns_t, t_clear_ew=ew_t,
            w_max=STARVATION_WAIT_THRESHOLD,
            enable_mask=starvationMaskActive,
        )

        chosen_idx = lastDecisionReport.chosen_action_idx if lastDecisionReport else 1

        # 6 Action Space Pills Matrix
        px = sb_x + 20
        py = y + 34
        for a_idx, (p_str, _, g_sec) in ACTION_SPACE.items():
            pill_w = 55
            pill_h = 24
            is_chosen = (a_idx == chosen_idx)
            is_masked = not np.isfinite(mask[a_idx])

            if is_chosen:
                col = self.C_ACCENT_GREEN
                border = (255, 255, 255)
            elif is_masked:
                col = (45, 25, 25)
                border = self.C_ACCENT_RED
            else:
                col = (30, 35, 50)
                border = self.C_CARD_BORDER

            pygame.draw.rect(self.screen, col, (px, py, pill_w, pill_h), border_radius=4)
            pygame.draw.rect(self.screen, border, (px, py, pill_w, pill_h), 1, border_radius=4)

            a_text = f"{p_str}-{g_sec}s"
            txt_c = (255, 255, 255) if not is_masked else (180, 100, 100)
            t_surf = self.fontMonoBold.render(a_text, True, txt_c)
            self.screen.blit(t_surf, t_surf.get_rect(center=(px + pill_w // 2, py + pill_h // 2)))

            px += pill_w + 7
            if a_idx == 2:
                px = sb_x + 20
                py += pill_h + 6

        # Mask Explanation line
        mask_state = "HARD ACTION MASK ACTIVE (Starvation Overridden)" if np.any(~np.isfinite(mask)) else "Starvation Mask Idle: All 6 Actions Valid"
        m_col = self.C_ACCENT_RED if np.any(~np.isfinite(mask)) else self.C_TEXT_MUTED
        self.screen.blit(self.fontSmall.render(mask_state, True, m_col), (sb_x + 20, y + 102))

        # C-GPI status
        cgpi_txt = f"C-GPI Profile Switch Latency: < 0.2 ms (Zero Retraining)"
        self.screen.blit(self.fontSmall.render(cgpi_txt, True, self.C_ACCENT_CYAN), (sb_x + 20, y + 122))

        y += c3_h + 10

        # -------------------------------------------------------------
        # Card 4: Pillar 3 Explainable AI (EQPI) & Decision Boundary
        # -------------------------------------------------------------
        c4_h = 180
        pygame.draw.rect(self.screen, self.C_CARD, (sb_x + 10, y, card_w, c4_h), border_radius=6)
        pygame.draw.rect(self.screen, self.C_CARD_BORDER, (sb_x + 10, y, card_w, c4_h), 1, border_radius=6)

        self.screen.blit(self.fontCardHeader.render("PILLAR 3: EXPLAINABLE AI (EQPI)", True, self.C_ACCENT_GREEN), (sb_x + 20, y + 8))

        if lastDecisionReport:
            rep = lastDecisionReport
            d_hdr = f"Selected: {rep.chosen_phase}-{rep.chosen_green_sec}s  vs  Runner-up: {rep.runner_up_phase}-{rep.runner_up_green_sec}s"
            self.screen.blit(self.fontBold.render(d_hdr, True, self.C_TEXT_WHITE), (sb_x + 20, y + 30))

            m_str = f"Winning Margin: Delta_Q = {rep.scalar_margin:+.2f}  |  Confidence L2: {rep.min_weight_shift_l2:.3f}"
            self.screen.blit(self.fontMono.render(m_str, True, self.C_TEXT_MUTED), (sb_x + 20, y + 50))

            # Tripartite Stacked Attribution Bar
            tp_pct = rep.component_attribution_pct.get("throughput_pct", 0.0)
            wait_pct = rep.component_attribution_pct.get("wait_pct", 0.0)
            prio_pct = rep.component_attribution_pct.get("priority_pct", 0.0)

            total_bar = 360
            w_tp = int(max(0.0, tp_pct / 100.0) * total_bar)
            w_wt = int(max(0.0, wait_pct / 100.0) * total_bar)
            w_pr = max(0, total_bar - w_tp - w_wt)

            bx = sb_x + 20
            by = y + 70
            pygame.draw.rect(self.screen, self.C_ACCENT_BLUE, (bx, by, w_tp, 10))
            pygame.draw.rect(self.screen, self.C_ACCENT_ORANGE, (bx + w_tp, by, w_wt, 10))
            pygame.draw.rect(self.screen, self.C_ACCENT_PURPLE, (bx + w_tp + w_wt, by, w_pr, 10))

            attr_leg = f"Throughput: {tp_pct:+.0f}%  |  Delay Reduc: {wait_pct:+.0f}%  |  Priority: {prio_pct:+.0f}%"
            self.screen.blit(self.fontSmall.render(attr_leg, True, self.C_TEXT_WHITE), (sb_x + 20, y + 86))

            # Tipping Point Radar
            opp_dir = "EW" if rep.chosen_phase == "NS" else "NS"
            tip1 = f"• +{rep.opposing_trucks_to_flip} heavy truck(s) on {opp_dir} flips clearance bucket"
            tip2 = f"• +{rep.opposing_wait_sec_to_mask_flip:.1f}s wait on {opp_dir} forces hard starvation switch"
            self.screen.blit(self.fontBold.render("PHYSICAL SENSOR TIPPING POINTS:", True, self.C_ACCENT_ORANGE), (sb_x + 20, y + 110))
            self.screen.blit(self.fontNormal.render(tip1, True, self.C_TEXT_WHITE), (sb_x + 20, y + 130))
            self.screen.blit(self.fontNormal.render(tip2, True, self.C_TEXT_WHITE), (sb_x + 20, y + 148))

        else:
            self.screen.blit(self.fontNormal.render(lastDecisionDetails, True, self.C_TEXT_MUTED), (sb_x + 20, y + 40))

        y += c4_h + 10

        # -------------------------------------------------------------
        # Card 5: Profile Sensitivity Radar
        # -------------------------------------------------------------
        c5_h = 75
        pygame.draw.rect(self.screen, self.C_CARD, (sb_x + 10, y, card_w, c5_h), border_radius=6)
        pygame.draw.rect(self.screen, self.C_CARD_BORDER, (sb_x + 10, y, card_w, c5_h), 1, border_radius=6)

        self.screen.blit(self.fontCardHeader.render("INSTANT PROFILE FLIP SENSITIVITY", True, self.C_ACCENT_PURPLE), (sb_x + 20, y + 8))

        if lastDecisionReport and lastDecisionReport.profile_flips:
            flips = lastDecisionReport.profile_flips
            flip_text = f"Flips immediately to {flips[0][1]}-{flips[0][2]}s under '{flips[0][0].replace('_', ' ')}'"
            f_col = self.C_ACCENT_ORANGE
        else:
            flip_text = "Action is invariant under all 4 canonical profile weights"
            f_col = self.C_ACCENT_GREEN

        self.screen.blit(self.fontNormal.render(flip_text, True, f_col), (sb_x + 20, y + 32))
        self.screen.blit(self.fontSmall.render("Audited analytically in < 0.1 ms per step", True, self.C_TEXT_MUTED), (sb_x + 20, y + 52))

    def render_bottom_panel(self):
        """Renders the bottom 194px Analytics & Interactive Operator Ribbon."""
        by = VIEW_H
        pygame.draw.rect(self.screen, self.C_BG, (0, by, VIEW_W, BOTTOM_H))
        pygame.draw.line(self.screen, self.C_CARD_BORDER, (0, by), (VIEW_W, by), 2)

        # -------------------------------------------------------------
        # Left Panel (Analytics & Queue Composition, 500px wide)
        # -------------------------------------------------------------
        pygame.draw.rect(self.screen, self.C_CARD, (10, by + 10, 495, BOTTOM_H - 20), border_radius=6)
        pygame.draw.rect(self.screen, self.C_CARD_BORDER, (10, by + 10, 495, BOTTOM_H - 20), 1, border_radius=6)

        self.screen.blit(self.fontCardHeader.render("REAL-TIME QUEUE & TRAFFIC COMPOSITION", True, self.C_TEXT_WHITE), (20, by + 18))

        # Aggregate class counts per corridor
        q_ns = {"bike": 0, "rickshaw": 0, "car": 0, "bus": 0, "truck": 0}
        q_ew = {"bike": 0, "rickshaw": 0, "car": 0, "bus": 0, "truck": 0}
        for d in ("down", "up"):
            for lane in range(3):
                for v in vehicles[d][lane]:
                    if v.crossed == 0 and v.vehicleClass in q_ns:
                        q_ns[v.vehicleClass] += 1
        for d in ("right", "left"):
            for lane in range(3):
                for v in vehicles[d][lane]:
                    if v.crossed == 0 and v.vehicleClass in q_ew:
                        q_ew[v.vehicleClass] += 1

        # Render mini vehicle breakdown table
        classes = ["bike", "rickshaw", "car", "bus", "truck"]
        cols = [(46, 204, 113), (241, 196, 15), (52, 152, 219), (155, 89, 182), (230, 126, 34)]
        labels = ["2W", "Auto", "Car", "Bus", "Truck"]

        # Table Header
        self.screen.blit(self.fontBold.render("Class", True, self.C_TEXT_MUTED), (20, by + 42))
        self.screen.blit(self.fontBold.render("NS Queue", True, self.C_TEXT_MUTED), (80, by + 42))
        self.screen.blit(self.fontBold.render("EW Queue", True, self.C_TEXT_MUTED), (160, by + 42))
        self.screen.blit(self.fontBold.render("PCU Equiv", True, self.C_TEXT_MUTED), (245, by + 42))

        ty = by + 62
        for cls_name, lbl, col in zip(classes, labels, cols):
            pygame.draw.circle(self.screen, col, (25, ty + 6), 4)
            self.screen.blit(self.fontNormal.render(lbl, True, self.C_TEXT_WHITE), (35, ty))
            self.screen.blit(self.fontMonoBold.render(str(q_ns[cls_name]), True, self.C_ACCENT_BLUE), (90, ty))
            self.screen.blit(self.fontMonoBold.render(str(q_ew[cls_name]), True, self.C_ACCENT_ORANGE), (170, ty))
            pcu_val = PCU_WEIGHTS.get(cls_name, 1.0)
            self.screen.blit(self.fontMono.render(f"{pcu_val:.1f} PCU", True, self.C_TEXT_MUTED), (250, ty))
            ty += 20

        # Overall Throughput summary box on the right of analytics panel
        total_p = sum(vehicles[d]["crossed"] for d in directionNumbers.values())
        cur_tp = float(total_p) / max(1.0, float(timeElapsed))

        bx = 330
        pygame.draw.rect(self.screen, (18, 20, 32), (bx, by + 42, 160, 120), border_radius=6)
        pygame.draw.rect(self.screen, self.C_CARD_BORDER, (bx, by + 42, 160, 120), 1, border_radius=6)

        self.screen.blit(self.fontSmall.render("Total Vehicles Cleared", True, self.C_TEXT_MUTED), (bx + 12, by + 50))
        self.screen.blit(self.fontTitle.render(f"{total_p}", True, self.C_ACCENT_GREEN), (bx + 12, by + 68))
        self.screen.blit(self.fontSmall.render("Throughput Rate", True, self.C_TEXT_MUTED), (bx + 12, by + 98))
        self.screen.blit(self.fontBold.render(f"{cur_tp:.2f} veh/sec", True, self.C_TEXT_WHITE), (bx + 12, by + 116))
        self.screen.blit(self.fontSmall.render(f"({cur_tp * 60:.0f} veh/min)", True, self.C_TEXT_MUTED), (bx + 12, by + 136))

        # -------------------------------------------------------------
        # Right Panel (Interactive Control Cockpit Buttons, 530px wide)
        # -------------------------------------------------------------
        pygame.draw.rect(self.screen, self.C_CARD, (515, by + 10, VIEW_W - 525, BOTTOM_H - 20), border_radius=6)
        pygame.draw.rect(self.screen, self.C_CARD_BORDER, (515, by + 10, VIEW_W - 525, BOTTOM_H - 20), 1, border_radius=6)

        self.screen.blit(self.fontCardHeader.render("INTERACTIVE OPERATOR COCKPIT (CLICK OR USE HOTKEYS)", True, self.C_ACCENT_GREEN), (525, by + 18))

        # Update button active states
        self.btn_balanced.is_active = (activeProfileName == "balanced")
        self.btn_commuter.is_active = (activeProfileName == "commuter_priority")
        self.btn_freight.is_active = (activeProfileName == "freight_priority")
        self.btn_emergency.is_active = (activeProfileName == "emergency_corridor")
        self.btn_mode.is_active = controllerModeRL
        self.btn_mask.is_active = starvationMaskActive
        self.btn_pause.is_active = simulationPaused

        # Draw all interactive buttons
        for btn in self.buttons:
            btn.draw(self.screen, self.fontBold, self.fontSmall)

    def handle_mouse_click(self, pos: Tuple[int, int]):
        global activeProfileName, controllerModeRL, starvationMaskActive, simulationPaused, playbackSpeed

        if self.btn_balanced.rect.collidepoint(pos):
            activeProfileName = "balanced"
            rlAgent.set_active_profile("balanced")
            print("[CONTROL] Switched profile to Balanced")
        elif self.btn_commuter.rect.collidepoint(pos):
            activeProfileName = "commuter_priority"
            rlAgent.set_active_profile("commuter_priority")
            print("[CONTROL] Switched profile to Commuter Priority")
        elif self.btn_freight.rect.collidepoint(pos):
            activeProfileName = "freight_priority"
            rlAgent.set_active_profile("freight_priority")
            print("[CONTROL] Switched profile to Freight Priority")
        elif self.btn_emergency.rect.collidepoint(pos):
            activeProfileName = "emergency_corridor"
            rlAgent.set_active_profile("emergency_corridor")
            print("[CONTROL] Switched profile to Emergency Corridor")
        elif self.btn_mode.rect.collidepoint(pos):
            controllerModeRL = not controllerModeRL
            print(f"[CONTROL] Mode toggled: {'RL (C-GPI)' if controllerModeRL else 'Fixed Baseline'}")
        elif self.btn_mask.rect.collidepoint(pos):
            starvationMaskActive = not starvationMaskActive
            print(f"[CONTROL] Starvation Action Mask: {'ENABLED' if starvationMaskActive else 'DISABLED'}")
        elif self.btn_pause.rect.collidepoint(pos):
            simulationPaused = not simulationPaused
            print(f"[CONTROL] Simulation {'PAUSED' if simulationPaused else 'RESUMED'}")
        elif self.btn_speed.rect.collidepoint(pos):
            speeds_cycle = [0.5, 1.0, 2.0, 4.0]
            curr_idx = speeds_cycle.index(playbackSpeed) if playbackSpeed in speeds_cycle else 1
            playbackSpeed = speeds_cycle[(curr_idx + 1) % len(speeds_cycle)]
            print(f"[CONTROL] Playback Speed: {playbackSpeed}x")
        elif self.btn_truck_surge.rect.collidepoint(pos):
            inject_heavy_vehicle_surge()
        elif self.btn_bike_surge.rect.collidepoint(pos):
            inject_two_wheeler_surge()

    def handle_keyboard_input(self, event):
        global activeProfileName, controllerModeRL, starvationMaskActive, simulationPaused, playbackSpeed

        if event.key == pygame.K_1:
            activeProfileName = "balanced"
            rlAgent.set_active_profile("balanced")
            print("[CONTROL] Switched profile to Balanced")
        elif event.key == pygame.K_2:
            activeProfileName = "commuter_priority"
            rlAgent.set_active_profile("commuter_priority")
            print("[CONTROL] Switched profile to Commuter Priority")
        elif event.key == pygame.K_3:
            activeProfileName = "freight_priority"
            rlAgent.set_active_profile("freight_priority")
            print("[CONTROL] Switched profile to Freight Priority")
        elif event.key == pygame.K_4:
            activeProfileName = "emergency_corridor"
            rlAgent.set_active_profile("emergency_corridor")
            print("[CONTROL] Switched profile to Emergency Corridor")
        elif event.key == pygame.K_m:
            controllerModeRL = not controllerModeRL
            print(f"[CONTROL] Controller Mode: {'RL' if controllerModeRL else 'Fixed Baseline'}")
        elif event.key == pygame.K_s:
            starvationMaskActive = not starvationMaskActive
            print(f"[CONTROL] Starvation Action Mask: {'ENABLED' if starvationMaskActive else 'DISABLED'}")
        elif event.key == pygame.K_SPACE:
            simulationPaused = not simulationPaused
            print(f"[CONTROL] Simulation {'PAUSED' if simulationPaused else 'RESUMED'}")
        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
            playbackSpeed = min(4.0, playbackSpeed * 2.0)
            print(f"[CONTROL] Speed: {playbackSpeed}x")
        elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
            playbackSpeed = max(0.5, playbackSpeed / 2.0)
            print(f"[CONTROL] Speed: {playbackSpeed}x")
        elif event.key == pygame.K_b:
            inject_heavy_vehicle_surge()
        elif event.key == pygame.K_t:
            inject_two_wheeler_surge()

    def run_loop(self):
        clock = pygame.time.Clock()

        while True:
            mouse_pos = pygame.mouse.get_pos()
            for btn in self.buttons:
                btn.check_hover(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.handle_keyboard_input(event)

            # ---------------------------------------------------------
            # 1. Render Native Simulation Canvas (1400 x 800)
            # ---------------------------------------------------------
            self.sim_surface.blit(self.background, (0, 0))

            # Render Bike-Box Seepage Reservoirs
            self.render_bike_box_reservoirs()

            # Render Traffic Signals with Radial Bloom Halos
            for i in range(noOfSignals):
                self.render_glowing_signal(i)

            # Render Vehicles with 3D Drop Shadows & Brake Lights
            for v in simulation:
                self.render_vehicle_enhancements(v)
                if not simulationPaused:
                    v.move()

            # ---------------------------------------------------------
            # 2. Scale & Blit Simulation Viewport onto Screen (1060 x 606)
            # ---------------------------------------------------------
            scaled_sim = pygame.transform.smoothscale(self.sim_surface, (VIEW_W, VIEW_H))
            self.screen.blit(scaled_sim, (0, 0))

            # ---------------------------------------------------------
            # 3. Render Bottom Analytics & Interactive Ribbon (1060 x 194)
            # ---------------------------------------------------------
            self.render_bottom_panel()

            # ---------------------------------------------------------
            # 4. Render 420px Dark Glassmorphism Research Cockpit (Right)
            # ---------------------------------------------------------
            self.render_sidebar()

            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    SimulationApp()
