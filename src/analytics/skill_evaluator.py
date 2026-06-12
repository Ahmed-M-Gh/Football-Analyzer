import numpy as np

class SkillEvaluator:
    def __init__(self, fps):
        self.fps = fps
        
        # Updated realistic thresholds
        self.max_youth_speed = 32.0
        self.max_youth_shot = 85.0 
        
        self.ball_proximity_threshold = 6.0
        self.shot_speed_min = 30.0       
        self.pass_speed_min = 10.0 

    def get_target_player_id(self, match_timeline):
        appearance_counts = {}
        for frame in match_timeline:
            for t_id in frame['players'].keys():
                appearance_counts[t_id] = appearance_counts.get(t_id, 0) + 1

        if not appearance_counts:
            return None

        return max(appearance_counts, key=appearance_counts.get) # type: ignore

    def evaluate_player(self, match_timeline, target_id):
        raw_speeds = []
        ball_distances = []
        shot_powers = []
        pass_events = 0

        last_touch_frame = -100 

        # 1. Collect speeds
        for frame in match_timeline:
            if target_id in frame['players']:
                raw_speeds.append(frame['players'][target_id]['speed_kmh'])

        if len(raw_speeds) < 5:
            return self._empty_stats()

        # 2. Smooth speed
        window = 5
        smoothed_speeds = np.convolve(raw_speeds, np.ones(window)/window, mode='valid')

        accelerations = [
            abs(smoothed_speeds[i] - smoothed_speeds[i-1]) * (self.fps / 3.6)
            for i in range(1, len(smoothed_speeds))
        ]

        # 3. Analyze interactions
        for i in range(1, len(match_timeline)):
            frame = match_timeline[i]
            prev_frame = match_timeline[i-1]

            if target_id not in frame['players']:
                continue
            if not frame['ball'] or not prev_frame['ball']:
                continue

            p_pos = np.array(frame['players'][target_id]['position'])
            b_pos = np.array(frame['ball']['position'])
            prev_b_pos = np.array(prev_frame['ball']['position'])

            dist = np.linalg.norm(p_pos - b_pos)
            ball_distances.append(dist)

            # touch detection
            if dist < self.ball_proximity_threshold:

                # prevent duplicate detection
                if i - last_touch_frame < 5:
                    continue

                last_touch_frame = i

                # ball movement
                ball_move = np.linalg.norm(b_pos - prev_b_pos)
                b_speed = ball_move * self.fps * 3.6

                # direction vector
                movement_vector = b_pos - prev_b_pos
                player_vector = b_pos - p_pos

                dot_product = np.dot(movement_vector, player_vector)

                
                if b_speed > self.shot_speed_min and dot_product > 0:
                    shot_powers.append(b_speed)

                elif b_speed > self.pass_speed_min:
                    pass_events += 1

        # 4. Scores

        # Speed
        top_speed = float(np.percentile(smoothed_speeds, 90))
        speed_score = min(100, (top_speed / self.max_youth_speed) * 100)

        # Shooting
        max_shot = float(max(shot_powers)) if shot_powers else 0.0
        shot_score = min(100, (max_shot / self.max_youth_shot) * 100)

        # Passing
        pass_score = min(100, pass_events * 10)

        # Positioning
        min_dist = np.min(ball_distances) if ball_distances else 20.0
        pos_score = max(0, 100 - (min_dist * 5))

        # Reaction
        top_accel = np.percentile(accelerations, 95) if accelerations else 0.0
        react_score = min(100, (top_accel / 6.0) * 100)

        return {
            "speed": int(round(speed_score)),
            "shooting": int(round(shot_score)),
            "passing": int(round(pass_score)),
            "positioning": int(round(pos_score)),
            "reaction": int(round(react_score)),
        }

    def _empty_stats(self):
        return {
            "speed": 0,
            "shooting": 0,
            "passing": 0,
            "positioning": 0,
            "reaction": 0,
        }