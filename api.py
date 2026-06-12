import os
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

from src.tracker import FootballTracker
from src.analytics.physics import FieldTransformer
from src.analytics.skill_evaluator import SkillEvaluator


class ChatQuery(BaseModel):
    question: str

class VideoUrlRequest(BaseModel):
    video_url: str


load_dotenv()
app = FastAPI(title="Football AI Analyzer API")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                  )

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/analyze-video-and-recommendation")
async def analyze_video_url(request: VideoUrlRequest):
    cap = None
    try:
        tracker = FootballTracker()
        transformer = FieldTransformer()

        cap = cv2.VideoCapture(request.video_url)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video URL.")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0 and total_frames < 30:
            raise HTTPException(status_code=400, detail="Video too short for analysis (min 1 second).")

        skip_rate = max(1, round(original_fps / 3))
        effective_fps = original_fps / skip_rate

        # Calibration
        max_calibration_frames = min(60, max(30, int(total_frames * 0.10)))
        calibrated = False
        calibration_attempts = 0

        while not calibrated and calibration_attempts < max_calibration_frames:
            ret, frame = cap.read()
            if not ret:
                break
            calibrated = transformer.auto_calibrate(frame)
            calibration_attempts += 1

        if not calibrated:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not detect a football field after scanning "
                    f"{calibration_attempts} frames. "
                    "Please ensure the video clearly shows a grass football field."
                )
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        match_timeline = []
        previous_positions = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_rate != 0:
                frame_idx += 1
                continue

            h_orig, w_orig = frame.shape[:2]
            target_width = 640
            target_height = int(h_orig * (target_width / w_orig))
            processed_frame = cv2.resize(frame, (target_width, target_height))
            scale_ratio = w_orig / target_width

            data = tracker.detect_and_track(processed_frame)

            if data:
                frame_record = {"frame_id": frame_idx, "ball": None, "players": {}}

                if data["ball"]:
                    bx1, by1, bx2, by2 = data["ball"]["coords"]
                    real_b = transformer.transform_point(
                        ((bx1 + bx2) / 2) * scale_ratio,
                        by2 * scale_ratio
                    )
                    frame_record["ball"] = {"position": real_b.tolist()}

                for player in data["players"]:
                    t_id = player["track_id"]
                    if t_id == -1:
                        continue

                    px1, py1, px2, py2 = player["coords"]
                    real_p = transformer.transform_point(
                        ((px1 + px2) / 2) * scale_ratio,
                        py2 * scale_ratio
                    )

                    speed = 0.0
                    if t_id in previous_positions:
                        speed = transformer.calculate_speed(
                            previous_positions[t_id], real_p, effective_fps
                        )

                    previous_positions[t_id] = real_p
                    frame_record["players"][t_id] = {
                        "position": real_p.tolist(),
                        "speed_kmh": speed
                    }

                match_timeline.append(frame_record)

            frame_idx += 1

        cap.release()

        if not match_timeline:
            raise HTTPException(status_code=400, detail="No players detected in the video.")

        if len(match_timeline) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient tracking data: only {len(match_timeline)} frames processed."
            )

        evaluator = SkillEvaluator(fps=effective_fps)
        target_id = evaluator.get_target_player_id(match_timeline)

        if target_id is None:
            raise HTTPException(status_code=400, detail="No consistent player found for analysis.")

        final_skills = evaluator.evaluate_player(match_timeline, target_id)

        if all(v == 0 for v in final_skills.values()):
            raise HTTPException(
                status_code=422,
                detail="Analysis produced zero scores — video may not contain clear football gameplay."
            )

        prompt = f"""
        Analyze this youth player's performance metrics (scored 0-100) and provide a high-level scout report.
        
        Current Player Metrics:
        - Speed: {final_skills.get('speed', 0)}
        - Shooting: {final_skills.get('shooting', 0)}
        - Passing: {final_skills.get('passing', 0)}
        - Positioning: {final_skills.get('positioning', 0)}
        - Reaction: {final_skills.get('reaction', 0)}
        
        Report Requirements:
        1. Format using Markdown.
        2. Provide 5 actionable tips for improvement.
        3. Be professional and concise.
        """

        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an elite European football academy scout."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            ai_advice = completion.choices[0].message.content
        except Exception:
            ai_advice = "AI Scout report generation failed, but your physical metrics are ready."

        return {"skills": final_skills, "advice": ai_advice}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()


# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

@app.post("/football-chat")
async def football_chatbot(query: ChatQuery):
    try:
        system_instructions = """
        You are 'ProScout AI', a specialized football assistant. 
        Answer ONLY football-related questions (tactics, history, skills, rules). 
        If asked about anything else, politely decline. 
        Keep answers professional, insightful, and formatted in Markdown.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": query.question}
            ],
            temperature=0.7,
            max_tokens=600
        )

        return {"answer": completion.choices[0].message.content}

    except Exception as e:
        print(f"Chatbot Error: {str(e)}")
        raise HTTPException(status_code=500, detail="The football expert is unavailable.")